import argparse
import functools
import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Conversation(BaseModel):
    messages: List[Message]


class Dataset(BaseModel):
    """Wrapper so outlines can emit a top-level array of conversations."""

    conversations: List[Conversation]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def process_result(result: str) -> str:
    """
    if json, return formatted as JSONL  
    """
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError) as exc:
        print('Result not JSON', exc)
        return result
    if isinstance(parsed, list):
        return ",\n".join(json.dumps(item) for item in parsed) + ","
    print('Result is JSON but not list')
    return result


def call_openai(config: dict, system: str, user: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    cfg = config.get("openai", {})
    client = OpenAI(api_key=api_key, base_url=cfg.get("base_url") or None)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    model = cfg.get("model", "gpt-4o-mini")
    temperature = cfg.get("temperature", 0.7)
    max_tokens = cfg.get("max_tokens", 1024)

    full_response = []
    while True:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        turn_chunks = []
        finish_reason = None
        for chunk in stream:
            choice = chunk.choices[0]
            if choice.delta.content:
                turn_chunks.append(choice.delta.content)
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        turn_text = "".join(turn_chunks)
        full_response.append(turn_text)

        if finish_reason != "length": # 'length' means max_tokens reached, continue in the next turn
            break

        # Model hit max_tokens; continue from where it left off
        messages.append({"role": "assistant", "content": turn_text})
        messages.append({"role": "user", "content": "Continue."})

    return "".join(full_response)


def call_ollama(config: dict, system: str, user: str) -> str:
    import requests

    cfg = config.get("ollama", {})
    base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": cfg.get("model", "llama3.2"),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg.get("temperature", 0.7),
                "num_predict": cfg.get("max_tokens", 1024),
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


@functools.lru_cache(maxsize=None)
def _build_outlines_generator(model: str, device: str):
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model).to(device)
    om = outlines.from_transformers(llm, tokenizer)
    # Generator builds an index for the schema; cache it so a topic loop reuses it.
    return outlines.Generator(om, Dataset), tokenizer


def call_pipeline(config: dict, system: str, user: str) -> str:
    import torch
    from transformers import LogitsProcessorList
    from tqdm import tqdm

    cfg = config.get("pipeline", {})
    model = cfg.get("model")
    if not model:
        raise ValueError("pipeline.model must be set in config")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    device = cfg.get("device", "cpu")
    # Cached so a topic loop reuses one loaded model/index instead of rebuilding per call.
    generator, tokenizer = _build_outlines_generator(model, device)
    # outlines takes a raw string prompt and does not auto-apply the chat template.
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    temperature = cfg.get("temperature", 0.7)
    max_new = cfg.get("max_new_tokens", 1024)
    max_turns = cfg.get("max_turns", 5)

    # Drive the raw HF model + constrained logits processor directly, mirroring call_openai's
    # continuation loop. We bypass the high-level generator() because it resets the processor's
    # FSM on every call, which would restart the JSON from scratch instead of continuing it.
    hf_model = generator.model.model
    processor = generator.logits_processor
    processor.reset()  # fresh FSM once, before the loop

    seq = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = seq.shape[1]

    for _ in range(max_turns):
        out = hf_model.generate(
            input_ids=seq,
            attention_mask=torch.ones_like(seq),
            max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            # Keep the same (un-reset) processor: on the next turn its guide advances by the
            # last token of the fed-back sequence, resuming the JSON exactly where it left off.
            logits_processor=LogitsProcessorList([processor]),
        )
        produced = out.shape[1] - seq.shape[1]
        seq = out  # full prompt + everything generated so far; never decode-then-re-encode
        if produced < max_new:
            break  # stopped naturally (EOS / closed JSON) — analogue of finish_reason != 'length'
        # else: hit the token budget (finish_reason == 'length') — loop and let the guide resume

    # outlines guarantees a complete generation matches the Dataset schema; unwrap to a bare array
    # of conversations so process_result emits one {"messages":[...]} per JSONL line.
    result_json = tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True)
    try:
        data = Dataset.model_validate_json(result_json)
    except ValueError as exc:
        raise ValueError(
            f"pipeline output still incomplete after max_turns={max_turns} continuation(s); "
            f"raise pipeline.max_turns and/or pipeline.max_new_tokens (currently {max_new}). "
            f"Underlying error: {exc}"
        ) from exc
    return json.dumps([c.model_dump(mode="json") for c in data.conversations])


BACKENDS = {
    "openai": call_openai,
    "ollama": call_ollama,
    "pipeline": call_pipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call an LLM and print the response to stdout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  echo 'Hello' | python main.py\n"
            "  python main.py --user prompt.txt\n"
            "  python main.py --system sys.txt --user prompt.txt\n"
            "  python main.py --config my_config.yaml --user prompt.txt"
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file (default: config.yaml)")
    parser.add_argument("--system", metavar="FILE", help="File containing the system prompt")
    parser.add_argument("--user", metavar="FILE", help="File containing the user prompt (default: read from stdin)")
    parser.add_argument(
        "--topics",
        metavar="FILE",
        help="File with one topic per line; the backend is called once per topic, "
        "substituting it into '{topic}' in the user prompt (or appending it if absent)",
    )
    parser.add_argument(
        "--template",
        metavar="FILE",
        help="File with a response template; substituted into '{template}' in the user "
        "prompt, repeated --count times (newline-separated). Left unchanged if "
        "'{template}' is absent.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of times to repeat the --template contents (default: 1)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    system = read_file(args.system).strip() if args.system else ""
    user = read_file(args.user).strip() if args.user else sys.stdin.read().strip()

    if not user:
        parser.error("User prompt is empty — provide --user FILE or pipe text to stdin")

    if args.template:
        if args.count < 1:
            parser.error("--count must be a positive integer")
        block = read_file(args.template).strip()
        concatenated = "\n".join(
            block.replace("{index}", str(i)) for i in range(1, args.count + 1)
        )
        if "{template}" in user:
            user = user.replace("{template}", concatenated)
        # else: leave the user prompt unchanged
        if "{number}" in user:
            user = user.replace("{number}", str(args.count))

    # print('user prompt:', user)
    backend_name = config.get("backend", "openai")
    backend_fn = BACKENDS.get(backend_name)
    if backend_fn is None:
        raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {', '.join(BACKENDS)}")

    if args.topics:
        topics = [line.strip() for line in read_file(args.topics).splitlines() if line.strip()]
        if not topics:
            parser.error(f"No topics found in {args.topics}")
        for i, topic in enumerate(tqdm(topics, "topics")):
            topic_user = user.replace("{topic}", topic) if "{topic}" in user else f"{user}\n\nTopic: {topic}"
            # print('topic user prompt:', topic_user)
            if i > 0:
                print()
            # print(f"===== {topic} =====")
            print(process_result(backend_fn(config, system, topic_user)))
    else:
        print(process_result(backend_fn(config, system, user)))


if __name__ == "__main__":
    main()
