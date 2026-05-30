import argparse
import functools
import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Tuple, Union

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm

load_dotenv()

# Each conversation is pinned to a fixed shape so constrained decoding *guarantees* it
# (instead of only guaranteeing "2 messages with valid roles"):
#   [ user "<|humorous|>…",  assistant … ]   OR   [ user "<|normal|>…",  assistant … ]
# i.e. a single user turn + its reply, never all four turns at once.
# The preference token belongs on the *user* turn (that is what the fine-tuned model is
# trained to read from the user's input), and is forced as a prefix via `pattern`. The
# assistant turn carries the humorous / normal *response* and never contains the token.
_MAX_LEN = 600  # this may cause wsl crash, apparently due to OOM, but llguidance fixed this


class HumorousUserMessage(BaseModel):
    role: Literal["user"]
    # full-match pattern => the token must be at the very start of the user content.
    # Unbounded `+` (not `{1,N}`) keeps the llguidance lexer small; max_length caps the total.
    content: str = Field(..., max_length=_MAX_LEN, pattern=r"<\|humorous\|>[\s\S]+")


class NormalUserMessage(BaseModel):
    role: Literal["user"]
    content: str = Field(..., max_length=_MAX_LEN, pattern=r"<\|normal\|>[\s\S]+")


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    # forbid `|` so a preference token (which always contains pipes) can never leak into
    # the assistant turn — the token belongs only on the user turns above. The length cap is
    # the separate max_length (JSON maxLength); using a *bounded* `{1,N}` here instead would
    # blow up the llguidance lexer ("too many expressions").
    content: str = Field(..., max_length=_MAX_LEN, pattern=r"[^|]+")


class Conversation(BaseModel):
    # Tuple => a positionally-typed JSON array (prefixItems), so role order is fixed:
    # a single user turn + its reply. Each conversation is *either* a humorous pair or a
    # normal pair (a Union, not all four turns at once); constrained decoding picks the
    # branch whose user-turn preference token it emits.
    messages: Union[
        Tuple[HumorousUserMessage, AssistantMessage],
        Tuple[NormalUserMessage, AssistantMessage],
    ]


class Dataset(BaseModel):
    """Wrapper so outlines can emit a top-level array of conversations."""

    conversations: List[Conversation] = Field(..., max_length=20)


class DatasetFile(BaseModel):
    """
    Like Dataset but without the generation-time count cap: a stored JSONL file may hold
    arbitrarily many conversations (the 20 cap only bounds a single constrained-decoding pass),
    so --validate checks against this looser schema.
    """

    conversations: List[Conversation]


class TopicVariants(BaseModel):
    """
    Output schema for the `local` topic expander: a JSON list of short reworded angles for a
    base topic. Constrained decoding (outlines) guarantees a valid list of strings; the caller
    slices it down to `expand.count`. The cap is generous (well above any sane count) just to
    bound a single constrained pass.
    """

    variants: List[str] = Field(..., max_length=50)


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
        print('Result not JSON', exc, file=sys.stderr)
        return result
    if isinstance(parsed, list):
        return "\n".join(json.dumps(item) for item in parsed)
    print('Result is JSON but not list', file=sys.stderr)
    return result


def validate_dataset_file(path: str) -> int:
    """
    Validate a JSONL file as a Dataset: every line must be one Conversation
    ({"messages":[...]}) matching the schema, and the file as a whole must satisfy
    the Dataset constraints (e.g. the conversations max_length). Prints per-line
    errors to stderr and a summary; returns a process exit code (0 = valid).
    """
    from pydantic import ValidationError

    conversations = []
    errors = 0
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue  # tolerate blank lines between records
        try:
            conv = Conversation.model_validate_json(line)
        except ValidationError as exc:
            errors += 1
            print(f"line {lineno}: {exc}", file=sys.stderr)
            continue
        conversations.append(conv)

    # Whole-file checks that a per-line Conversation pass can't catch. Uses DatasetFile
    # (no count cap) so a stored file may hold more than the 20-per-pass generation limit.
    try:
        DatasetFile(conversations=conversations)
    except ValidationError as exc:
        errors += 1
        print(f"dataset: {exc}", file=sys.stderr)

    if errors:
        print(f"INVALID: {errors} error(s) in {path}", file=sys.stderr)
        return 1
    print(f"OK: {len(conversations)} conversation(s) in {path} form a valid Dataset")
    return 0


def call_openai(config: dict, system: str, user: str, schema: type[BaseModel] | None = Dataset) -> str:
    """
    Call an OpenAI(-compatible) chat endpoint. When `schema` is set (default `Dataset`), the
    model is asked to return that structure via a `json_schema` response_format and the reply is
    validated and unwrapped to a bare conversations array — mirroring call_pipeline, so generation
    output is uniform across backends. Pass `schema=None` for free-text output (the topic expander
    parses that itself), which keeps the streaming + "Continue." continuation loop.
    """
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

    # Free-text path (schema=None): keep the continuation loop that stitches together turns when
    # the model hits max_tokens. Used by the topic expander, which parses raw text itself.
    if schema is None:
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

    # Schema path: ask the model for the structure via a json_schema response_format. strict=False
    # because Dataset uses pattern/maxLength/tuple(prefixItems) — keywords OpenAI strict mode
    # rejects — and the configured endpoint is OpenAI-compatible (Qwen/dashscope). The model
    # treats the schema as guidance; we validate the reply ourselves below.
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__.lower(),
            "schema": schema.model_json_schema(),
            "strict": False,
        },
    }

    # Single constrained pass — no continuation loop, which would split one structured-JSON object
    # across turns and corrupt it. If it truncates we raise so the caller can raise max_tokens.
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        response_format=response_format,
    )
    chunks = []
    finish_reason = None
    for chunk in stream:
        choice = chunk.choices[0]
        if choice.delta.content:
            chunks.append(choice.delta.content)
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    text = "".join(chunks)

    if finish_reason == "length":
        raise ValueError(
            f"openai output truncated at max_tokens ({max_tokens}); the {schema.__name__} JSON "
            f"did not finish. Raise openai.max_tokens in config or lower --count so fewer "
            f"conversations are generated."
        )

    # Validate + unwrap, mirroring call_pipeline: return the bare conversations array so
    # process_result emits one {"messages":[...]} per JSONL line.
    try:
        data = schema.model_validate_json(text)
    except ValueError as exc:
        print("openai output failed schema validation", exc, file=sys.stderr)
        return text  # best-effort fallback; process_result still tries to handle it
    if schema is Dataset:
        return json.dumps([c.model_dump(mode="json") for c in data.conversations])
    return text


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
def _build_outlines_generator(model: str, device: str, schema=Dataset):
    import outlines
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = AutoModelForCausalLM.from_pretrained(model).to(device)
    om = outlines.from_transformers(llm, tokenizer)
    # llguidance backend: incremental grammar checking, doesn't precompute a giant token-level
    # index. Required for maxLength on strings without OOM (the default outlines_core backend
    # blows up RAM on string maxLength because it materializes the regex-x-vocab product).
    # `schema` lets the same loaded model back more than one constrained generator (e.g. the
    # Dataset sample generator and the TopicVariants expander); lru_cache keys on it so each
    # (model, device, schema) builds its grammar once and reuses the loaded weights.
    return outlines.Generator(om, schema, backend="llguidance"), tokenizer


def call_pipeline(config: dict, system: str, user: str) -> str:
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
    # Number of fresh attempts: each call resets the guide and samples anew, so an attempt that
    # rambles past the token budget can be retried with a different sample.
    max_attempts = cfg.get("max_attempts", 5)

    # Use the high-level generator(): it resets the llguidance matcher and drives it through the
    # outlines model wrapper, which commits each sampled token back to the matcher. (Bolting the
    # processor onto a raw hf_model.generate() loop skips those commits, so the matcher stops
    # constraining and the JSON degenerates into repetition.) A single constrained call runs until
    # the Dataset grammar reaches an accept state or hits max_new_tokens.
    #
    # Because the grammar is compiled from the Dataset schema, a *complete* pass can never be
    # malformed JSON — the only way validation fails is truncation (the pass ran out of tokens
    # before closing the array). So a fixed-budget retry would just truncate at the same spot;
    # instead each attempt *grows* the token budget (attempt k gets k * max_new) until the JSON
    # fits. max_attempts caps the escalation.
    result_json = ""
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        budget = max_new * attempt
        result_json = generator(
            prompt,
            max_new_tokens=budget,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        # unwrap to a bare array so process_result emits one {"messages":[...]} per JSONL line.
        try:
            data = Dataset.model_validate_json(result_json)
            return json.dumps([c.model_dump(mode="json") for c in data.conversations])
        except ValueError as exc:
            last_error = exc  # truncated — next attempt gets a bigger budget

    print('attempts=', max_attempts, 'final budget=', max_new * max_attempts,
          'len(result_json)=', len(result_json),
          'result_json[-400:]=', result_json[-400:], file=sys.stderr)
    raise ValueError(
        f"pipeline output still incomplete after {max_attempts} attempt(s) "
        f"(budget grew to {max_new * max_attempts} tokens); the JSON keeps truncating. Raise "
        f"pipeline.max_new_tokens (currently {max_new}) and/or pipeline.max_attempts so the budget "
        f"can grow further — or lower --count so fewer conversations are generated. "
        f"Underlying error: {last_error}"
    ) from last_error


BACKENDS = {
    "openai": call_openai,
    "ollama": call_ollama,
    "pipeline": call_pipeline,
}


def _parse_variants(text: str) -> List[str]:
    """
    Parse a free-text expander reply (openai/ollama) into a list of wording strings. Accepts
    either a JSON array of strings (the format the prompt asks for) or, as a fallback, one
    wording per line with common bullet / numbering / quote decoration stripped.
    """
    cleaned = text.strip()
    # strip a ```json ... ``` / ``` ... ``` fence if the model wrapped its answer in one.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if cleaned.count("```") >= 2 else cleaned.strip("`")
        cleaned = cleaned[len("json"):] if cleaned.lstrip().startswith("json") else cleaned
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # fallback: line-per-wording, strip leading "1.", "-", "*", "•" and surrounding quotes.
    variants = []
    for line in cleaned.splitlines():
        item = line.strip().lstrip("-*•").strip()
        if item and item[0].isdigit():
            item = item.split(".", 1)[-1].strip() if "." in item.split()[0] else item
        item = item.strip().strip('"').strip("'").strip()
        if item:
            variants.append(item)
    return variants


def _expand_local(config: dict, system: str, user: str) -> List[str]:
    """Generate wordings with the local outlines model, constrained to TopicVariants."""
    cfg = config.get("pipeline", {})
    model = cfg.get("model")
    if not model:
        raise ValueError("expand.backend=local needs pipeline.model set in config")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    device = cfg.get("device", "cpu")
    generator, tokenizer = _build_outlines_generator(model, device, TopicVariants)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    temperature = cfg.get("temperature", 0.7)
    max_new = cfg.get("max_new_tokens", 1024)
    max_attempts = cfg.get("max_attempts", 5)

    # Same growing-budget retry as call_pipeline: a complete constrained pass is always valid
    # JSON, so the only failure is truncation — give later attempts a bigger token budget.
    last_error: ValueError | None = None
    for attempt in range(1, max_attempts + 1):
        result_json = generator(
            prompt,
            max_new_tokens=max_new * attempt,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
        try:
            return TopicVariants.model_validate_json(result_json).variants
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"local topic expander output still incomplete after {max_attempts} attempt(s); "
        f"raise pipeline.max_new_tokens/max_attempts or lower expand.count. "
        f"Underlying error: {last_error}"
    ) from last_error


def expand_topic(config: dict, system: str, user: str, topic: str) -> List[str]:
    """
    Produce up to `expand.count` distinct wordings for one base topic. The model is chosen by
    `expand.backend` (openai | ollama | local). `system`/`user` are the expansion prompt for
    this --expand run; `{topic}` and `{count}` are substituted (topic appended if `{topic}` is
    absent, mirroring the generation topic loop). Results are de-duplicated and capped to count.
    """
    ecfg = config.get("expand", {})
    count = int(ecfg.get("count", 10))
    backend = ecfg.get("backend", "openai")

    # System prompt: only substitute placeholders, never append the topic (it's framing, not
    # the per-topic input). User prompt: substitute, or append the topic if `{topic}` is absent,
    # mirroring the generation topic loop.
    sys_prompt = system.replace("{count}", str(count)).replace("{topic}", topic) if system else ""
    user_prompt = user.replace("{count}", str(count))
    user_prompt = (
        user_prompt.replace("{topic}", topic)
        if "{topic}" in user_prompt
        else f"{user_prompt}\n\nTopic: {topic}"
    )

    if backend == "local":
        variants = _expand_local(config, sys_prompt, user_prompt)
    elif backend == "openai":
        variants = _parse_variants(call_openai(config, sys_prompt, user_prompt, schema=None))
    elif backend == "ollama":
        variants = _parse_variants(call_ollama(config, sys_prompt, user_prompt))
    else:
        raise ValueError(
            f"Unknown expand.backend '{backend}'. Choose from: openai, ollama, local"
        )

    # de-duplicate (preserving order) and cap to the requested count.
    seen, unique = set(), []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique[:count]


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
    parser.add_argument(
        "--validate",
        metavar="FILE",
        help="Validate a JSONL file as a Dataset (one Conversation per line) and exit; "
        "no LLM call is made",
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
        "--expand",
        action="store_true",
        help="Topic-expansion mode: read --topics, generate expand.count wordings per topic "
        "(model chosen by expand.backend in config), and print one '<topic><sep><wording>' "
        "line per variant to stdout. No Dataset/JSONL is produced; --system/--user are the "
        "expansion prompt for this run. Feed the output back via --topics for generation.",
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

    if args.validate:
        sys.exit(validate_dataset_file(args.validate))

    config = load_config(args.config)

    system = read_file(args.system).strip() if args.system else ""
    user = read_file(args.user).strip() if args.user else sys.stdin.read().strip()

    if not user:
        parser.error("User prompt is empty — provide --user FILE or pipe text to stdin")

    if args.expand:
        if not args.topics:
            parser.error("--expand requires --topics FILE")
        topics = [line.strip() for line in read_file(args.topics).splitlines() if line.strip()]
        if not topics:
            parser.error(f"No topics found in {args.topics}")
        separator = config.get("expand", {}).get("separator", " — ")
        for topic in tqdm(topics, "expand"):
            for variant in expand_topic(config, system, user, topic):
                print(f"{topic}{separator}{variant}")
        return

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
    elif "{template}" in user:
        # --template omitted: drop the placeholder so it isn't sent literally
        user = user.replace("{template}", "")

    if "{number}" in user:
        user = user.replace("{number}", str(args.count))

    # print('user prompt:', user, file=sys.stderr)
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
            # print('topic user prompt:', topic_user, file=sys.stderr)
            if i > 0:
                print()
            # print(f"===== {topic} =====")
            print(process_result(backend_fn(config, system, topic_user))) # NOTE: for openai backed the Dataset schema is passed by default
    else:
        print(process_result(backend_fn(config, system, user)))


if __name__ == "__main__":
    main()
