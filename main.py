import argparse
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def call_openai(config: dict, system: str, user: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    cfg = config.get("openai", {})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=cfg.get("model", "gpt-4o-mini"),
        messages=messages,
        temperature=cfg.get("temperature", 0.7),
        max_tokens=cfg.get("max_tokens", 1024),
    )
    return response.choices[0].message.content


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


def call_pipeline(config: dict, system: str, user: str) -> str:
    from transformers import pipeline as hf_pipeline

    cfg = config.get("pipeline", {})
    model = cfg.get("model")
    if not model:
        raise ValueError("pipeline.model must be set in config")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        device=cfg.get("device", "cpu"),
    )
    result = pipe(
        messages,
        max_new_tokens=cfg.get("max_new_tokens", 1024),
        temperature=cfg.get("temperature", 0.7),
        do_sample=cfg.get("temperature", 0.7) > 0,
    )
    generated = result[0]["generated_text"]
    # chat-template models return a list of message dicts; plain models return a string
    if isinstance(generated, list):
        return generated[-1]["content"]
    return generated


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
    args = parser.parse_args()

    config = load_config(args.config)

    system = read_file(args.system).strip() if args.system else ""
    user = read_file(args.user).strip() if args.user else sys.stdin.read().strip()

    if not user:
        parser.error("User prompt is empty — provide --user FILE or pipe text to stdin")

    backend_name = config.get("backend", "openai")
    backend_fn = BACKENDS.get(backend_name)
    if backend_fn is None:
        raise ValueError(f"Unknown backend '{backend_name}'. Choose from: {', '.join(BACKENDS)}")

    print(backend_fn(config, system, user))


if __name__ == "__main__":
    main()
