# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY
```

For the `pipeline` backend also install:
```bash
pip install outlines transformers torch
```

## Running

```bash
echo "What is 2+2?" | python main.py
python main.py --user prompt.txt
python main.py --system system.txt --user prompt.txt
python main.py --config my_config.yaml --user prompt.txt
```

## Architecture

Everything lives in `main.py`. The flow is:

1. CLI args parsed by `argparse` — `--config`, `--system`, `--user`
2. `load_config()` reads `config.yaml` (YAML)
3. System prompt comes from `--system FILE`; user prompt from `--user FILE` or stdin
4. `config.backend` selects one of three backend functions from the `BACKENDS` dict
5. The backend function is called with `(config, system, user)` and its return value is printed to stdout

**Adding a new backend:** write a `call_<name>(config, system, user) -> str` function, add it to the `BACKENDS` dict, and document its config block in `config.yaml`.

## Config file

`config.yaml` has a top-level `backend` key and one section per backend. Each backend function reads only its own section via `config.get("<backend>", {})`. Defaults are hardcoded in the backend functions, not in the YAML.

## Backends

| backend | key | notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` env var (loaded via python-dotenv) | uses `openai` SDK |
| `ollama` | none | calls `/api/chat` on `ollama.base_url` with `stream: false` |
| `pipeline` | none | Local HuggingFace model loaded via `outlines.from_transformers`; downloads model on first run; **constrained decoding** (`outlines`) guarantees output matches the `Dataset` Pydantic schema — returns a JSON array of `{"messages":[...]}` conversations, which `process_result` then emits as JSONL. Each conversation is **positionally typed** (a `Tuple`), so the schema itself pins the layout: `[user question (no token), assistant "<\|humorous\|>…", assistant "<\|normal\|>…"]`. The preference token is forced as a prefix on the two assistant turns via `pattern` (full-match at generation time; use unbounded `+`, not `{1,N}`, or llguidance's lexer overflows with "too many expressions"), the user turn forbids `\|` so the token can't leak into it, and the conversation always ends on the `<\|normal\|>` assistant turn. Generation goes through the **high-level `generator()` call** (one constrained pass per attempt); each call resets and correctly drives the llguidance matcher, so do NOT bolt the logits processor onto a raw `hf_model.generate()` loop — that skips the per-token matcher commits and the output degenerates into repetition. Since the grammar is compiled from the schema, a complete pass is never malformed JSON — validation only fails on **truncation** (ran out of tokens before closing the array). So a fixed-budget retry is pointless; instead each attempt **grows the budget** (attempt _k_ gets `k * max_new_tokens`) and `max_attempts` caps the escalation. Raise `max_new_tokens` (base budget) and/or `max_attempts`, or lower `--count`, if it stays incomplete |
