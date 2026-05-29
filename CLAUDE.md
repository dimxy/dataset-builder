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
python main.py --validate result.jsonl   # check a JSONL file is a valid Dataset, then exit
```

`--validate FILE` makes no LLM call: it validates each JSONL line against the `Conversation` schema and the whole file against `DatasetFile`, prints per-line errors to stderr, and exits non-zero if any line is invalid. `DatasetFile` is `Dataset` without the 20-conversation `max_length` cap — that cap only bounds a single constrained-decoding pass, so a stored file may legitimately hold more.

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
| `pipeline` | none | Local HuggingFace model loaded via `outlines.from_transformers`; downloads model on first run; **constrained decoding** (`outlines`) guarantees output matches the `Dataset` Pydantic schema — returns a JSON array of `{"messages":[...]}` conversations, which `process_result` then emits as JSONL. Each conversation is **positionally typed** (a `Tuple`), so the schema itself pins the layout: `[user "<\|humorous\|>…", assistant …, user "<\|normal\|>…", assistant …]`. The preference token lives on the **user** turns (that is the input the fine-tuned model is trained to read), forced as a prefix via `pattern` (full-match at generation time; use unbounded `+`, not `{1,N}`, or llguidance's lexer overflows with "too many expressions"); the **assistant** turns carry the humorous / normal *response* and forbid `\|` so the token can't leak into them. Order is fixed by position: humorous user turn + its reply, then normal user turn + its reply. Generation goes through the **high-level `generator()` call** (one constrained pass per attempt); each call resets and correctly drives the llguidance matcher, so do NOT bolt the logits processor onto a raw `hf_model.generate()` loop — that skips the per-token matcher commits and the output degenerates into repetition. Since the grammar is compiled from the schema, a complete pass is never malformed JSON — validation only fails on **truncation** (ran out of tokens before closing the array). So a fixed-budget retry is pointless; instead each attempt **grows the budget** (attempt _k_ gets `k * max_new_tokens`) and `max_attempts` caps the escalation. Raise `max_new_tokens` (base budget) and/or `max_attempts`, or lower `--count`, if it stays incomplete |
