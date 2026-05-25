# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then set OPENAI_API_KEY
```

For the `pipeline` backend also install:
```bash
pip install transformers torch
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
| `pipeline` | none | HuggingFace `text-generation` pipeline; downloads model on first run; returns `generated_text[-1]["content"]` for chat-template models or a raw string for plain models |
