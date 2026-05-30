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
python main.py --expand --topics prompts/topics.txt --user prompts/expand-user.txt > prompts/topics-expanded.txt
```

### Topic expansion (two-pass, to beat small-context duplication)

The generator makes a batch of conversations per topic. With a small-context generation model,
asking for many distinct samples per topic overflows the window and forces a mid-topic model
reset, which causes **duplicate** responses. To work around it, expand each base topic into
several distinct *wordings* first, then generate a *small* batch per extended topic:

```bash
# Pass 1: multiply topics. Each base topic -> expand.count "<topic><sep><wording>" lines.
python main.py --expand --topics prompts/topics.txt --user prompts/expand-user.txt > prompts/topics-expanded.txt
# Pass 2: the normal flow, now over the (larger, more varied) expanded topic list.
python main.py --topics prompts/topics-expanded.txt --system prompts/system.txt --user prompts/user.txt > result.jsonl
```

`--expand` makes no `Dataset` output: for each line of `--topics` it generates `expand.count`
wordings and prints `"<topic><separator><wording>"` lines to stdout. The wording model is chosen
by `expand.backend` (`openai` | `ollama` | `local`) — the `pipeline` backend can't expand because
it only emits `Dataset` JSON. `--system`/`--user` on the `--expand` run are the *expansion* prompt
(`{topic}` and `{count}` are substituted). `openai`/`ollama` return free text parsed into a list
(JSON array preferred, newline-list fallback); `local` reuses the `pipeline` model constrained to
the `TopicVariants` (`List[str]`) schema via the shared `_build_outlines_generator(..., schema)`.
Count and separator live in the `expand:` config block.

`--validate FILE` makes no LLM call: it validates each JSONL line against the `StoredConversation` schema (a flattened `{"messages":[...]}` record) and the whole file against `DatasetFile`, prints per-line errors to stderr, and exits non-zero if any line is invalid. `DatasetFile` drops the 20-conversation `max_length` cap that `Dataset` carries — that cap only bounds a single constrained-decoding pass, so a stored file may legitimately hold more.

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
| `pipeline` | none | Local HuggingFace model loaded via `outlines.from_transformers`; downloads model on first run; **constrained decoding** (`outlines`) guarantees output matches the `Dataset` Pydantic schema — returns a JSON array of generation-shape conversations, which `process_result` flattens and emits as JSONL. Each generated conversation is an **object with two named pairs**, `{"humorous": [user "<\|humorous\|>…", assistant …], "normal": [user "<\|normal\|>…", assistant …]}`, each pair a positionally-typed 2-tuple. Named keys were chosen deliberately over a `Union` of two 2-tuples: OpenAI's `json_schema` turns a Union into an `anyOf` and then packs *both* pairs into a single object anyway, producing output that its own schema can't validate. The named-object shape sidesteps that and reads unambiguously for every backend. `process_result` then **flattens** each conversation into the stored `StoredConversation` form `{"messages": [hum_user, hum_assistant, norm_user, norm_assistant]}` — the layout the fine-tuning app reads and `--validate` checks — so `openai` and `pipeline` emit identical JSONL. The preference token lives on each **user** turn (that is the input the fine-tuned model is trained to read), forced as a prefix via `pattern` (full-match at generation time; use unbounded `+`, not `{1,N}`, or llguidance's lexer overflows with "too many expressions"); the **assistant** turns carry the humorous / normal *responses* and forbid `\|` so the token can't leak into them. Generation goes through the **high-level `generator()` call** (one constrained pass per attempt); each call resets and correctly drives the llguidance matcher, so do NOT bolt the logits processor onto a raw `hf_model.generate()` loop — that skips the per-token matcher commits and the output degenerates into repetition. Since the grammar is compiled from the schema, a complete pass is never malformed JSON — validation only fails on **truncation** (ran out of tokens before closing the array). So a fixed-budget retry is pointless; instead each attempt **grows the budget** (attempt _k_ gets `k * max_new_tokens`) and `max_attempts` caps the escalation. Raise `max_new_tokens` (base budget) and/or `max_attempts`, or lower `--count`, if it stays incomplete |
