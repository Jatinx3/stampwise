---
title: StampWise Mini
emoji: 🇮🇪
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---

# StampWise Mini

Small, free, citation-grounded RAG assistant for Irish immigration rules
(registration stamps, student pathways, work permits). Answers come **only**
from ~30 scraped official pages, every claim carries a numbered citation, and
questions the sources don't cover get an explicit refusal.

> ⚠️ Informational only, **not legal advice**. Sources may be out of date —
> always verify against the linked official pages.

**Live demo:** _not yet deployed — see [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces)._

## How it works

```
ingest.py   sources.yaml allowlist -> scrape (robots-aware, 1 req/s, raw HTML
            cached in data/raw) -> markdown -> heading chunks (<= ~500 tokens)
            -> SQLite + FTS5 -> bge-small-en-v1.5 embeddings (embeddings.npy)
search.py   FTS5 BM25 top-10 + brute-force cosine top-10
            -> Reciprocal Rank Fusion (k=60) -> top-4 chunks with scores
answer.py   abstention gate -> prompt with numbered excerpts -> [n] citation
            mapping -> grounding guard -> disclaimer
app.py      Gradio chat + sources/freshness tab
eval.py     golden-set metrics (no LLM needed)
```

Corpus: 31 pages / 483 chunks from irishimmigration.ie,
citizensinformation.ie and enterprise.gov.ie (explicit allowlist in
[data/sources.yaml](data/sources.yaml); the scraper never follows links).

### Grounding, in three layers

1. **Abstention gate** — if the top RRF score is below `ABSTAIN_THRESHOLD`
   (0.02), the LLM is never called and the app answers
   *"That isn't covered in my indexed sources."* Clearing 0.02 requires the
   keyword and vector rankings to agree on a chunk; vector candidates below
   `COSINE_FLOOR` (0.65) are dropped first. Queries are normalized before
   search ("stamp2" -> "stamp 2").
2. **Prompt** — the LLM may only use the four provided excerpts and must cite
   each claim as `[n]`.
3. **Grounding guard** — a returned answer containing no `[n]` citations at
   all is rejected and replaced with the refusal message.

## Eval

`python eval.py` — retrieval and abstention metrics, no LLM call.
25 golden questions: 20 answerable + 5 traps (off-corpus / non-existent).

| Metric | Value |
|---|---|
| hit@4 (answerable) | 0.95 |
| MRR@10 (answerable) | 0.96 |
| Trap abstention (score gate) | 1/5 |
| Trap refusal end-to-end (gate + LLM layers) | 5/5 |
| False-abstain rate | 0/20 (0.00) |

Threshold tuning (`python eval.py --sweep`): the gate is stable across
0.018–0.028; 0.02 sits mid-plateau.

**Documented near-miss / design tradeoff.** On a 483-chunk corpus,
bge-small's cosine range is too compressed to separate on-topic-adjacent
traps ("minimum wage", "Stamp 9") from tersely phrased real questions:
"what is stamp 2" peaks at cosine 0.695 while four of the five traps peak
higher (up to 0.751). Any `COSINE_FLOOR` strict enough to score-gate all
five traps also false-abstains on casual definitional queries — the floor
therefore stays loose (0.65) and only 1/5 traps is caught by the score gate
alone. The other four are refused at the next layers (prompt rules +
no-citation guard): all 5/5 traps produce the refusal message end-to-end,
verified live against `llama-3.3-70b` / `nemotron-3-super` via OpenRouter.
The tradeoff buys a 0/20 false-abstain rate with natural phrasings.

**Golden-set status:** machine-drafted, `reviewed: false` on every item.
`eval.py` prints a loud warning until a human verifies each question and
flips the flag.

## Run locally

```bash
python3 -m venv ~/.venvs/stampwise-mini      # outside iCloud-synced folders!
~/.venvs/stampwise-mini/bin/pip install -r requirements.txt
export GROQ_API_KEY=...                      # free key: console.groq.com

python ingest.py     # scrape + index + embed (~2 min first run)
python app.py        # Gradio UI at http://127.0.0.1:7860
python eval.py       # metrics table (no LLM needed)
python -m pytest     # 12 offline unit tests
python answer.py "What is Stamp 1G?"   # CLI one-shot
```

### Using Ollama instead of Groq

Same code path — just point the OpenAI-compatible client elsewhere:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1        # any local model
python app.py
```

### Using OpenRouter free models

Also the same code path (free key: openrouter.ai/keys):

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-v1-...
export LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

Caveats: free variants are rate-limited (~50 req/day without credits) and
periodically 429 when the shared upstream pool is congested — retry later or
switch models. Reasoning models (e.g.
`nvidia/nemotron-3-super-120b-a12b:free`) can leak chain-of-thought into the
reply; set `LLM_REASONING_EXCLUDE=1` to ask OpenRouter to strip it (honored
by most, not all, upstream providers).

Note: small local models may answer without `[n]` citations; the grounding
guard then replaces the reply with the refusal message. The default
(`llama-3.3-70b-versatile` on Groq) follows the citation format reliably.

## Deploying to Hugging Face Spaces

1. Create a free Gradio Space, add `GROQ_API_KEY` under
   *Settings -> Variables and secrets*.
2. Push this repo to the Space. The index artifacts are gitignored locally,
   so force-add them for the Space (they're ~1.5 MB total):

```bash
git remote add space https://huggingface.co/spaces/<user>/stampwise-mini
git add -f data/index.db data/embeddings.npy
git commit -m "add prebuilt index for Space"
git push space main
```

3. Link the running Space here.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` / `OPENAI_API_KEY` | — | LLM auth (Ollama needs none) |
| `OPENAI_BASE_URL` | Groq endpoint | any OpenAI-compatible server |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | chat model |
| `ABSTAIN_THRESHOLD` | `0.02` | min top RRF score to call the LLM |
| `COSINE_FLOOR` | `0.65` | min cosine for vector candidates |
| `LLM_REASONING_EXCLUDE` | unset | `1` strips reasoning traces (OpenRouter) |
