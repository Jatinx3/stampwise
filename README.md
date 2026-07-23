# 🇮🇪 StampWise Mini

A small, free, **citation-grounded** RAG assistant for Irish immigration rules —
registration stamps, student pathways, and work permits.

Every answer is built **only** from ~30 official government pages, each claim
carries a numbered `[n]` citation back to its source, and any question the
sources don't cover gets an honest *"That isn't covered in my indexed sources"*
instead of a hallucination.

> ⚠️ **Informational only — not legal advice.** Sources may be out of date;
> always verify against the linked official pages.

---

## Highlights

- **Grounded or silent.** Three independent layers keep answers tied to the
  sources; when confidence is low, the model is never even called.
- **Hybrid retrieval.** Keyword search (SQLite FTS5 / BM25) fused with semantic
  search (embeddings) via Reciprocal Rank Fusion — no vector database needed.
- **Runs entirely free.** No paid APIs, no servers, no Docker. Works with Groq's
  free tier, OpenRouter's free models, or a fully local Ollama model.
- **Honest evaluation.** A 25-question golden set measures retrieval quality and
  refusal accuracy — with no LLM calls required.

## Quick start

```bash
# 1. Set up an isolated environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add a free LLM key (pick one)
export GROQ_API_KEY=...          # free key from console.groq.com

# 3. Build the index, then launch
python ingest.py                 # scrape + chunk + embed (~2 min first run)
python app.py                    # open http://127.0.0.1:7860
```

Prefer the command line?

```bash
python answer.py "What is Stamp 1G?"
```

> **Tip:** keep the virtualenv out of iCloud-synced folders (Desktop, Documents).
> PyTorch can hang for minutes reading its libraries through iCloud.

## Example

**Q — "What is Stamp 2?"**

> Stamp 2 is the immigration permission to study a full-time course on the
> official Interim List of Eligible Programmes (ILEP). It allows casual work of
> up to 20 hours per week during term and 40 hours per week during holidays, and
> it is not reckonable toward citizenship by naturalisation. [2]
>
> *This is general information, not legal advice.*
>
> **Sources** — [2] Immigration permission / stamps — Immigration Service Delivery

**Q — "How do I get a fishing licence in Spain?"**

> That isn't covered in my indexed sources.

## How it works

```
ingest.py   sources.yaml allowlist → scrape (robots-aware, 1 req/s, HTML cached)
            → markdown → heading chunks (≤ ~500 tokens)
            → SQLite + FTS5  and  bge-small-en-v1.5 embeddings

search.py   FTS5 keyword top-10  +  cosine top-10
            → Reciprocal Rank Fusion (k=60)  → top-4 chunks with scores

answer.py   abstention gate → prompt with numbered excerpts
            → [n] citation mapping → grounding guard → disclaimer

app.py      Gradio chat UI + a "Sources & freshness" tab
eval.py     golden-set metrics (retrieval + refusal, no LLM)
```

**Corpus:** 31 pages / 483 chunks from `irishimmigration.ie`,
`citizensinformation.ie`, and `enterprise.gov.ie`. The scraper only visits the
explicit allowlist in [`data/sources.yaml`](data/sources.yaml) — it never
follows links off it.

### Three layers of grounding

1. **Abstention gate** — if the top fused score is below `ABSTAIN_THRESHOLD`
   (0.02), the LLM is never called and the app returns the refusal message.
   Clearing the gate requires the keyword and semantic rankings to *agree* on a
   chunk; weak semantic matches (cosine below `COSINE_FLOOR`, 0.65) are dropped
   first. Queries are normalized beforehand, so `"stamp2"` becomes `"stamp 2"`.
2. **Prompt** — the model may use *only* the four supplied excerpts and must
   cite every claim as `[n]`.
3. **Grounding guard** — any answer that comes back with no `[n]` citation at
   all is discarded and replaced with the refusal message.

## Project structure

| File | Responsibility |
|---|---|
| `ingest.py` | Scrape the allowlist → chunk → SQLite (FTS5) + `embeddings.npy` |
| `search.py` | Hybrid FTS5 + cosine retrieval fused with RRF |
| `answer.py` | Prompt assembly, citation mapping, abstention logic |
| `llm.py` | OpenAI-compatible client (Groq / OpenRouter / Ollama) |
| `app.py` | Gradio chat interface |
| `eval.py` | Golden-set metrics |
| `test_core.py` | Offline unit tests (chunking, RRF, citations, threshold) |
| `data/sources.yaml` | The URL allowlist |
| `data/golden.jsonl` | 25 golden questions (20 answerable + 5 traps) |

## Evaluation

```bash
python eval.py          # metrics table — no LLM needed
python eval.py --sweep  # threshold-tuning report
python -m pytest        # 13 offline unit tests
```

`eval.py` runs 25 golden questions (20 answerable, 5 traps about things not in
the corpus):

| Metric | Value |
|---|---|
| hit@4 (answerable) | 0.95 |
| MRR@10 (answerable) | 0.96 |
| Trap refusal, end-to-end | 5 / 5 |
| False-abstain rate | 0 / 20 |

<details>
<summary><b>Design note — why the score gate only catches 1 of 5 traps</b></summary>

On a corpus this small, `bge-small`'s cosine similarities are too tightly
bunched to cleanly separate topic-adjacent traps ("minimum wage", "Stamp 9")
from tersely phrased real questions. For example, *"what is stamp 2"* peaks at
cosine 0.695, while four of the five traps peak *higher* (up to 0.751).

Any `COSINE_FLOOR` strict enough to score-gate all five traps would also start
refusing legitimate casual questions. So the floor stays loose (0.65): the score
gate alone catches only 1 of 5 traps, and the remaining four are refused at the
next layers (prompt rules + the no-citation guard). The net result — verified
live against `llama-3.3-70b` and `nemotron-3-super` on OpenRouter — is **5/5
traps refused end-to-end** with **zero false refusals** on natural phrasings.

</details>

> **Golden-set status:** the questions are machine-drafted with
> `reviewed: false` on every item. `eval.py` prints a warning until a human
> verifies each question and flips the flag.

## Configuration

All configuration is via environment variables — nothing is hard-coded.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` / `OPENAI_API_KEY` | — | LLM auth (Ollama needs none) |
| `OPENAI_BASE_URL` | Groq endpoint | Any OpenAI-compatible server |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Chat model |
| `ABSTAIN_THRESHOLD` | `0.02` | Min fused score before calling the LLM |
| `COSINE_FLOOR` | `0.65` | Min cosine for a semantic candidate |
| `LLM_REASONING_EXCLUDE` | unset | `1` strips reasoning traces (OpenRouter) |

Keys can also live in a local `.env` file beside the code (git-ignored); real
environment variables always take precedence.

### Choosing an LLM provider

The client is OpenAI-compatible, so switching providers is just environment
variables — **one code path, no branching.**

**Groq free tier** (default, most generous limits):

```bash
export GROQ_API_KEY=...           # console.groq.com
```

**OpenRouter free models:**

```bash
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-v1-...            # openrouter.ai/keys
export LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

> Free variants are rate-limited (~50 requests/day without credits) and
> periodically return HTTP 429 when the shared pool is busy — retry or switch
> models. Reasoning models (e.g. `nvidia/nemotron-3-super-120b-a12b:free`) may
> leak their chain-of-thought; set `LLM_REASONING_EXCLUDE=1` to strip it.

**Local Ollama** (fully offline, no key):

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3.1
```

> Small local models sometimes answer without `[n]` citations; the grounding
> guard then replaces the reply with the refusal message. Larger models
> (the Groq default) follow the citation format reliably.

## Deploying to Hugging Face Spaces

Hugging Face Spaces configures itself from a YAML block at the very top of
`README.md`. Prepend this block (it renders as a table on GitHub, so it's kept
here rather than at the top of this file):

```yaml
---
title: StampWise Mini
emoji: 🇮🇪
colorFrom: green
colorTo: yellow
sdk: gradio
app_file: app.py
pinned: false
---
```

Then:

1. Create a free **Gradio** Space and add your `GROQ_API_KEY` under
   *Settings → Variables and secrets*.
2. The index files are git-ignored locally, so force-add them for the Space
   (~1.5 MB total) and push:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/stampwise-mini
   git add -f data/index.db data/embeddings.npy
   git commit -m "Add prebuilt index for Space"
   git push space main
   ```

## License & scope

Built as a focused demo: no Postgres, no Docker, no reranker, no streaming, no
multi-turn memory, no accounts, and no other countries. The indexed content
belongs to its respective Irish government sources and is used here for
informational retrieval only.
