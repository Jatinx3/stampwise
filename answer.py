"""Prompt assembly, [n] citation mapping, abstention threshold."""

import os
import re

import search

# Tuned on data/golden.jsonl via `python eval.py --sweep` (see README).
ABSTAIN_THRESHOLD = float(os.environ.get("ABSTAIN_THRESHOLD", "0.02"))

NOT_FOUND = "That isn't covered in my indexed sources."
DISCLAIMER = "This is general information, not legal advice."

SYSTEM_PROMPT = """\
You are StampWise Mini, an assistant for Irish immigration questions.
Answer using ONLY the numbered source excerpts provided. Rules:
- Cite every claim with the excerpt number in square brackets, e.g. [1] or [2][3].
- Never use knowledge that is not in the excerpts, even for topics you know well.
- If the excerpts do not directly address the question's topic, do not answer
  from adjacent material. Reply exactly:
  "That isn't covered in my indexed sources."
- Be concise and factual. Do not add a disclaimer; one is appended automatically.
Answers without [n] citations are rejected automatically."""


def should_abstain(top_score: float | None, threshold: float = ABSTAIN_THRESHOLD) -> bool:
    """Abstain (skip the LLM entirely) when retrieval confidence is too low."""
    return top_score is None or top_score < threshold


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    excerpts = "\n\n".join(
        f"[{i}] {c['page_title']} — {c['heading_path']}\n({c['source_url']})\n{c['content']}"
        for i, c in enumerate(chunks, start=1))
    user = f"Source excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def extract_citations(text: str, n_chunks: int) -> list[int]:
    """Ordered unique [n] markers that point at a real chunk (1..n_chunks)."""
    seen = []
    for m in re.finditer(r"\[(\d+)\]", text):
        n = int(m.group(1))
        if 1 <= n <= n_chunks and n not in seen:
            seen.append(n)
    return seen


def map_citations(text: str, chunks: list[dict]) -> list[tuple[int, dict]]:
    return [(n, chunks[n - 1]) for n in extract_citations(text, len(chunks))]


def answer(question: str) -> dict:
    """Full pipeline. Returns {answer, citations, chunks, abstained}."""
    chunks = search.search(question)
    top_score = chunks[0]["score"] if chunks else None
    if should_abstain(top_score):
        return {"answer": f"{NOT_FOUND}\n\n{DISCLAIMER}", "citations": [],
                "chunks": chunks, "abstained": True}
    import llm
    text = llm.chat(build_prompt(question, chunks)).strip()
    citations = map_citations(text, chunks)
    abstained = NOT_FOUND.rstrip(".") in text
    if not abstained and not citations:
        # Grounding guard: every claim must cite an excerpt. No [n] anywhere
        # means the model answered from its own knowledge — reject it.
        text, citations, abstained = NOT_FOUND, [], True
    return {"answer": f"{text}\n\n{DISCLAIMER}", "citations": citations,
            "chunks": chunks, "abstained": abstained}


if __name__ == "__main__":
    import sys
    result = answer(" ".join(sys.argv[1:]) or "What is Stamp 1G?")
    print(result["answer"])
    if result["citations"]:
        print("\nSources:")
        for n, c in result["citations"]:
            print(f"  [{n}] {c['page_title']} — {c['source_url']}")
