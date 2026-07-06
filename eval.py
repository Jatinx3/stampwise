"""Golden-set eval: hit@4, MRR@10, trap abstention, false-abstain rate.

Retrieval + abstention only — no LLM calls. Usage:
  python eval.py            # run metrics at the current threshold
  python eval.py --sweep    # sweep thresholds to tune abstention
"""

import json
import sys
from pathlib import Path

import answer as answer_mod
import search

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.jsonl"


def load_golden() -> list[dict]:
    items = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]
    unreviewed = sum(1 for i in items if not i.get("reviewed"))
    if unreviewed:
        print("=" * 70)
        print(f"!! WARNING: {unreviewed}/{len(items)} golden items are UNREVIEWED !!")
        print("!! Metrics below are against machine-drafted ground truth.       !!")
        print("!! A human must verify each item and set reviewed: true.         !!")
        print("=" * 70)
    return items


def run_retrieval(items: list[dict]) -> list[dict]:
    """Attach top-10 results + top score to every golden item."""
    for item in items:
        results = search.search(item["question"], top_n=10)
        item["_results"] = results
        item["_top_score"] = results[0]["score"] if results else None
    return items


def retrieval_metrics(answerable: list[dict]) -> tuple[float, float]:
    hits, rr_sum = 0, 0.0
    for item in answerable:
        urls = [c["source_url"] for c in item["_results"]]
        expected = set(item["expected_urls"])
        if any(u in expected for u in urls[:4]):
            hits += 1
        rank = next((i for i, u in enumerate(urls, start=1) if u in expected), None)
        rr_sum += 1.0 / rank if rank else 0.0
    n = len(answerable)
    return hits / n, rr_sum / n


def abstention_metrics(items: list[dict], threshold: float) -> tuple[int, int, int, int]:
    traps = [i for i in items if i["type"] == "trap"]
    answerable = [i for i in items if i["type"] == "answerable"]
    trap_abstained = sum(1 for i in traps
                         if answer_mod.should_abstain(i["_top_score"], threshold))
    false_abstain = sum(1 for i in answerable
                        if answer_mod.should_abstain(i["_top_score"], threshold))
    return trap_abstained, len(traps), false_abstain, len(answerable)


def sweep(items: list[dict]):
    print("\n| Threshold | Traps abstained | False abstains |")
    print("|---|---|---|")
    for t in [0.010, 0.014, 0.016, 0.018, 0.020, 0.024, 0.028, 0.032, 0.036]:
        ta, nt, fa, na = abstention_metrics(items, t)
        print(f"| {t:.3f} | {ta}/{nt} | {fa}/{na} |")
    scores = sorted((i["_top_score"], i["type"], i["question"][:60]) for i in items
                    if i["_top_score"] is not None)
    print("\nTop RRF score per question (sorted):")
    for s, typ, q in scores:
        print(f"  {s:.4f}  {typ:10s}  {q}")


def main():
    items = run_retrieval(load_golden())
    answerable = [i for i in items if i["type"] == "answerable"]
    if "--sweep" in sys.argv:
        sweep(items)
        return
    hit4, mrr = retrieval_metrics(answerable)
    t = answer_mod.ABSTAIN_THRESHOLD
    ta, nt, fa, na = abstention_metrics(items, t)
    print(f"\n## Eval results ({len(items)} questions, threshold={t})\n")
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| hit@4 (answerable) | {hit4:.2f} |")
    print(f"| MRR@10 (answerable) | {mrr:.2f} |")
    print(f"| Trap abstention | {ta}/{nt} |")
    print(f"| False-abstain rate | {fa}/{na} ({fa / na:.2f}) |")
    misses = [i for i in answerable
              if not any(c["source_url"] in set(i["expected_urls"]) for c in i["_results"][:4])]
    if misses:
        print("\nhit@4 misses:")
        for i in misses:
            print(f"  - {i['question']}")


if __name__ == "__main__":
    main()
