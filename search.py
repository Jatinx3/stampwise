"""Hybrid retrieval: FTS5 BM25 top-10 + cosine top-10 -> RRF -> top-4."""

import os
import re
import sqlite3
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "index.db"
EMB_PATH = DATA_DIR / "embeddings.npy"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60
CANDIDATES = 10
TOP_N = 4
# Vector candidates below this cosine are dropped. Keyword-only matches then
# top out at 1/(RRF_K+1) ≈ 0.016, under the abstention threshold — this is
# what lets off-corpus questions abstain despite sharing keywords ("Stamp",
# "Ireland") with the corpus. Tuned on data/golden.jsonl (see README).
# Loose on purpose: casual phrasings like "what is stamp 2" embed as low as
# ~0.67 on bge-small, so tighter floors abstain on real questions. Traps that
# clear this floor are caught by the RRF agreement threshold or, past that,
# the prompt rules + no-citation grounding guard.
COSINE_FLOOR = float(os.environ.get("COSINE_FLOOR", "0.65"))


def get_db() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _embeddings():
    import numpy as np
    return np.load(EMB_PATH)


def normalize_query(query: str) -> str:
    """Split glued letter->digit tokens: "stamp2" -> "stamp 2",
    "stamp1g" -> "stamp 1g". Digit->letter stays joined ("1g", "2a")
    since stamp names are written that way in the corpus."""
    return re.sub(r"(?<=[A-Za-z])(?=\d)", " ", query)


# Question-phrasing words that dilute BM25 on short chunks.
STOPWORDS = frozenset("""
a about after am an and any are as at be before by can could do does for from
get has have how i if in into is it its me my need of on or should so than
that the their there they this to under until wants was we what when where
which while who whose why will with would you your
""".split())


def fts_search(con: sqlite3.Connection, query: str, k: int = CANDIDATES) -> list[int]:
    """BM25 keyword search. Returns chunk ids, best first."""
    terms = [t for t in re.findall(r"[A-Za-z0-9]+", query)
             if t.lower() not in STOPWORDS]
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    rows = con.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?"
        " ORDER BY bm25(chunks_fts) LIMIT ?", (match, k)).fetchall()
    return [r["rowid"] for r in rows]


def vector_search(query: str, k: int = CANDIDATES,
                  floor: float = COSINE_FLOOR) -> list[int]:
    """Brute-force cosine over normalized embeddings. Returns chunk ids."""
    q = _embedder().encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
    sims = _embeddings() @ q
    order = sims.argsort()[::-1][:k]
    return [int(i) + 1 for i in order if sims[i] >= floor]  # row i <-> chunk id i+1


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1/(k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def search(query: str, top_n: int = TOP_N) -> list[dict]:
    """Return top_n chunks with RRF scores, best first."""
    query = normalize_query(query)
    con = get_db()
    try:
        fused = rrf_fuse([fts_search(con, query), vector_search(query)])
        best = sorted(fused, key=fused.get, reverse=True)[:top_n]
        if not best:
            return []
        placeholders = ",".join("?" * len(best))
        rows = {r["id"]: dict(r) for r in con.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", best)}
        return [{**rows[i], "score": fused[i]} for i in best]
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    for c in search(" ".join(sys.argv[1:]) or "What is Stamp 1G?"):
        print(f"{c['score']:.4f}  [{c['page_title']} > {c['heading_path']}]"
              f"  {c['content'][:80]!r}")
