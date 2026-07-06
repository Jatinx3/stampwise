"""Ingest the URL allowlist into SQLite (FTS5) + embeddings.npy.

Pipeline: sources.yaml -> fetch (robots-aware, 1 req/s, cached in data/raw)
-> main-content HTML -> markdown -> heading chunks (<= ~500 tokens)
-> SQLite chunks table + FTS5 index -> bge-small-en-v1.5 embeddings.
"""

import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "index.db"
EMB_PATH = DATA_DIR / "embeddings.npy"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 500


def est_tokens(text: str) -> float:
    """Rough token estimate: ~1.3 tokens per whitespace word."""
    return len(text.split()) * 1.3


def load_sources():
    with open(DATA_DIR / "sources.yaml") as f:
        return yaml.safe_load(f)


class Fetcher:
    """Allowlist-only fetcher: robots-aware, rate-limited, raw-HTML cache."""

    def __init__(self, user_agent: str, delay: float):
        self.ua = user_agent
        self.delay = delay
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.last_request = 0.0
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _throttle(self):
        wait = self.delay - (time.time() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        self.last_request = time.time()

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._throttle()
                r = requests.get(f"https://{host}/robots.txt",
                                 headers={"User-Agent": self.ua}, timeout=15)
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except requests.RequestException:
                rp.parse([])  # unreachable robots.txt -> allow
            self.robots[host] = rp
        return self.robots[host].can_fetch(self.ua, url)

    def fetch(self, url: str) -> tuple[str, str] | None:
        """Return (html, fetched_at_iso) or None. Uses data/raw cache."""
        key = hashlib.sha1(url.encode()).hexdigest()
        html_file, meta_file = RAW_DIR / f"{key}.html", RAW_DIR / f"{key}.json"
        if html_file.exists() and meta_file.exists():
            meta = json.loads(meta_file.read_text())
            return html_file.read_text(), meta["fetched_at"]
        if not self._allowed(url):
            print(f"  BLOCKED by robots.txt: {url}")
            return None
        self._throttle()
        try:
            r = requests.get(url, headers={"User-Agent": self.ua}, timeout=30)
        except requests.RequestException as e:
            print(f"  FETCH ERROR {url}: {e}")
            return None
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {url}")
            return None
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        html_file.write_text(r.text)
        meta_file.write_text(json.dumps({"url": url, "fetched_at": fetched_at}))
        return r.text, fetched_at


def html_to_markdown(html: str) -> tuple[str, str]:
    """Return (page_title, markdown of the main content)."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    title = re.sub(r"\s*[-|–]\s*(Irish Immigration.*|Citizens Information.*|.*enterprise\.gov\.ie.*)$",
                   "", title, flags=re.I).strip() or title
    main = soup.find("main") or soup.find("article") or soup.body or soup
    for tag in main.find_all(["nav", "header", "footer", "aside", "script",
                              "style", "form", "noscript", "iframe", "svg", "button"]):
        tag.decompose()
    md = markdownify(str(main), heading_style="ATX", strip=["a", "img"])
    md = re.sub(r"\n{3,}", "\n\n", md)
    return title, md.strip()


def chunk_markdown(md: str, max_tokens: int = MAX_TOKENS) -> list[dict]:
    """Split markdown on headings, then cap sections at ~max_tokens.

    Returns [{"heading_path": "H1 > H2", "content": "..."}].
    """
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
    sections: list[tuple[list[str], list[str]]] = []  # (path, lines)
    path: list[tuple[int, str]] = []
    lines: list[str] = []

    def flush():
        if any(l.strip() for l in lines):
            sections.append(([h for _, h in path], lines[:]))
        lines.clear()

    for line in md.splitlines():
        m = heading_re.match(line)
        if m:
            flush()
            level, text = len(m.group(1)), m.group(2).strip()
            path = [(lv, t) for lv, t in path if lv < level] + [(level, text)]
        else:
            lines.append(line)
    flush()

    chunks = []
    for hpath, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        parts = [text]
        if est_tokens(text) > max_tokens:
            parts, cur = [], ""
            for para in re.split(r"\n\s*\n", text):
                if cur and est_tokens(cur + "\n\n" + para) > max_tokens:
                    parts.append(cur)
                    cur = para
                else:
                    cur = f"{cur}\n\n{para}" if cur else para
            if cur:
                parts.append(cur)
        for part in parts:
            part = part.strip()
            if est_tokens(part) >= 20:  # drop nav crumbs / boilerplate slivers
                chunks.append({"heading_path": " > ".join(hpath), "content": part})
    return chunks


def build_db(rows: list[dict]):
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            source_url TEXT NOT NULL,
            page_title TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            content TEXT NOT NULL,
            fetched_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            content, page_title, heading_path,
            content='chunks', content_rowid='id'
        );
    """)
    con.executemany(
        "INSERT INTO chunks (source_url, page_title, heading_path, content, fetched_at)"
        " VALUES (:source_url, :page_title, :heading_path, :content, :fetched_at)", rows)
    con.execute("INSERT INTO chunks_fts(rowid, content, page_title, heading_path)"
                " SELECT id, content, page_title, heading_path FROM chunks")
    con.commit()
    con.close()


def build_embeddings(rows: list[dict]):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"{r['page_title']} — {r['heading_path']}\n{r['content']}" for r in rows]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(EMB_PATH, emb.astype("float32"))


def main():
    cfg = load_sources()
    fetcher = Fetcher(cfg["user_agent"], cfg["delay_seconds"])
    rows, pages = [], 0
    for url in cfg["urls"]:
        result = fetcher.fetch(url)
        if result is None:
            continue
        html, fetched_at = result
        title, md = html_to_markdown(html)
        page_chunks = chunk_markdown(md)
        for c in page_chunks:
            rows.append({"source_url": url, "page_title": title,
                         "heading_path": c["heading_path"],
                         "content": c["content"], "fetched_at": fetched_at})
        pages += 1
        print(f"  {len(page_chunks):3d} chunks  {title[:60]}")
    if not rows:
        sys.exit("No chunks ingested — aborting.")
    build_db(rows)
    build_embeddings(rows)
    print(f"\nIngested {pages} pages -> {len(rows)} chunks")
    print(f"DB: {DB_PATH}  Embeddings: {EMB_PATH}")


if __name__ == "__main__":
    main()
