"""Offline unit tests: chunking, RRF math, citation mapping, threshold rule.

No network, no LLM, no index files needed.
"""

import pytest

from answer import extract_citations, map_citations, should_abstain
from ingest import chunk_markdown, est_tokens
from search import normalize_query, rrf_fuse


# --- chunking ---

def test_chunks_split_on_headings_with_paths():
    md = ("# Stamps\nintro text about stamps " + "w " * 30 + "\n"
          "## Stamp 1G\ngraduate permission details " + "w " * 30 + "\n"
          "## Stamp 4\nlong-term residence details " + "w " * 30)
    chunks = chunk_markdown(md)
    assert [c["heading_path"] for c in chunks] == \
        ["Stamps", "Stamps > Stamp 1G", "Stamps > Stamp 4"]
    assert "graduate permission" in chunks[1]["content"]


def test_heading_path_resets_at_sibling_level():
    md = ("# A\n" + "x " * 25 + "\n## B\n" + "x " * 25 + "\n"
          "# C\n" + "y " * 25)
    paths = [c["heading_path"] for c in chunk_markdown(md)]
    assert paths == ["A", "A > B", "C"]


def test_long_sections_split_under_token_cap():
    para = "word " * 150  # ~195 est tokens per paragraph
    md = "# Big\n" + "\n\n".join([para] * 5)
    chunks = chunk_markdown(md, max_tokens=500)
    assert len(chunks) > 1
    assert all(est_tokens(c["content"]) <= 500 for c in chunks)
    assert all(c["heading_path"] == "Big" for c in chunks)


def test_tiny_boilerplate_slivers_dropped():
    md = "# Nav\nHome\n# Real\n" + "content " * 40
    chunks = chunk_markdown(md)
    assert [c["heading_path"] for c in chunks] == ["Real"]


# --- query normalization ---

def test_normalize_query_splits_glued_stamp_names():
    assert normalize_query("what is stamp2") == "what is stamp 2"
    assert normalize_query("what is stamp1g") == "what is stamp 1g"
    assert normalize_query("Stamp 1G rules") == "Stamp 1G rules"  # unchanged


# --- RRF ---

def test_rrf_math_exact():
    scores = rrf_fuse([[1, 2], [2, 3]], k=60)
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 62)


def test_rrf_doc_in_both_lists_beats_single_list_winner():
    scores = rrf_fuse([[1, 2, 3], [9, 2, 8]], k=60)
    assert max(scores, key=scores.get) == 2


def test_rrf_empty_lists():
    assert rrf_fuse([[], []]) == {}


# --- citation mapping ---

def test_extract_citations_ordered_unique_in_range():
    assert extract_citations("A [2] then [1], again [2], bogus [7].", 4) == [2, 1]


def test_map_citations_maps_to_chunks():
    chunks = [{"source_url": "u1"}, {"source_url": "u2"}]
    mapped = map_citations("claim [2] and [1]", chunks)
    assert mapped == [(2, chunks[1]), (1, chunks[0])]


def test_no_citations():
    assert extract_citations("no brackets here", 4) == []


# --- abstention threshold ---

def test_should_abstain_below_threshold():
    assert should_abstain(0.01, threshold=0.02)
    assert not should_abstain(0.03, threshold=0.02)
    assert not should_abstain(0.02, threshold=0.02)  # at threshold -> answer


def test_should_abstain_on_no_results():
    assert should_abstain(None, threshold=0.02)
