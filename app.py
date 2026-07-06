"""Gradio UI: chat with citations + sources/freshness tab."""

import sqlite3

import gradio as gr

import answer as answer_mod
from search import DB_PATH

DISCLAIMER_MD = ("⚠️ **Informational only, not legal advice.** Answers come solely from "
                 "indexed official pages (irishimmigration.ie, citizensinformation.ie, "
                 "enterprise.gov.ie) and may be out of date. Verify with the source links.")


def respond(message: str, history: list) -> str:
    if not message.strip():
        return "Please ask a question."
    result = answer_mod.answer(message)  # single-turn by design: history unused
    text = result["answer"]
    if result["citations"]:
        lines = [f"[{n}] [{c['page_title']} — {c['heading_path']}]({c['source_url']})"
                 for n, c in result["citations"]]
        text += "\n\n**Sources**\n" + "\n".join(lines)
    return text


def sources_table() -> str:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT source_url, page_title, COUNT(*) AS chunks, MAX(fetched_at) AS fetched_at"
        " FROM chunks GROUP BY source_url ORDER BY source_url").fetchall()
    con.close()
    md = ["| Page | Chunks | Fetched |", "|---|---|---|"]
    for url, title, chunks, fetched in rows:
        md.append(f"| [{title}]({url}) | {chunks} | {fetched} |")
    md.append(f"\n**{len(rows)} pages indexed.** Re-run `python ingest.py` to refresh.")
    return "\n".join(md)


with gr.Blocks(title="StampWise Mini") as demo:
    gr.Markdown("# 🇮🇪 StampWise Mini\nCitation-grounded Q&A on Irish immigration "
                "(stamps, student pathways, work permits).")
    gr.Markdown(DISCLAIMER_MD)
    with gr.Tab("Chat"):
        gr.ChatInterface(
            respond,
            examples=["What is Stamp 1G?",
                      "How long can I stay on the Third Level Graduate Programme?",
                      "What is the minimum salary for a Critical Skills Employment Permit?"],
        )
    with gr.Tab("Sources & freshness"):
        gr.Markdown(sources_table())

if __name__ == "__main__":
    demo.launch()
