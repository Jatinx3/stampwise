"""OpenAI-compatible chat client. One code path for Groq and Ollama.

Default: Groq free tier. For Ollama set e.g.
  OPENAI_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.1
"""

import os

from openai import OpenAI

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")


def get_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        # Ollama accepts any key; Groq needs GROQ_API_KEY.
        api_key=os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "ollama",
    )


def chat(messages: list[dict], temperature: float = 0.1, max_tokens: int = 800) -> str:
    resp = get_client().chat.completions.create(
        model=MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
