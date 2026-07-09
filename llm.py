"""OpenAI-compatible chat client. One code path for Groq and Ollama.

Default: Groq free tier. For Ollama set e.g.
  OPENAI_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.1
"""

import os
from pathlib import Path

from openai import OpenAI


def _load_dotenv():
    """Load KEY=VALUE lines from .env next to this file (stdlib, no dep).

    Real environment variables win; .env only fills gaps.
    """
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv()

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
# Set LLM_REASONING_EXCLUDE=1 for reasoning models behind OpenRouter
# (e.g. nemotron) so chain-of-thought is stripped from the reply.
EXTRA_BODY = ({"reasoning": {"exclude": True}}
              if os.environ.get("LLM_REASONING_EXCLUDE") == "1" else None)


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
        extra_body=EXTRA_BODY,
    )
    return resp.choices[0].message.content or ""
