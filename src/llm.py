"""Structured text from a hosted language model. The only module that knows the provider.

    result = llm.generate(PROMPT, facts, Schema)   # -> Generated(value, provider, model)

Callers pass a prompt with a {facts} placeholder, the facts as a dict and a pydantic schema.
Each model in models() is tried in turn with a TIMEOUT_MS deadline; if all fail,
LLMUnavailable is raised and the caller uses its own template. Callers never import the SDK.

Provider: Google Gemini (why: decisions.md, Phase 3). Config: LLM_API_KEY and optional
LLM_MODEL. (The old GEMINI_API_KEY / GEMINI_MODEL names were removed in Phase 8b.)
"""

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROVIDER = "gemini"
# Flash-Lite: on the free tier the full Flash models were overloaded or timed out; Lite answers in about 2 s.
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
# Per model. The API rejects deadlines under 10 s, so this is the shortest allowed.
TIMEOUT_MS = 10_000
TEMPERATURE = 0.2  # low: summaries should read the same way each time


class LLMUnavailable(RuntimeError):
    """No model produced an answer. The message lists each model's error."""


@dataclass(frozen=True)
class Generated:
    value: BaseModel
    provider: str
    model: str


def _setting(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def api_key() -> str | None:
    return _setting("LLM_API_KEY")


def models() -> list[str]:
    """LLM_MODEL first if set, then the defaults."""
    preferred = _setting("LLM_MODEL")
    return [preferred, *DEFAULT_MODELS] if preferred else list(DEFAULT_MODELS)


def _call(text: str, schema: type[BaseModel], model: str) -> BaseModel:
    """One request to one model. Raises on any failure."""
    from google import genai  # imported here so the app starts without it
    from google.genai import types

    key = api_key()
    if not key:
        raise RuntimeError("LLM_API_KEY is not set")
    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema, temperature=TEMPERATURE
        ),
    )
    if response.parsed is None:
        raise ValueError("no parsable answer")
    return response.parsed


def generate(prompt: str, facts: dict, schema: type[BaseModel]) -> Generated:
    """Fill the prompt with the facts and return the first model's structured answer."""
    text = prompt.format(facts=json.dumps(facts, indent=2))
    errors = []
    for model in models():
        try:
            return Generated(_call(text, schema, model), PROVIDER, model)
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:300]}")
            if "LLM_API_KEY" in str(exc):
                break  # no key: other models won't help
    raise LLMUnavailable(" | ".join(errors))
