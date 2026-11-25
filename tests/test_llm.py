"""Tests for src/llm.py: configuration, model order, fallback and errors. The SDK call is a fake."""

import json
import logging

import pytest
from pydantic import BaseModel

from src import llm


class Answer(BaseModel):
    text: str


@pytest.fixture
def calls(monkeypatch):
    """Replace the one SDK call; answers are queued per model name."""
    record = {"models": [], "texts": [], "answers": {}}

    def fake_call(text, schema, model):
        record["models"].append(model)
        record["texts"].append(text)
        result = record["answers"].get(model, RuntimeError("503 overloaded"))
        if isinstance(result, Exception):
            raise result
        return schema(text=result)

    monkeypatch.setattr(llm, "_call", fake_call)
    monkeypatch.setattr(llm, "DEFAULT_MODELS", ["first", "second"])
    return record


# --- Configuration ---

def test_new_names_are_read(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "new-key")
    monkeypatch.setenv("GEMINI_API_KEY", "old-key")
    monkeypatch.setenv("LLM_MODEL", "mine")
    monkeypatch.setattr(llm, "DEFAULT_MODELS", ["a", "b"])
    assert llm.api_key() == "new-key"
    assert llm.models() == ["mine", "a", "b"]


def test_old_names_still_work_for_one_release_with_a_warning(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "old-key")
    monkeypatch.setenv("GEMINI_MODEL", "old-model")
    monkeypatch.setattr(llm, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="src.llm"):
        assert llm.api_key() == "old-key"
        assert llm.api_key() == "old-key"
        assert llm.models()[0] == "old-model"
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings.count("GEMINI_API_KEY is deprecated; rename it to LLM_API_KEY.") == 1  # once, not per call
    assert "GEMINI_MODEL is deprecated; rename it to LLM_MODEL." in warnings


def test_no_key_and_no_model_setting(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "  ")  # blank counts as unset
    monkeypatch.setattr(llm, "DEFAULT_MODELS", ["a"])
    assert llm.api_key() is None
    assert llm.models() == ["a"]


def test_missing_key_is_a_clear_error_without_calling_the_api(monkeypatch):
    with pytest.raises(llm.LLMUnavailable, match="LLM_API_KEY is not set"):
        llm.generate("{facts}", {}, Answer)


# --- generate(): structured output, timeout per model, fallback across models ---

def test_returns_the_first_answer_with_provenance(calls):
    calls["answers"]["first"] = "hello"
    result = llm.generate("Facts:\n{facts}", {"in_violation": 2}, Answer)
    assert result == llm.Generated(Answer(text="hello"), llm.PROVIDER, "first")
    assert calls["models"] == ["first"]


def test_facts_are_sent_as_json_in_the_prompt(calls):
    calls["answers"]["first"] = "ok"
    llm.generate("Facts:\n{facts}", {"in_violation": 2, "too_far_to_judge": 1}, Answer)
    sent = calls["texts"][0]
    assert sent.startswith("Facts:\n")
    assert json.loads(sent.removeprefix("Facts:\n")) == {"in_violation": 2, "too_far_to_judge": 1}


def test_falls_back_to_the_next_model(calls):
    calls["answers"]["second"] = "from second"
    result = llm.generate("{facts}", {}, Answer)
    assert (result.model, result.value.text) == ("second", "from second")
    assert calls["models"] == ["first", "second"]


def test_all_models_failing_raises_with_every_error(calls):
    calls["answers"] = {"first": TimeoutError("deadline"), "second": ValueError("no parsable answer")}
    with pytest.raises(llm.LLMUnavailable) as caught:
        llm.generate("{facts}", {}, Answer)
    assert str(caught.value) == "first: TimeoutError: deadline | second: ValueError: no parsable answer"


def test_a_missing_key_stops_after_the_first_model(calls):
    calls["answers"]["first"] = RuntimeError("LLM_API_KEY is not set")
    with pytest.raises(llm.LLMUnavailable):
        llm.generate("{facts}", {}, Answer)
    assert calls["models"] == ["first"]  # other models would fail the same way


def test_timeout_is_the_apis_minimum():
    assert llm.TIMEOUT_MS == 10_000  # the API rejects shorter deadlines


def test_callers_never_import_the_sdk():
    """The provider SDK is imported only inside src/llm.py."""
    from pathlib import Path

    src = Path(llm.__file__).parent
    for path in src.glob("*.py"):
        if path.name != "llm.py":
            text = path.read_text(encoding="utf-8")
            assert "google" not in text and "genai" not in text, path.name
