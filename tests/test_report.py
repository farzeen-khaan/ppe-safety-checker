"""Unit tests for F4: severity rules, payload, template and fallback. No real API calls."""

import pytest

import src.report as report
from src.detect import Detection
from src.report import LLMReport, build_payload, compute_severity, generate_report, template_report
from src.violations import PersonVerdict


def person(has_helmet, has_vest) -> PersonVerdict:
    """A verdict built directly; None/None means too far to judge."""
    box = Detection(class_id=5, class_name="Person", confidence=0.9, x1=0, y1=0, x2=100, y2=300)
    return PersonVerdict(person=box, has_helmet=has_helmet, has_vest=has_vest)


OK = person(True, True)
NO_HELMET = person(False, True)
NO_VEST = person(True, False)
NO_BOTH = person(False, False)
FAR = person(None, None)


# --- Severity (PRD rule) ---

@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ([], "None"),
        ([OK, OK], "None"),
        ([OK, FAR, FAR, FAR], "None"),  # too-far people never change severity
        ([NO_HELMET], "Medium"),
        ([NO_HELMET, NO_VEST, OK], "Medium"),
        ([NO_HELMET, NO_VEST, NO_HELMET], "High"),  # 3 or more in violation
        ([NO_BOTH], "High"),  # anyone missing both items
    ],
)
def test_severity(verdicts, expected):
    assert compute_severity(verdicts) == expected


# --- Payload sent to the LLM ---

def test_payload_counts():
    payload = build_payload([OK, NO_HELMET, NO_VEST, NO_BOTH, FAR])
    assert payload["people_detected"] == 5
    assert payload["compliant"] == 1
    assert payload["checked_for_ppe"] == 4
    assert payload["in_violation"] == 3
    assert payload["too_far_to_judge"] == 1
    assert payload["missing_helmet_only"] == 1
    assert payload["missing_vest_only"] == 1
    assert payload["missing_helmet_and_vest"] == 1
    assert payload["severity"] == "High"


def test_payload_contains_no_coordinates_or_image():
    # The PRD sends facts only. Box coordinates are not needed by the LLM either.
    text = str(build_payload([OK, NO_HELMET]))
    for forbidden in ("x1", "y1", "image", "pixel"):
        assert forbidden not in text


# --- Template report ---

def test_template_all_compliant():
    summary, action = template_report(build_payload([OK, OK]))
    assert "All 2 workers" in summary
    assert action == "No action needed for the workers checked."


def test_template_violations_name_each_type():
    summary, action = template_report(build_payload([NO_HELMET, NO_BOTH, OK]))
    assert "2 of 3 workers" in summary
    assert "1 missing a helmet" in summary
    assert "1 missing both" in summary
    assert "flagged workers" in action


def test_template_mentions_too_far_workers():
    summary, action = template_report(build_payload([OK, FAR, FAR]))
    assert "2 workers were too far away to check" in summary
    assert "distant workers" in action


def test_template_all_too_far_never_says_compliant():
    # Regression: this once read "All 0 workers checked were wearing a helmet... No action needed."
    summary, action = template_report(build_payload([FAR, FAR, FAR]))
    assert "3 workers detected, but all were too far away" in summary
    assert "No action needed" not in action
    assert "compliant" not in summary and "wearing a helmet and a vest" not in summary


def test_template_singular_wording():
    summary, _ = template_report(build_payload([OK]))
    assert summary == "The worker checked was wearing a helmet and a vest."
    summary, _ = template_report(build_payload([NO_HELMET]))
    assert summary.startswith("1 of 1 worker checked")


def test_template_no_people():
    summary, _ = template_report(build_payload([]))
    assert "No workers were detected" in summary


# --- Fallback: the report never fails because of the LLM ---

def test_uses_llm_answer_when_it_works(monkeypatch):
    monkeypatch.setattr(report, "ask_llm", lambda payload: (LLMReport(summary="S.", recommended_action="A."), "test-model"))
    r = generate_report([NO_HELMET])
    assert (r.source, r.summary, r.recommended_action, r.severity) == ("gemini", "S.", "A.", "Medium")
    assert r.model == "test-model"


def test_falls_back_when_llm_raises(monkeypatch):
    def broken(payload):
        raise TimeoutError("no answer")

    monkeypatch.setattr(report, "ask_llm", broken)
    r = generate_report([NO_HELMET])
    assert r.source == "template"
    assert "TimeoutError" in r.error
    assert r.severity == "Medium"  # severity is unaffected by the LLM failing


def test_falls_back_on_empty_llm_answer(monkeypatch):
    monkeypatch.setattr(report, "ask_llm", lambda payload: (LLMReport(summary="  ", recommended_action="A."), "test-model"))
    assert generate_report([NO_HELMET]).source == "template"


def test_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = generate_report([NO_HELMET])
    assert r.source == "template"
    assert "GEMINI_API_KEY" in r.error


def test_no_llm_never_calls_llm(monkeypatch):
    def must_not_be_called(payload):
        raise AssertionError("LLM was called")

    monkeypatch.setattr(report, "ask_llm", must_not_be_called)
    r = generate_report([NO_HELMET], use_llm=False)
    assert r.source == "template"


# --- Model fallback order ---

def test_tries_next_model_when_first_fails(monkeypatch):
    tried = []

    def fake_gemini(payload, model, *options):
        tried.append(model)
        if model == "first":
            raise ConnectionError("503 overloaded")
        return LLMReport(summary="S.", recommended_action="A.")

    monkeypatch.setattr(report, "DEFAULT_MODELS", ["first", "second"])
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(report, "ask_gemini", fake_gemini)
    r = generate_report([NO_HELMET])
    assert tried == ["first", "second"]
    assert (r.source, r.model) == ("gemini", "second")


def test_all_models_fail_gives_template_with_every_error(monkeypatch):
    def always_fails(payload, model, *options):
        raise ConnectionError(f"{model} down")

    monkeypatch.setattr(report, "DEFAULT_MODELS", ["first", "second"])
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(report, "ask_gemini", always_fails)
    r = generate_report([NO_HELMET])
    assert r.source == "template"
    assert "first down" in r.error and "second down" in r.error


def test_gemini_model_setting_is_tried_first(monkeypatch):
    monkeypatch.setattr(report, "DEFAULT_MODELS", ["a", "b"])
    monkeypatch.setenv("GEMINI_MODEL", "mine")
    assert report.gemini_models() == ["mine", "a", "b"]
    monkeypatch.setenv("GEMINI_MODEL", "")  # empty line in .env counts as not set
    assert report.gemini_models() == ["a", "b"]
