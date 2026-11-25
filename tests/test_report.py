"""Tests for F4: severity rules, the facts sent to the model, the template, and the fallback.

The model is always a fake (llm.generate is replaced); nothing calls the real API.
"""

import pytest

from src import llm, report
from src.verdicts import Detection
from src.report import LLMReport, build_payload, compute_severity, generate_report, template_report
from src.violations import PersonVerdict


def person(has_helmet, has_vest) -> PersonVerdict:
    """None/None means too far to judge."""
    box = Detection(class_id=5, class_name="Person", confidence=0.9, x1=0, y1=0, x2=100, y2=300)
    return PersonVerdict(person=box, has_helmet=has_helmet, has_vest=has_vest)


OK = person(True, True)
NO_HELMET = person(False, True)
NO_VEST = person(True, False)
NO_BOTH = person(False, False)
FAR = person(None, None)


def answer(summary="S.", action="A.", model="model-x"):
    return lambda prompt, facts, schema: llm.Generated(schema(summary=summary, recommended_action=action), "acme", model)


# --- Severity ---

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


# --- The facts sent to the model ---

def test_payload_counts():
    assert build_payload([OK, NO_HELMET, NO_VEST, NO_BOTH, FAR]) == {
        "people_detected": 5, "checked_for_ppe": 4, "compliant": 1, "in_violation": 3, "too_far_to_judge": 1,
        "missing_helmet_only": 1, "missing_vest_only": 1, "missing_helmet_and_vest": 1, "severity": "High",
    }


def test_payload_is_counts_only():
    """No image, no boxes, nothing per person: every value is a number, except the severity."""
    facts = build_payload([OK, NO_HELMET, FAR])
    assert all(isinstance(v, int) for k, v in facts.items() if k != "severity")
    for forbidden in ("x1", "image", "pixel", "people", "confidence"):
        assert forbidden not in facts


def test_payload_adds_up():
    facts = build_payload([OK, NO_HELMET, NO_VEST, NO_BOTH, NO_BOTH, FAR, FAR])
    assert facts["checked_for_ppe"] + facts["too_far_to_judge"] == facts["people_detected"]
    assert facts["compliant"] + facts["in_violation"] == facts["checked_for_ppe"]
    assert facts["missing_helmet_only"] + facts["missing_vest_only"] + facts["missing_helmet_and_vest"] == facts["in_violation"]


def test_model_gets_the_payload_and_the_rules(monkeypatch):
    sent = {}

    def fake(prompt, facts, schema):
        sent.update(prompt=prompt, facts=facts, schema=schema)
        return answer()(prompt, facts, schema)

    monkeypatch.setattr(llm, "generate", fake)
    generate_report([NO_BOTH, FAR])
    assert sent["facts"] == build_payload([NO_BOTH, FAR]) and sent["schema"] is LLMReport
    prompt = sent["prompt"]
    assert "{facts}" in prompt  # filled in by llm.generate
    for rule in ("Use only the facts", "it is important to", "never call them compliant", "camera",
                 "site supervisor", "do not repeat them", "Do not restate the counts", "never say \"stop work\"",
                 "Verify on site, then correct:", "2 or 3 short sentences"):
        assert rule.lower() in prompt.lower(), rule


# --- Template report: what stands out, then verify-first action ---

def sentences(text: str) -> int:
    return len([part for part in text.replace("…", "").split(". ") if part.strip()])


def test_template_all_compliant():
    summary, action = template_report(build_payload([OK, OK]))
    assert summary == "Every worker who could be checked wears a helmet and a vest."
    assert action == "No PPE action needed."


def test_template_says_what_stands_out_without_repeating_counts():
    summary, action = template_report(build_payload([NO_HELMET, NO_BOTH, NO_BOTH, OK]))
    assert summary == "Most flagged workers are missing both helmets and vests."
    assert action == "Verify on site, then correct: make sure the flagged workers wear a helmet and a high-visibility vest."
    assert not any(ch.isdigit() for ch in summary + action)  # the counts are shown next to the text


@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ([NO_VEST], "The flagged worker is missing a vest."),
        ([NO_HELMET, NO_HELMET], "All flagged workers are missing helmets."),
        ([NO_HELMET, NO_VEST], "The flagged workers are missing a mix of helmets and vests."),
        ([NO_BOTH, NO_BOTH, NO_HELMET], "Most flagged workers are missing both helmets and vests."),
    ],
)
def test_template_pattern(verdicts, expected):
    assert template_report(build_payload(verdicts))[0] == expected


def test_template_names_only_the_missing_items():
    assert template_report(build_payload([NO_VEST]))[1] == (
        "Verify on site, then correct: make sure the flagged worker wears a high-visibility vest."
    )
    assert template_report(build_payload([NO_HELMET, NO_HELMET]))[1].endswith("wear a helmet.")


def test_template_never_says_stop_work():
    for verdicts in ([NO_HELMET], [NO_BOTH, FAR], [NO_VEST, NO_HELMET, NO_BOTH], [OK], [FAR], []):
        assert "stop work" not in " ".join(template_report(build_payload(verdicts))).lower()


def test_template_distance_is_a_camera_problem():
    summary, action = template_report(build_payload([NO_HELMET, OK, OK, FAR]))
    assert summary.endswith("Some workers were too far from the camera to check.")
    assert action.endswith(", and move the camera closer or retake the photo so every worker can be checked.")
    summary, _ = template_report(build_payload([OK, FAR, FAR]))
    assert summary.endswith("Half or more of the people detected were too far from the camera to check.")


def test_template_all_too_far_never_says_compliant():
    summary, action = template_report(build_payload([FAR, FAR, FAR]))
    assert summary == "No worker could be checked: everyone in the photo was too far from the camera."
    assert action == "Move the camera closer or retake the photo so every worker can be checked."
    assert "compliant" not in summary and "wears a helmet" not in summary


def test_template_no_people():
    summary, action = template_report(build_payload([]))
    assert summary == "No workers were detected in this photo."
    assert action.startswith("No PPE action needed.")


@pytest.mark.parametrize("verdicts", [[OK], [NO_BOTH, FAR], [FAR], [], [NO_HELMET, NO_VEST, NO_BOTH, FAR, FAR]])
def test_template_is_two_or_three_short_sentences_without_filler(verdicts):
    summary, action = template_report(build_payload(verdicts))
    assert 2 <= sentences(summary) + sentences(action) <= 3
    text = (summary + " " + action).lower()
    assert "important" not in text and "please" not in text and "overall" not in text


# --- The model, and the fallback ---

def test_uses_the_model_answer_and_keeps_provenance(monkeypatch):
    monkeypatch.setattr(llm, "generate", answer("Short summary.", "Do this."))
    r = generate_report([NO_HELMET])
    assert (r.summary, r.recommended_action, r.severity) == ("Short summary.", "Do this.", "Medium")
    assert (r.source, r.provider, r.model, r.error, r.automated) == ("llm", "acme", "model-x", None, True)


def test_falls_back_when_the_model_is_unavailable(monkeypatch):
    def unavailable(prompt, facts, schema):
        raise llm.LLMUnavailable("a: 503 | b: timeout")

    monkeypatch.setattr(llm, "generate", unavailable)
    r = generate_report([NO_HELMET, FAR])
    assert (r.source, r.provider, r.model, r.automated) == ("template", None, None, False)
    assert r.error == "LLMUnavailable: a: 503 | b: timeout"  # kept for diagnostics, never shown
    assert r.summary.startswith("The flagged worker is missing a helmet.")


@pytest.mark.parametrize("summary, action", [("  ", "A."), ("S.", ""), ("x" * 801, "A.")])
def test_falls_back_on_empty_or_runaway_answers(monkeypatch, summary, action):
    monkeypatch.setattr(llm, "generate", answer(summary, action))
    assert generate_report([NO_HELMET]).source == "template"


def test_no_llm_never_calls_the_model(monkeypatch):
    monkeypatch.setattr(llm, "generate", lambda *a: pytest.fail("must not be called"))
    r = generate_report([NO_HELMET], use_llm=False)
    assert (r.source, r.error) == ("template", "LLM disabled")


def test_without_a_key_the_report_still_works(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = generate_report([NO_HELMET])
    assert r.source == "template" and "LLM_API_KEY is not set" in r.error
