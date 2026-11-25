"""Live tests against the real model API. Skipped unless RUN_LIVE_TESTS=1.

Unit tests replace the model with fakes, so they can't catch the real API's rules (for
example, it rejects timeouts under 10 s) or check how real summaries read.

    $env:RUN_LIVE_TESTS = "1"; python -m pytest -m live -s      # PowerShell; -s prints the samples
"""

import json
import os
import re

import pytest

from src import digest, llm, report
from src.verdicts import Detection
from src.violations import PersonVerdict

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to call the real API"),
]

FILLER = ("it is important", "please note", "overall", "it's important", "ensure that all")


def person(has_helmet, has_vest):
    return PersonVerdict(Detection(5, "Person", 0.9, 0, 0, 100, 300), has_helmet, has_vest)


def numbers_in(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\b\d+\b", text)}


def check(text: str, facts: dict) -> None:
    """The summary rules, as far as code can check them."""
    lower = text.lower()
    assert not any(f in lower for f in FILLER), text
    assert "gemini" not in lower and "llm" not in lower, text
    allowed = {v for v in facts.values() if isinstance(v, int)}
    assert numbers_in(text) <= allowed | {1, 2, 3}, f"numbers not in the facts: {numbers_in(text) - allowed}"
    if facts.get("too_far_to_judge"):
        assert "camera" in lower or "photo" in lower, "distance must be framed as a camera problem"
    assert "stop work" not in lower, "the check can be wrong: verify first"
    if facts.get("in_violation"):
        assert "verify on site, then correct" in lower, text
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    assert len(sentences) <= 3, f"{len(sentences)} sentences: {text}"


def test_every_model_accepts_the_report_request():
    """A 400 means our request is wrong; overload (503) and rate limits (429) are the service's state."""
    text = report.PROMPT.format(facts=json.dumps(report.build_payload([person(False, True)])))
    for model in llm.models():
        try:
            answer = llm._call(text, report.LLMReport, model)
        except Exception as exc:
            assert "400" not in str(exc), f"{model} rejected our request: {exc}"
            continue
        assert answer.summary and answer.recommended_action


SAMPLES = {
    "incident: 13 people, 9 in violation, 3 too far": [person(True, True)] + [person(False, False)] * 7
    + [person(False, True)] * 2 + [person(None, None)] * 3,
    "incident: 1 in violation, nobody too far": [person(True, True), person(True, False)],
}


@pytest.mark.parametrize("name", SAMPLES)
def test_sample_incident_summary(name, capsys):
    verdicts = SAMPLES[name]
    facts = report.build_payload(verdicts)
    r = report.generate_report(verdicts)
    with capsys.disabled():
        print(f"\n--- {name} ({r.source}) ---\nFacts: {json.dumps(facts)}\nSummary: {r.summary}\nAction: {r.recommended_action}")
    if r.source != "llm":
        pytest.skip(f"model unavailable right now: {r.error}")
    check(r.summary + " " + r.recommended_action, facts)


def test_sample_digest_summary(capsys):
    stats = {
        "analyses": 5, "people_detected": 33, "checked_for_ppe": 17, "compliant": 5, "in_violation": 12,
        "too_far_to_judge": 16, "missing_helmet_only": 2, "missing_vest_only": 1, "missing_helmet_and_vest": 9,
        "severity_high": 2, "severity_medium": 1, "severity_none": 2,
    }
    facts = digest.summary_facts(stats)
    text, automated = digest.write_summary(stats)
    with capsys.disabled():
        print(f"\n--- digest: 5 analyses ({'automated' if automated else 'standard'}) ---\nFacts: {json.dumps(facts)}\nSummary: {text}")
    if not automated:
        pytest.skip("model unavailable right now")
    check(text, facts)
