"""F4 — Incident report: severity by rules, wording by Gemini, template as fallback.

Usage:
    python -m src.report IMAGE [--conf 0.25] [--no-llm]

Only the structured verdict counts are sent to Gemini, never the image.
Severity is computed here by rules; the LLM only writes the summary and the action.
If Gemini fails for any reason (no key, timeout, quota, bad output), a template report is used.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Light imports only: the digest imports this module and must not load torch or OpenCV.
# main() imports the detection model itself.
from src.verdicts import PersonVerdict, count_verdicts

load_dotenv()  # reads .env into environment variables; real environment variables win

# Models tried in order; then the template. Measured Nov 5, 2026 on the free tier: full Flash
# models returned 503 (overloaded) or timed out after ~29 s; Flash-Lite models answered in ~2 s.
# A Lite model is plenty for turning a few counts into three sentences.
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
# Per model. The Gemini API rejects deadlines under 10 s (400 INVALID_ARGUMENT), so 10 s is
# the shortest allowed. Measured over 24 calls: median success 2 s, 11 of 12 under 6 s, and
# failures (503 overloaded, 429 rate limit) usually arrive within 2 s.
LLM_TIMEOUT_MS = 10_000
MAX_FIELD_CHARS = 800  # reject runaway answers


@dataclass
class Report:
    severity: str  # "High", "Medium" or "None"
    summary: str
    recommended_action: str
    source: str  # "gemini" or "template"
    model: str | None = None  # which Gemini model wrote it
    error: str | None = None  # why the template was used, if it was


class LLMReport(BaseModel):
    """The exact shape we ask Gemini to return."""

    summary: str
    recommended_action: str


def compute_severity(verdicts: list[PersonVerdict]) -> str:
    """PRD rule: High = 3+ people in violation or anyone missing both items; Medium = 1-2; None = 0.

    People too far to judge never change severity.
    """
    violations = [v for v in verdicts if v.is_violation]
    if len(violations) >= 3 or any(len(v.missing) == 2 for v in violations):
        return "High"
    if violations:
        return "Medium"
    return "None"


def build_payload(verdicts: list[PersonVerdict]) -> dict:
    """The facts sent to the LLM: counts and per-person statuses. No image, no pixels."""
    counts = count_verdicts(verdicts)
    return {
        "people_detected": counts["people"],
        # Stated explicitly: without it, Gemini reported all detected people as "checked".
        "checked_for_ppe": counts["people"] - counts["too_far"],
        "compliant": counts["compliant"],
        "in_violation": counts["violations"],
        "too_far_to_judge": counts["too_far"],
        "missing_helmet_only": sum(v.missing == ["helmet"] for v in verdicts),
        "missing_vest_only": sum(v.missing == ["vest"] for v in verdicts),
        "missing_helmet_and_vest": sum(len(v.missing) == 2 for v in verdicts),
        "severity": compute_severity(verdicts),
        "people": [{"id": i, "status": v.status} for i, v in enumerate(verdicts, 1)],
    }


def workers(n: int) -> str:
    """'1 worker', '3 workers'."""
    return f"{n} worker" if n == 1 else f"{n} workers"


def template_report(payload: dict) -> tuple[str, str]:
    """Fallback wording built only from the counts. Returns (summary, recommended_action)."""
    too_far = payload["too_far_to_judge"]
    judged = payload["people_detected"] - too_far
    violations = payload["in_violation"]

    if payload["people_detected"] == 0:
        return (
            "No workers were detected in this photo.",
            "No action needed. If workers should be visible, take a closer or clearer photo.",
        )

    if judged == 0:
        # Nobody could be checked: never say "all compliant" or "no action needed" here.
        return (
            f"{workers(too_far)} detected, but all were too far away to check for a helmet and vest.",
            "Check these workers in person or take a closer photo.",
        )

    if violations == 0:
        summary = "The worker checked was wearing a helmet and a vest." if judged == 1 else (
            f"All {judged} workers checked were wearing a helmet and a vest."
        )
        action = "No action needed for the workers checked."
    else:
        parts = []
        for key, label in (
            ("missing_helmet_only", "missing a helmet"),
            ("missing_vest_only", "missing a vest"),
            ("missing_helmet_and_vest", "missing both a helmet and a vest"),
        ):
            if payload[key]:
                parts.append(f"{payload[key]} {label}")
        summary = f"{violations} of {workers(judged)} checked not wearing required PPE: {', '.join(parts)}."
        action = (
            f"Stop work for the flagged {'worker' if violations == 1 else 'workers'} "
            "until they wear a helmet and a high-visibility vest, and record the incident."
        )

    if too_far:
        summary += f" {workers(too_far)} {'was' if too_far == 1 else 'were'} too far away to check."
        action += f" Check the distant {'worker' if too_far == 1 else 'workers'} in person or with a closer photo."
    return summary, action


PROMPT = """You write short incident reports for a construction-site safety officer.
Use ONLY the facts in the JSON below. Do not invent names, locations, causes or numbers.
The severity is already decided; do not change or question it.

Write:
- summary: 2-3 plain sentences stating how many workers were checked (checked_for_ppe),
  who is missing what, and how many were too far to check (if any). Workers too far to
  check were not checked; never call them compliant.
- recommended_action: 1-2 sentences with a concrete next step for the safety officer.
  If too_far_to_judge > 0, include checking those workers.

Facts:
{facts}
"""


def gemini_models() -> list[str]:
    """Models to try in order. GEMINI_MODEL in .env, if set, is tried first."""
    preferred = os.environ.get("GEMINI_MODEL")  # empty (GEMINI_MODEL=) counts as not set
    return [preferred, *DEFAULT_MODELS] if preferred else list(DEFAULT_MODELS)


def ask_gemini(payload: dict, model: str, prompt: str = PROMPT, schema: type[BaseModel] = LLMReport) -> BaseModel:
    """Send the facts to one Gemini model and return its structured answer. Raises on any failure.

    prompt (with a {facts} placeholder) and schema default to the incident report; the digest passes its own.
    """
    from google import genai  # imported here so the rest of the module works without it
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_MS))
    response = client.models.generate_content(
        model=model,
        contents=prompt.format(facts=json.dumps(payload, indent=2)),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,  # low randomness: reports should read consistently
        ),
    )
    if response.parsed is None:
        raise ValueError("Gemini returned no parsable report")
    return response.parsed


def ask_llm(payload: dict, prompt: str = PROMPT, schema: type[BaseModel] = LLMReport) -> tuple[BaseModel, str]:
    """The single place that picks the LLM provider; swap Gemini for Groq here if needed.

    Tries each Gemini model in turn and returns (answer, model). Raises the last error if all fail.
    """
    errors = []
    for model in gemini_models():
        try:
            return ask_gemini(payload, model, prompt, schema), model
        except Exception as exc:
            errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:300]}")
            if "GEMINI_API_KEY" in str(exc):
                break  # no key: other models won't help
    raise RuntimeError(" | ".join(errors))


def generate_report(verdicts: list[PersonVerdict], use_llm: bool = True) -> Report:
    """Build the incident report. Never raises because of the LLM."""
    payload = build_payload(verdicts)
    severity = payload["severity"]

    if use_llm:
        try:
            answer, model = ask_llm(payload)
            summary, action = answer.summary.strip(), answer.recommended_action.strip()
            if not summary or not action:
                raise ValueError("Gemini returned an empty field")
            if len(summary) > MAX_FIELD_CHARS or len(action) > MAX_FIELD_CHARS:
                raise ValueError("Gemini returned an overly long field")
            return Report(severity, summary, action, source="gemini", model=model)
        except Exception as exc:  # any LLM problem falls back to the template
            summary, action = template_report(payload)
            return Report(severity, summary, action, source="template", error=f"{type(exc).__name__}: {exc}")

    summary, action = template_report(payload)
    return Report(severity, summary, action, source="template", error="LLM disabled")


def main() -> None:
    import cv2  # heavy: only the command line needs them

    from src.detect import DEFAULT_CONF, detect
    from src.violations import check_people

    parser = argparse.ArgumentParser(description="Write an incident report for an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--no-llm", action="store_true", help="use the template report only")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"{args.image}: not a readable image")

    verdicts = check_people(detect(image, conf=args.conf), image_height=image.shape[0])
    report = generate_report(verdicts, use_llm=not args.no_llm)

    print(f"Facts sent to the LLM:\n{json.dumps(build_payload(verdicts), indent=2)}\n")
    print(f"Severity: {report.severity}")
    print(f"Summary:  {report.summary}")
    print(f"Action:   {report.recommended_action}")
    detail = report.model or report.error
    print(f"Source:   {report.source}" + (f" ({detail})" if detail else ""))


if __name__ == "__main__":
    main()
