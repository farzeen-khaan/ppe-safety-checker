"""F4 — Incident report: severity by rules, wording by a language model, template as fallback.

Usage:
    python -m src.report IMAGE [--conf 0.25] [--no-llm]

Only counts go to the model, never the image. Severity comes from rules; the model only
writes the summary and the recommended action. Any model failure gives the template.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from src import llm
from src.verdicts import PersonVerdict, count_verdicts  # light: the digest imports this module

MAX_FIELD_CHARS = 800  # longer answers are treated as failures


@dataclass
class Report:
    severity: str  # "High", "Medium" or "None"
    summary: str
    recommended_action: str
    source: str  # "llm" (automated) or "template" (standard)
    provider: str | None = None  # set when source is "llm"; kept in the log, never shown to users
    model: str | None = None
    error: str | None = None  # why the template was used

    @property
    def automated(self) -> bool:
        return self.source == "llm"


class LLMReport(BaseModel):
    summary: str
    recommended_action: str


def compute_severity(verdicts: list[PersonVerdict]) -> str:
    """High: 3+ in violation, or anyone missing both. Medium: 1-2. None: 0. Too-far workers never count."""
    violations = [v for v in verdicts if v.is_violation]
    if len(violations) >= 3 or any(len(v.missing) == 2 for v in violations):
        return "High"
    if violations:
        return "Medium"
    return "None"


def build_payload(verdicts: list[PersonVerdict]) -> dict:
    """The facts for the model: counts only. No image, no boxes, nothing per person."""
    counts = count_verdicts(verdicts)
    return {
        "people_detected": counts["people"],
        # Stated explicitly: without it, the model counted too-far workers as checked.
        "checked_for_ppe": counts["people"] - counts["too_far"],
        "compliant": counts["compliant"],
        "in_violation": counts["violations"],
        "too_far_to_judge": counts["too_far"],
        "missing_helmet_only": sum(v.missing == ["helmet"] for v in verdicts),
        "missing_vest_only": sum(v.missing == ["vest"] for v in verdicts),
        "missing_helmet_and_vest": sum(len(v.missing) == 2 for v in verdicts),
        "severity": compute_severity(verdicts),
    }


def workers(n: int) -> str:
    """'1 worker', '3 workers'."""
    return f"{n} worker" if n == 1 else f"{n} workers"


def violation_parts(facts: dict) -> str:
    """'1 missing a helmet, 2 missing both helmet and vest'."""
    labels = (
        ("missing_helmet_only", "missing a helmet"),
        ("missing_vest_only", "missing a vest"),
        ("missing_helmet_and_vest", "missing both helmet and vest"),
    )
    return ", ".join(f"{facts[key]} {label}" for key, label in labels if facts[key])


CAMERA_FIX = "move the camera closer or retake the photo so every worker can be checked"
CAMERA_SENTENCE = "Move the camera closer or retake the photo so every worker can be checked."
VERIFY = "Verify on site, then correct"  # the check is automated and can be wrong: never "stop work"


def missing_items(facts: dict) -> str:
    """What the flagged workers need, naming only the missing items."""
    helmet = facts["missing_helmet_only"] or facts["missing_helmet_and_vest"]
    vest = facts["missing_vest_only"] or facts["missing_helmet_and_vest"]
    if helmet and vest:
        return "a helmet and a high-visibility vest"
    return "a helmet" if helmet else "a high-visibility vest"


def pattern(facts: dict, flagged: str = "flagged workers") -> str:
    """The violation pattern in words, without repeating the counts the reader already sees."""
    kinds = {
        "both a helmet and a vest": facts["missing_helmet_and_vest"],
        "a helmet": facts["missing_helmet_only"],
        "a vest": facts["missing_vest_only"],
    }
    plural = {"both a helmet and a vest": "both helmets and vests", "a helmet": "helmets", "a vest": "vests"}
    total = sum(kinds.values())
    kind, n = max(kinds.items(), key=lambda item: item[1])
    if total == 1:
        return f"The flagged worker is missing {kind}."
    if n == total:
        return f"All {flagged} are missing {plural[kind]}."
    if 2 * n > total:
        return f"Most {flagged} are missing {plural[kind]}."
    return f"The {flagged} are missing a mix of helmets and vests."


def distance_note(facts: dict) -> str:
    """How much of the photo could not be checked, in words."""
    if 2 * facts["too_far_to_judge"] >= facts["people_detected"]:
        return "Half or more of the people detected were too far from the camera to check."
    return "Some workers were too far from the camera to check."


def template_report(facts: dict) -> tuple[str, str]:
    """Fallback wording from the counts, without repeating them. Returns (summary, recommended_action)."""
    too_far, checked, violations = facts["too_far_to_judge"], facts["checked_for_ppe"], facts["in_violation"]

    if facts["people_detected"] == 0:
        return (
            "No workers were detected in this photo.",
            "No PPE action needed. If workers should be in view, retake the photo closer or from a better angle.",
        )
    if checked == 0:  # never "all compliant" when nobody was checked
        return (
            "No worker could be checked: everyone in the photo was too far from the camera.",
            CAMERA_SENTENCE,
        )

    if violations == 0:
        summary = "Every worker who could be checked wears a helmet and a vest."
        action = CAMERA_SENTENCE if too_far else "No PPE action needed."
    else:
        summary = pattern(facts)
        who = "the flagged worker wears" if violations == 1 else "the flagged workers wear"
        action = f"{VERIFY}: make sure {who} {missing_items(facts)}"
        action += f", and {CAMERA_FIX}." if too_far else "."
    if too_far:
        summary += " " + distance_note(facts)
    return summary, action


PROMPT = """You write the short incident summary for a construction-site supervisor. The counts
are shown right next to your text, so do not repeat them.

Rules:
- 2 or 3 short sentences in total: summary 1 or 2 sentences, recommended_action 1 sentence.
- Do not restate the counts (people, checked, in violation, too far, or per violation type).
  Say what stands out, in words: which item is missing most, whether it is one worker or most
  of them, whether many workers could not be checked.
- Use only the facts. Do not invent names, places, causes or numbers.
- No filler or lead-ins such as "it is important to", "please note", "overall", "a total of"
  or "specifically".
- Workers in too_far_to_judge were not checked. Never call them compliant.
- Distance is about the camera, not the workers: the fix is camera placement or photo quality.
- The check is automated and can be wrong, so never say "stop work". If anyone is in
  violation, recommended_action starts with "Verify on site, then correct:" and names only the
  items they are missing. If too_far_to_judge > 0, add the camera fix to the same sentence.
  If nobody is in violation and nobody is too far, say no PPE action is needed.
- The severity is already decided; do not mention it.

Example (for 4 in violation, all missing only a helmet, nobody too far):
  summary: "All flagged workers are missing helmets; their vests were seen."
  recommended_action: "Verify on site, then correct: make sure the flagged workers wear a helmet."

Facts:
{facts}
"""


def generate_report(verdicts: list[PersonVerdict], use_llm: bool = True) -> Report:
    """Build the incident report. Never raises because of the model."""
    facts = build_payload(verdicts)
    severity = facts["severity"]

    if not use_llm:
        summary, action = template_report(facts)
        return Report(severity, summary, action, source="template", error="LLM disabled")
    try:
        result = llm.generate(PROMPT, facts, LLMReport)
        summary, action = result.value.summary.strip(), result.value.recommended_action.strip()
        if not summary or not action:
            raise ValueError("empty field")
        if len(summary) > MAX_FIELD_CHARS or len(action) > MAX_FIELD_CHARS:
            raise ValueError("overly long field")
        return Report(severity, summary, action, source="llm", provider=result.provider, model=result.model)
    except Exception as exc:  # any model problem: the template
        summary, action = template_report(facts)
        return Report(severity, summary, action, source="template", error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    import cv2  # heavy imports: only the command line needs them

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

    print(f"Facts sent to the model:\n{json.dumps(build_payload(verdicts), indent=2)}\n")
    print(f"Severity: {report.severity}")
    print(f"Summary:  {report.summary}")
    print(f"Action:   {report.recommended_action}")
    print(f"Source:   {report.source} ({report.model or report.error})")  # developer tool: provenance shown


if __name__ == "__main__":
    main()
