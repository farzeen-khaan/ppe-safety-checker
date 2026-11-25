"""F8 — Daily digest: summarise the last 24 hours of the incident log on Discord.

Usage:
    python -m src.digest                                   # build and send (the last 24 hours)
    python -m src.digest --hours 6                         # a shorter or longer window
    python -m src.digest --dry-run                         # print only: send nothing, record nothing
    python -m src.digest --until 2026-11-10T13:00:00Z      # the window ending then (re-runs, tests)

Configured from environment variables only (a .env file is read if present). Posts to
DISCORD_DIGEST_WEBHOOK_URL, or DISCORD_WEBHOOK_URL if that is not set.

Exit code 0 when sent (or printed), 1 when the digest could not be built or sent, so a
scheduled run fails visibly. The message is checked against Discord's limits before
anything is sent and goes out in one request: never half a digest. Every run except a
dry run is recorded in digest_runs.
"""

import argparse
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from src import llm
from src.discord_webhook import (
    COLOURS, GREY, MAX_ERROR_CHARS, embed_problems, escape_markdown, footer_text, footer_time, redact,
    send_to_discord, shorten_name,
)
from src.report import VERIFY, distance_note, pattern, workers
from src.storage import PKT, get_since, now_utc, record_digest_run

log = logging.getLogger(__name__)

DEFAULT_HOURS = 24
WORST_COUNT = 3
MAX_NAME_CHARS = 40
MAX_SUMMARY_CHARS = 700  # longer model answers are rejected (template instead)
SEVERITY_RANK = {"High": 2, "Medium": 1, "None": 0}
NO_WEBHOOK = "no webhook: set DISCORD_DIGEST_WEBHOOK_URL (or DISCORD_WEBHOOK_URL)"


def digest_webhook_url() -> str | None:
    """DISCORD_DIGEST_WEBHOOK_URL, else DISCORD_WEBHOOK_URL. ALERTS_ENABLED does not apply."""
    for name in ("DISCORD_DIGEST_WEBHOOK_URL", "DISCORD_WEBHOOK_URL"):
        url = os.getenv(name, "").strip()
        if url:
            return url
    return None


# --- Facts from the log ---

def summarise(rows: list[dict]) -> dict:
    """Totals for the window, summed from the counts stored on each analysis."""
    def total(column: str) -> int:
        return sum(r[column] for r in rows)

    def number(column: str, value: str) -> int:
        return sum(r[column] == value for r in rows)

    return {
        "analyses": len(rows),
        "people_detected": total("people_count"),
        "checked_for_ppe": total("compliant_count") + total("violation_count"),
        "compliant": total("compliant_count"),
        "in_violation": total("violation_count"),
        "too_far_to_judge": total("too_far_count"),
        "missing_helmet_only": total("missing_helmet_only_count"),
        "missing_vest_only": total("missing_vest_only_count"),
        "missing_helmet_and_vest": total("missing_both_count"),
        "severity_high": number("severity", "High"),
        "severity_medium": number("severity", "Medium"),
        "severity_none": number("severity", "None"),
        "summaries_automated": number("report_source", "llm"),
        "summaries_standard": number("report_source", "template"),
        "alerts_sent": number("alert_status", "sent"),
        "alerts_failed": number("alert_status", "failed"),
        "alerts_disabled": number("alert_status", "disabled"),
        "alerts_pending": number("alert_status", "pending"),
    }


# The only facts the model sees: safety totals, nothing per photo or per person.
SUMMARY_FACTS = (
    "analyses", "people_detected", "checked_for_ppe", "compliant", "in_violation", "too_far_to_judge",
    "missing_helmet_only", "missing_vest_only", "missing_helmet_and_vest",
    "severity_high", "severity_medium", "severity_none",
)


def summary_facts(stats: dict) -> dict:
    """Aggregates for the summary, plus totals precomputed so the model never does arithmetic."""
    facts = {key: stats[key] for key in SUMMARY_FACTS}
    # A worker missing both counts for each item, so "the most common problem" is fair.
    facts["missing_helmet_total"] = stats["missing_helmet_only"] + stats["missing_helmet_and_vest"]
    facts["missing_vest_total"] = stats["missing_vest_only"] + stats["missing_helmet_and_vest"]
    checked = stats["checked_for_ppe"]
    facts["violation_rate_of_checked_pct"] = round(100 * stats["in_violation"] / checked) if checked else None
    return facts


def worst_analyses(rows: list[dict], n: int = WORST_COUNT) -> list[dict]:
    """Analyses with violations: most workers in violation, then severity, then newest."""
    flagged = [r for r in rows if r["violation_count"]]
    return sorted(
        flagged,
        key=lambda r: (r["violation_count"], SEVERITY_RANK[r["severity"]], r["created_at"], r["id"]),
        reverse=True,
    )[:n]


# --- Summary (model, with a template fallback) ---

class DigestSummary(BaseModel):
    summary: str


SUMMARY_PROMPT = """You write the end-of-day PPE summary for a construction-site supervisor. The
digest already shows every count (analyses, people, checked, in violation, not checked,
severity, violation types), so do not repeat them.

Rules:
- Write 2 or 3 short sentences in "summary": what stands out, then what to do tomorrow.
- Do not restate the counts. Say what stands out, in words: which item is missing most,
  whether violations are rare or common among the workers checked, whether a large share of
  people could not be checked.
- Use only the facts. Do not invent places, names, trends or causes.
- No filler or lead-ins such as "it is important to", "please note", "overall", "a total of"
  or "specifically".
- Workers in too_far_to_judge were not checked. Never call them compliant.
- Distance is about the camera, not the workers: the fix is camera placement or photo quality.
- The check is automated and can be wrong, so never say "stop work". If in_violation > 0,
  the last sentence starts with "Verify on site, then correct:" and names the item to check
  (helmets, vests or both); add the camera fix if many workers were too far.

Example (for few violations, all missing only a vest, nobody too far): "Violations were rare
among the workers checked, and every one was a missing vest. Verify on site, then correct:
check vests at the start of tomorrow's shift."

Facts:
{facts}
"""


def template_summary(facts: dict) -> str:
    """2 or 3 sentences from the counts, without repeating them. Never calls unchecked workers compliant."""
    checked, too_far, v = facts["checked_for_ppe"], facts["too_far_to_judge"], facts["in_violation"]
    if checked == 0:
        sentences = ["No worker could be checked: everyone was too far from the camera."]
    elif v:
        sentences = [pattern(facts, flagged="workers in violation")]
    else:
        sentences = ["No checked worker was missing PPE."]
    if too_far and checked:
        sentences.append(distance_note(facts))

    camera = "move the camera closer so more workers can be checked"
    if v:
        helmets, vests = facts["missing_helmet_total"], facts["missing_vest_total"]
        item = "helmets and vests" if helmets and vests else "helmets" if helmets else "vests"  # every missing item
        action = f"{VERIFY}: check {item} at the start of tomorrow's shift"
        sentences.append(action + (f", and {camera}." if too_far else "."))
    elif too_far:
        sentences.append(camera[0].upper() + camera[1:] + " tomorrow.")
    else:
        sentences.append("No PPE action needed tomorrow.")
    return " ".join(sentences)


def write_summary(stats: dict, use_llm: bool = True) -> tuple[str, bool]:
    """(summary text, automated?). Never raises."""
    facts = summary_facts(stats)
    if use_llm:
        try:
            text = llm.generate(SUMMARY_PROMPT, facts, DigestSummary).value.summary.strip()
            if text and len(text) <= MAX_SUMMARY_CHARS:
                return text, True
            log.warning("Automated summary was empty or too long; using the standard one.")
        except Exception as exc:
            log.warning("Automated summary failed (%s); using the standard one.", str(exc)[:300])
    return template_summary(facts), False


# --- Message ---

def hours_text(hours: int) -> str:
    return "1 hour" if hours == 1 else f"{hours} hours"


def pkt(moment: datetime, fmt: str = "%d %b %H:%M") -> str:
    return moment.astimezone(PKT).strftime(fmt)


def title(until: datetime, hours: int) -> str:
    name = "Daily PPE digest" if hours == DEFAULT_HOURS else f"PPE digest, last {hours_text(hours)}"
    return f"{name} — {pkt(until, '%d %b %Y')}"


def worst_line(rank: int, r: dict) -> str:
    when = datetime.strptime(r["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    name = escape_markdown(shorten_name(r["image_name"], MAX_NAME_CHARS))
    return f"{rank}. {pkt(when)} · {name} · {r['violation_count']} in violation · {r['severity']} (#{r['id']})"


def joined(parts: list[tuple[int, str]], always: int, sep: str = " · ") -> str:
    """The first `always` parts always, the rest only when non-zero."""
    return sep.join(f"{n} {label}" for i, (n, label) in enumerate(parts) if i < always or n)


def colour(stats: dict) -> int:
    """The worst severity in the window."""
    for severity in ("High", "Medium"):
        if stats[f"severity_{severity.lower()}"]:
            return COLOURS[severity]
    return COLOURS["None"]


@dataclass
class Digest:
    content: str  # one plain line: what notifications show
    embed: dict
    analyses: int
    automated: bool  # the summary was written by the model


def empty_digest(since: datetime, until: datetime, hours: int) -> Digest:
    embed = {
        "title": title(until, hours),
        "color": GREY,
        "description": (
            f"Window: {pkt(since)} – {pkt(until)} PKT\n\n"
            f"No analyses in the last {hours_text(hours)}. The digest ran normally; there was nothing to report."
        ),
        "footer": {"text": footer_time(until, PKT)},
        "timestamp": until.isoformat(),
    }
    return Digest(f"📋 {title(until, hours)}: no analyses in the last {hours_text(hours)}", embed, 0, False)


def build_embed(
    stats: dict, worst: list[dict], summary: str, automated: bool, since: datetime, until: datetime, hours: int
) -> dict:
    s = stats
    fields = [
        {"name": "Analyses", "value": str(s["analyses"]), "inline": True},
        {"name": "People detected", "value": str(s["people_detected"]), "inline": True},
        {"name": "Checked", "value": str(s["checked_for_ppe"]), "inline": True},
        {"name": "In violation", "value": str(s["in_violation"]), "inline": True},
        {"name": "Not checked (too far)", "value": str(s["too_far_to_judge"]), "inline": True},
        {"name": "Severity", "value": f"{s['severity_high']} High · {s['severity_medium']} Medium · "
                                      f"{s['severity_none']} None", "inline": True},
        {"name": "Violations", "value": f"{s['missing_helmet_only']} missing helmet · {s['missing_vest_only']} "
                                        f"missing vest · {s['missing_helmet_and_vest']} missing both", "inline": False},
    ]
    if worst:
        fields.append({"name": "Worst analyses", "value": "\n".join(worst_line(i, r) for i, r in enumerate(worst, 1)),
                       "inline": False})
    alerts = joined([(s["alerts_sent"], "sent"), (s["alerts_failed"], "failed"),
                     (s["alerts_disabled"], "not sent (alerts off)"), (s["alerts_pending"], "pending")], always=2)
    fields += [
        {"name": "Summary", "value": summary, "inline": False},
        {"name": "Reliability", "inline": False,
         "value": f"Alerts: {alerts}\nSummaries: {s['summaries_automated']} automated, {s['summaries_standard']} standard"},
    ]
    return {
        "title": title(until, hours),
        "color": colour(s),
        "description": f"Window: {pkt(since)} – {pkt(until)} PKT",
        "fields": fields,
        "footer": {"text": footer_text(automated, until, PKT)},
        "timestamp": until.isoformat(),
    }


def build_digest(since: datetime, until: datetime, hours: int = DEFAULT_HOURS, use_llm: bool = True) -> Digest:
    """The complete digest for since <= created_at < until. Raises if the log can't be read."""
    rows = get_since(since, until)  # half-open: consecutive digests never overlap
    if not rows:
        return empty_digest(since, until, hours)
    stats = summarise(rows)
    summary, automated = write_summary(stats, use_llm=use_llm)
    embed = build_embed(stats, worst_analyses(rows), summary, automated, since, until, hours)
    analyses = "1 analysis" if stats["analyses"] == 1 else f"{stats['analyses']} analyses"
    content = f"📋 {title(until, hours)}: {analyses}, {workers(stats['in_violation'])} in violation"
    return Digest(content, embed, len(rows), automated)


def as_text(digest: Digest) -> str:
    """The message as plain text, for --dry-run."""
    e = digest.embed
    lines = [digest.content, "", f"**{e['title']}**", e.get("description", "")]
    for f in e.get("fields", []):
        lines += ["", f"**{f['name']}**", f["value"]]
    lines += ["", e["footer"]["text"]]
    return "\n".join(lines)


# --- Command line ---

def parse_until(text: str) -> datetime:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError("include a time zone, e.g. 2026-11-10T13:00:00Z or +05:00")
    return moment.astimezone(timezone.utc)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def save_run(since: datetime, until: datetime, status: str, analyses: int | None, error: str | None) -> None:
    """Record the run. A recording problem is reported but never changes the run's outcome."""
    try:
        record_digest_run(window_start=since, window_end=until, status=status, analyses_count=analyses, error=error)
    except Exception as exc:
        print(f"Warning: could not record the digest run: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send the PPE digest for the last N hours to Discord.")
    parser.add_argument("--hours", type=positive_int, default=DEFAULT_HOURS, help="window length (default 24)")
    parser.add_argument("--until", type=parse_until, help="end of the window, with a time zone (default: now)")
    parser.add_argument("--dry-run", action="store_true", help="print the digest; send and record nothing")
    parser.add_argument("--no-llm", action="store_true", help="use the standard summary only")
    args = parser.parse_args(argv)

    until = args.until or now_utc()
    since = until - timedelta(hours=args.hours)
    url = None if args.dry_run else digest_webhook_url()

    if not args.dry_run and url is None:  # fail in seconds, before any work
        print(f"Digest not sent: {NO_WEBHOOK}.", file=sys.stderr)
        save_run(since, until, "failed", None, NO_WEBHOOK)
        return 1

    analyses = None
    try:
        digest = build_digest(since, until, args.hours, use_llm=not args.no_llm)
        analyses = digest.analyses
        problems = embed_problems(digest.content, [digest.embed])
        if problems:  # never cut: a truncated digest is a partial digest
            raise ValueError("digest breaks Discord's limits: " + "; ".join(problems))
        if args.dry_run:
            print(as_text(digest))
            return 0
        error = send_to_discord(url, digest.content, embeds=[digest.embed])  # synchronous; already redacted
    except Exception as exc:
        print(redact(traceback.format_exc(), url), file=sys.stderr)  # full trace for the Actions log
        error = redact(f"{type(exc).__name__}: {exc}", url)[:MAX_ERROR_CHARS]
        if args.dry_run:
            return 1

    if error:
        print(f"Digest not sent: {error}", file=sys.stderr)
        save_run(since, until, "failed", analyses, error)
        return 1
    save_run(since, until, "sent", analyses, None)
    kind = "empty window" if not analyses else "automated summary" if digest.automated else "standard summary"
    print(f"Digest sent: {analyses} analyses, {kind}.")
    return 0


if __name__ == "__main__":
    # The message has emoji; a Windows console's default code page can't print them.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
