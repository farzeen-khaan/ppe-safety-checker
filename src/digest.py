"""F8 — Daily digest: summarise the last 24 hours of the incident log on Discord.

Usage:
    python -m src.digest                                   # build and send (the last 24 hours)
    python -m src.digest --hours 6                         # a shorter or longer window
    python -m src.digest --dry-run                         # print only: send nothing, record nothing
    python -m src.digest --until 2026-11-10T13:00:00Z      # the window ending then (re-runs, tests)

Headless: configured from environment variables only (a .env file is read if present).
Posts to DISCORD_DIGEST_WEBHOOK_URL, or DISCORD_WEBHOOK_URL if that is not set.

Exit code 0 when sent (or printed), 1 when the digest could not be built or sent, so a
scheduled run (GitHub Actions, Phase 8) fails visibly. The whole message is built and
checked against Discord's limit before anything is sent, then goes out in one request:
a failure never sends half a digest. Every run except a dry run is recorded in the
log's digest_runs table.

Gemini writes a 3-sentence overview from aggregated counts only (no photos, names or
per-person data); if it is unavailable, a template overview is used and labelled.
"""

import argparse
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel

from src import report
from src.discord_webhook import MAX_CONTENT_CHARS, MAX_ERROR_CHARS, escape_markdown, redact, send_to_discord
from src.report import workers
from src.storage import PKT, get_since, now_utc, record_digest_run

log = logging.getLogger(__name__)

DEFAULT_HOURS = 24
WORST_COUNT = 3
MAX_NAME_CHARS = 40  # photo names in the worst list
MAX_OVERVIEW_CHARS = 700  # longer Gemini answers are rejected (template instead)
SEVERITY_RANK = {"High": 2, "Medium": 1, "None": 0}
NO_WEBHOOK = "no webhook: set DISCORD_DIGEST_WEBHOOK_URL (or DISCORD_WEBHOOK_URL)"


def digest_webhook_url() -> str | None:
    """DISCORD_DIGEST_WEBHOOK_URL, else DISCORD_WEBHOOK_URL, else None. ALERTS_ENABLED does not apply."""
    for name in ("DISCORD_DIGEST_WEBHOOK_URL", "DISCORD_WEBHOOK_URL"):
        url = os.getenv(name, "").strip()
        if url:
            return url
    return None


# --- Facts from the log ---

def summarise(rows: list[dict]) -> dict:
    """Totals for the window, summed from the counts stored on each analysis (nothing is re-run)."""
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
        "reports_gemini": number("report_source", "gemini"),
        "reports_template": number("report_source", "template"),
        "alerts_sent": number("alert_status", "sent"),
        "alerts_failed": number("alert_status", "failed"),
        "alerts_disabled": number("alert_status", "disabled"),
        "alerts_pending": number("alert_status", "pending"),
    }


# The only facts Gemini sees: safety totals, nothing per photo or per person.
OVERVIEW_FACTS = (
    "analyses", "people_detected", "checked_for_ppe", "compliant", "in_violation", "too_far_to_judge",
    "missing_helmet_only", "missing_vest_only", "missing_helmet_and_vest",
    "severity_high", "severity_medium", "severity_none",
)


def overview_facts(stats: dict) -> dict:
    """Aggregates for the overview, plus totals precomputed so the LLM never does arithmetic."""
    facts = {key: stats[key] for key in OVERVIEW_FACTS}
    # Per item: a worker missing both counts for each, so "the most common problem" is fair.
    facts["missing_helmet_total"] = stats["missing_helmet_only"] + stats["missing_helmet_and_vest"]
    facts["missing_vest_total"] = stats["missing_vest_only"] + stats["missing_helmet_and_vest"]
    checked = stats["checked_for_ppe"]
    facts["violation_rate_of_checked_pct"] = round(100 * stats["in_violation"] / checked) if checked else None
    return facts


def worst_analyses(rows: list[dict], n: int = WORST_COUNT) -> list[dict]:
    """Analyses with violations, worst first: most workers in violation, then severity, then newest."""
    flagged = [r for r in rows if r["violation_count"]]
    return sorted(
        flagged,
        key=lambda r: (r["violation_count"], SEVERITY_RANK[r["severity"]], r["created_at"], r["id"]),
        reverse=True,
    )[:n]


# --- Overview (Gemini, with a template fallback) ---

class Overview(BaseModel):
    """The exact shape we ask Gemini to return."""

    overview: str


OVERVIEW_PROMPT = """You are a construction-site safety assistant writing the end-of-day summary
for a safety officer. Use ONLY the facts below; do not invent locations, names, trends
or causes. Workers in too_far_to_judge were not checked; never call them compliant.

Write exactly 3 plain sentences in "overview":
1. How many analyses ran, how many workers were checked (checked_for_ppe) and how many
   were in violation (use violation_rate_of_checked_pct if it is not null).
2. The most common problem (compare missing_helmet_total and missing_vest_total) and,
   if too_far_to_judge > 0, how many workers could not be checked.
3. One concrete focus for tomorrow that follows from these facts.

Facts:
{facts}
"""


def most_common_problem(facts: dict) -> str | None:
    helmets, vests = facts["missing_helmet_total"], facts["missing_vest_total"]
    if not helmets and not vests:
        return None
    if helmets == vests:
        return f"missing helmets and missing vests equally ({workers(helmets)} each)"
    item, n = ("helmet", helmets) if helmets > vests else ("vest", vests)
    return f"a missing {item} ({workers(n)})"


def template_overview(facts: dict) -> str:
    """Three sentences built only from the counts. Never calls unchecked workers compliant."""
    n, checked, too_far = facts["analyses"], facts["checked_for_ppe"], facts["too_far_to_judge"]
    analyses = "1 analysis" if n == 1 else f"{n} analyses"
    if checked == 0:
        first = f"{analyses} detected {workers(facts['people_detected'])}, but none were close enough to check for PPE."
    else:
        first = (
            f"{analyses} checked {workers(checked)} for PPE; {facts['in_violation']} "
            f"({facts['violation_rate_of_checked_pct']}%) were in violation."
        )

    problem = most_common_problem(facts)
    second = f"The most common problem was {problem}." if problem else "No checked worker was missing a helmet or vest."
    if too_far:
        second += f" {workers(too_far).capitalize()} could not be checked because they were too far from the camera."

    focus = []
    if problem:
        helmets, vests = facts["missing_helmet_total"], facts["missing_vest_total"]
        focus.append("helmet checks" if helmets > vests else "vest checks" if vests > helmets else "helmet and vest checks")
    if too_far:
        focus.append("closer photos so every worker can be checked")
    third = f"Tomorrow, prioritise {' and '.join(focus)}." if focus else "Keep up the current PPE checks tomorrow."
    return f"{first} {second} {third}"


def write_overview(stats: dict, use_llm: bool = True) -> tuple[str, str | None]:
    """(overview text, Gemini model that wrote it, or None for the template). Never raises."""
    facts = overview_facts(stats)
    if use_llm:
        try:
            # Same model chain and 10 s per-model timeout as the incident reports.
            answer, model = report.ask_llm(facts, prompt=OVERVIEW_PROMPT, schema=Overview)
            text = answer.overview.strip()
            if text and len(text) <= MAX_OVERVIEW_CHARS:
                return text, model
            log.warning("Overview from %s was empty or too long; using the template.", model)
        except Exception as exc:  # any LLM problem: the template below
            log.warning("Gemini overview failed (%s); using the template.", str(exc)[:300])
    return template_overview(facts), None


# --- Message ---

def hours_text(hours: int) -> str:
    return "1 hour" if hours == 1 else f"{hours} hours"


def pkt(moment: datetime, fmt: str = "%d %b %H:%M") -> str:
    return moment.astimezone(PKT).strftime(fmt)


def header(since: datetime, until: datetime, hours: int) -> list[str]:
    title = "Daily PPE digest" if hours == DEFAULT_HOURS else f"PPE digest, last {hours_text(hours)}"
    return [f"📋 **{title} — {pkt(until, '%d %b %Y')}**", f"Window: {pkt(since)} – {pkt(until)} PKT"]


def empty_message(since: datetime, until: datetime, hours: int) -> str:
    return "\n".join(
        header(since, until, hours)
        + ["", f"No analyses in the last {hours_text(hours)}. The digest ran normally; there was nothing to report."]
    )


def worst_line(rank: int, r: dict) -> str:
    when = datetime.strptime(r["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    name = r["image_name"] if len(r["image_name"]) <= MAX_NAME_CHARS else r["image_name"][: MAX_NAME_CHARS - 1] + "…"
    return (
        f"{rank}. {pkt(when)} · {escape_markdown(name)} · {r['violation_count']} in violation · "
        f"{r['severity']} (#{r['id']})"
    )


def joined(parts: list[tuple[int, str]], always: int) -> str:
    """'5 sent · 1 failed · 2 disabled': the first `always` parts always, the rest only when non-zero."""
    return " · ".join(f"{n} {label}" for i, (n, label) in enumerate(parts) if i < always or n)


def build_message(
    stats: dict, worst: list[dict], overview: str, overview_model: str | None,
    since: datetime, until: datetime, hours: int,
) -> str:
    s = stats
    analyses = "1 analysis" if s["analyses"] == 1 else f"{s['analyses']} analyses"
    lines = header(since, until, hours) + [
        "",
        f"**{analyses}** · {s['people_detected']} people detected · {s['checked_for_ppe']} checked · "
        f"🚨 {s['in_violation']} in violation · ❔ {s['too_far_to_judge']} too far to judge",
        f"Violations: {s['missing_helmet_only']} missing helmet · {s['missing_vest_only']} missing vest · "
        f"{s['missing_helmet_and_vest']} missing both",
        f"Severity: {s['severity_high']} High · {s['severity_medium']} Medium · {s['severity_none']} None",
    ]
    if worst:
        lines += ["", "**Worst analyses**"] + [worst_line(i, r) for i, r in enumerate(worst, 1)]
    by = f"Gemini, {overview_model}" if overview_model else "template: Gemini was unavailable"
    alerts = joined(
        [(s["alerts_sent"], "sent"), (s["alerts_failed"], "failed"),
         (s["alerts_disabled"], "disabled"), (s["alerts_pending"], "pending")],
        always=2,
    )
    reports = joined([(s["reports_gemini"], "by Gemini"), (s["reports_template"], "template")], always=2)
    lines += ["", f"**Overview** ({by})", overview, "", f"Reliability: alerts {alerts} | reports {reports}"]
    return "\n".join(lines)


@dataclass
class Digest:
    message: str
    analyses: int
    overview_model: str | None  # None: template overview (or an empty window)


def build_digest(since: datetime, until: datetime, hours: int = DEFAULT_HOURS, use_llm: bool = True) -> Digest:
    """The complete digest for since <= created_at < until. Raises if the log can't be read."""
    rows = get_since(since, until)  # half-open window: consecutive digests never overlap
    if not rows:
        return Digest(empty_message(since, until, hours), 0, None)
    stats = summarise(rows)
    overview, model = write_overview(stats, use_llm=use_llm)
    message = build_message(stats, worst_analyses(rows), overview, model, since, until, hours)
    return Digest(message, len(rows), model)


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
    """Record the run. A recording problem is reported, never allowed to hide the run's own outcome."""
    try:
        record_digest_run(window_start=since, window_end=until, status=status, analyses_count=analyses, error=error)
    except Exception as exc:
        print(f"Warning: could not record the digest run: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send the PPE digest for the last N hours to Discord.")
    parser.add_argument("--hours", type=positive_int, default=DEFAULT_HOURS, help="window length (default 24)")
    parser.add_argument("--until", type=parse_until, help="end of the window, with a time zone (default: now)")
    parser.add_argument("--dry-run", action="store_true", help="print the digest; send and record nothing")
    parser.add_argument("--no-llm", action="store_true", help="use the template overview only")
    args = parser.parse_args(argv)

    until = args.until or now_utc()  # UTC, like the stored created_at
    since = until - timedelta(hours=args.hours)
    url = None if args.dry_run else digest_webhook_url()

    if not args.dry_run and url is None:  # checked before any work, so a misconfigured run fails in seconds
        print(f"Digest not sent: {NO_WEBHOOK}.", file=sys.stderr)
        save_run(since, until, "failed", None, NO_WEBHOOK)
        return 1

    analyses = None
    try:
        digest = build_digest(since, until, args.hours, use_llm=not args.no_llm)
        analyses = digest.analyses
        if len(digest.message) > MAX_CONTENT_CHARS:  # never cut: a truncated digest is a partial digest
            raise ValueError(f"digest is {len(digest.message)} characters, over Discord's {MAX_CONTENT_CHARS}")
        if args.dry_run:
            print(digest.message)
            return 0
        error = send_to_discord(url, digest.message)  # synchronous; already redacted
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
    overview = (
        "empty window" if not analyses
        else f"overview by {digest.overview_model}" if digest.overview_model else "template overview"
    )
    print(f"Digest sent: {analyses} analyses, {len(digest.message)} characters, {overview}.")
    return 0


if __name__ == "__main__":
    # The message has emoji; a Windows console defaults to a code page that can't print them.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
