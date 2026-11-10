"""F7 — Violation alerts: post the annotated photo and a short message to a Discord webhook.

Usage:
    python -m src.alerts --test     # post one test message, to check the webhook works

The pipeline calls alert_after_save() once an analysis is saved. Sending happens on a
background thread, so the web UI never waits for Discord. The outcome (sent, failed,
disabled) is written to the analysis's alert status in the incident log.

The webhook URL is a secret: anyone who has it can post to the channel. It is never
printed, logged or stored; error messages pass through redact() first.
"""

import argparse
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

# The Discord HTTP code is shared with the digest; re-exported here for existing callers.
from src.discord_webhook import (  # noqa: F401
    MAX_CONTENT_CHARS, MAX_ERROR_CHARS, MAX_RETRY_AFTER_S, RETRY_DELAY_S, TIMEOUT_S,
    describe, escape_markdown, post_to_discord, redact, retry_after, send_to_discord,
)
from src.report import Report, build_payload, workers
from src.storage import PKT, now_utc, update_alert_status

load_dotenv()
log = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 1600  # downscale before upload: smaller and faster, boxes stay readable

# One worker: alerts go out one at a time, which suits Discord's per-webhook rate limit.
# Non-daemon threads, so a command-line run waits for its alerts before exiting.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alert")


# --- Configuration ---

def webhook_config() -> tuple[str | None, str | None]:
    """(webhook URL, None) when alerts are on, or (None, reason) when they are off."""
    if os.getenv("ALERTS_ENABLED", "").strip().lower() != "true":
        return None, "ALERTS_ENABLED is not true"
    url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        return None, "DISCORD_WEBHOOK_URL is not set"
    return url, None


# --- Message ---

def build_message(
    verdicts, report: Report, analysis_id: int | None, image_name: str, created_at: datetime
) -> str:
    """The alert text: severity, who is in violation, who could not be checked, summary, time."""
    p = build_payload(verdicts)
    breakdown = [
        f"{n} missing {item}"
        for n, item in (
            (p["missing_helmet_only"], "helmet"),
            (p["missing_vest_only"], "vest"),
            (p["missing_helmet_and_vest"], "helmet and vest"),
        )
        if n
    ]
    lines = [
        f"🚨 **PPE violation — {report.severity} severity**",
        f"**{p['in_violation']} of {workers(p['checked_for_ppe'])} checked in violation:** {', '.join(breakdown)}.",
    ]
    if p["too_far_to_judge"]:
        people = "1 person" if p["too_far_to_judge"] == 1 else f"{p['too_far_to_judge']} people"
        lines.append(f"**{people} could not be checked** (too far to judge).")
    reference = f"Analysis #{analysis_id}" if analysis_id is not None else "Analysis not saved to the log"
    when = created_at.astimezone(PKT).strftime("%d %b %Y, %H:%M PKT")
    footer = f"{reference} · {escape_markdown(image_name[:100])} · {when}"

    # The summary is the only long, variable part, so it gets whatever room is left.
    label = "**Report:** "
    room = MAX_CONTENT_CHARS - len("\n".join(lines + [label, footer]))
    summary = report.summary if len(report.summary) <= room else report.summary[: room - 1].rstrip() + "…"
    return "\n".join(lines + [label + summary, footer])


def encode_image(image_bgr: np.ndarray) -> bytes:
    """JPEG bytes of the annotated image, downscaled if larger than MAX_IMAGE_SIDE."""
    height, width = image_bgr.shape[:2]
    scale = MAX_IMAGE_SIDE / max(height, width)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode the annotated image as JPEG")
    return buffer.tobytes()


# --- Sending ---

def record(analysis_id: int | None, status: str, error: str | None = None) -> None:
    """Write the alert outcome to the log. A log problem must not crash the alert thread."""
    if analysis_id is None:
        return
    try:
        update_alert_status(analysis_id, status, error)
    except Exception as exc:
        log.warning("Could not record alert status for analysis #%s: %s", analysis_id, type(exc).__name__)


def deliver(url: str, analysis_id: int | None, content: str, annotated: np.ndarray) -> str:
    """Background job: send and record the result. Never raises. Returns 'sent' or 'failed'."""
    try:
        error = send_to_discord(url, content, encode_image(annotated))
    except Exception as exc:  # a bug here must still end as a recorded failure
        error = redact(f"{type(exc).__name__}: {exc}", url)[:MAX_ERROR_CHARS]
    status = "sent" if error is None else "failed"
    record(analysis_id, status, error)
    if error:
        log.warning("Alert for analysis #%s failed: %s", analysis_id, error)
    return status


def alert_after_save(
    analysis_id: int | None,
    verdicts,
    report: Report,
    annotated: np.ndarray,
    image_name: str,
    created_at: datetime | None = None,
) -> tuple[str, Future | None]:
    """Start the alert for one analysis, without waiting for it.

    Returns (status, future):
    - ("not_needed", None): nobody is in violation. Too-far-to-judge workers alone never alert.
    - ("disabled", None):   alerts are switched off or not configured; recorded in the log.
    - ("sending", future):  sending on the background thread; future.result() is 'sent' or 'failed'.
    """
    if not any(v.is_violation for v in verdicts):
        return "not_needed", None
    url, reason = webhook_config()
    if url is None:
        record(analysis_id, "disabled", reason)
        return "disabled", None
    content = build_message(verdicts, report, analysis_id, image_name, created_at or now_utc())
    return "sending", _executor.submit(deliver, url, analysis_id, content, annotated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Discord webhook used for violation alerts.")
    parser.add_argument("--test", action="store_true", help="post one test message")
    args = parser.parse_args()
    if not args.test:
        parser.print_help()
        return
    url, reason = webhook_config()
    if url is None:
        print(f"Alerts are off: {reason}.")
        return
    image = np.full((120, 320, 3), 255, dtype=np.uint8)
    cv2.putText(image, "PPE test", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 3)
    error = send_to_discord(url, "✅ PPE Safety Checker: webhook test. Alerts are working.", encode_image(image))
    print("Sent." if error is None else f"Failed: {error}")


if __name__ == "__main__":
    main()
