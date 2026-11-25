"""F7 — Violation alerts: post the annotated photo and an embed to a Discord webhook.

Usage:
    python -m src.alerts --test     # post one test message, to check the webhook works

The pipeline calls alert_after_save() once an analysis is saved. The send runs on a
background thread, so the web UI never waits for Discord. The outcome (sent, failed,
disabled) is written to the analysis's alert status in the incident log.
"""

import argparse
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from dotenv import load_dotenv

# Shared with the digest; re-exported for existing callers.
from src.discord_webhook import (  # noqa: F401
    COLOURS, IMAGE_FILENAME, MAX_CONTENT_CHARS, MAX_ERROR_CHARS, MAX_RETRY_AFTER_S, RETRY_DELAY_S,
    TIMEOUT_S, describe, embed_problems, escape_markdown, footer_text, post_to_discord, redact,
    retry_after, send_to_discord, shorten_name, summary_note,
)
from src.report import Report, build_payload, violation_parts, workers
from src.storage import PKT, now_utc, update_alert_status

load_dotenv()
log = logging.getLogger(__name__)

MAX_IMAGE_SIDE = 1600  # smaller and faster to upload; boxes stay readable
MAX_SUMMARY_CHARS = 1500  # well inside the 4096 embed description limit
MAX_ACTION_CHARS = 1000  # field values are limited to 1024

# One worker: alerts go out one at a time, which suits Discord's per-webhook rate limit.
# Non-daemon, so a command-line run waits for its alerts before exiting.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="alert")


def webhook_config() -> tuple[str | None, str | None]:
    """(webhook URL, None) when alerts are on, or (None, reason) when they are off."""
    if os.getenv("ALERTS_ENABLED", "").strip().lower() != "true":
        return None, "ALERTS_ENABLED is not true"
    url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not url:
        return None, "DISCORD_WEBHOOK_URL is not set"
    return url, None


# --- Message ---

@dataclass
class AlertMessage:
    content: str  # one plain line: what notifications and previews show
    embed: dict


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def build_alert(
    verdicts, report: Report, analysis_id: int | None, image_name: str, created_at: datetime
) -> AlertMessage:
    """Severity colour, counts as fields, the summary, the annotated image, and the time."""
    p = build_payload(verdicts)
    title = f"PPE violation — {report.severity} severity"
    fields = [
        {"name": "In violation", "value": f"{p['in_violation']} of {workers(p['checked_for_ppe'])} checked",
         "inline": True},
        {"name": "Missing", "value": violation_parts(p), "inline": True},
    ]
    if p["too_far_to_judge"]:
        fields.append({"name": "Not checked", "value": f"{p['too_far_to_judge']} too far from the camera",
                       "inline": True})
    fields += [
        {"name": "Recommended action", "value": _cut(report.recommended_action, MAX_ACTION_CHARS), "inline": False},
        {"name": "Photo", "value": escape_markdown(shorten_name(image_name)), "inline": True},
        {"name": "Analysis", "value": f"#{analysis_id}" if analysis_id is not None else "Not saved to the log",
         "inline": True},
    ]
    embed = {
        "title": title,
        "color": COLOURS[report.severity],
        "description": _cut(report.summary, MAX_SUMMARY_CHARS),
        "fields": fields,
        "image": {"url": f"attachment://{IMAGE_FILENAME}"},
        "footer": {"text": footer_text(report.automated, created_at, PKT)},
        "timestamp": created_at.isoformat(),
    }
    content = f"🚨 {title}: {p['in_violation']} of {workers(p['checked_for_ppe'])} checked missing PPE"
    return AlertMessage(content, embed)


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


def deliver(url: str, analysis_id: int | None, message: AlertMessage, annotated: np.ndarray) -> str:
    """Background job: send and record the result. Never raises. Returns 'sent' or 'failed'."""
    try:
        problems = embed_problems(message.content, [message.embed])
        if problems:
            raise ValueError("; ".join(problems))
        error = send_to_discord(url, message.content, encode_image(annotated), [message.embed])
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
    message = build_alert(verdicts, report, analysis_id, image_name, created_at or now_utc())
    return "sending", _executor.submit(deliver, url, analysis_id, message, annotated)


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
