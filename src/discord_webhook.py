"""Post a message (and optionally one image) to a Discord webhook, with one retry.

Shared by the violation alerts (F7, src/alerts.py) and the daily digest (F8, src/digest.py).
Only `requests` is needed here, so the digest can send without OpenCV or numpy.

The webhook URL is a secret: anyone who has it can post to the channel. It is never
printed, logged or stored; every error message passes through redact() first.
"""

import json
import re
from time import sleep

import requests

MAX_CONTENT_CHARS = 2000  # Discord's limit for the text of one message
TIMEOUT_S = (5, 15)  # (connect, read) per attempt
RETRY_DELAY_S = 2  # wait before the one retry after a network error or 5xx
MAX_RETRY_AFTER_S = 30  # longer 429 waits would break the "alert within 1 minute" target
MAX_ERROR_CHARS = 300


def redact(text: str, url: str | None = None) -> str:
    """Remove the webhook URL and its token from text before it is stored or shown."""
    if url:
        text = text.replace(url, "<webhook>")
    # Also catches the path alone, as in urllib3 errors ("... with url: /api/webhooks/123/abc?wait=true").
    return re.sub(r"webhooks/\d+/[\w-]+", "webhooks/<redacted>", text)


def escape_markdown(text: str) -> str:
    """Photo names like 'site_1_*.jpg' must not turn into italics or bold in Discord."""
    return re.sub(r"([\\*_~`|>])", r"\\\1", text)


def post_to_discord(url: str, content: str, image_jpg: bytes | None = None) -> requests.Response:
    """One webhook call: the text, plus the image as an attached file if there is one."""
    payload = {
        "content": content,
        # Never ping anyone, even if a photo name or report contains "@everyone".
        "allowed_mentions": {"parse": []},
    }
    params = {"wait": "true"}  # Discord answers 200 only once the message really exists
    if image_jpg is None:
        return requests.post(url, params=params, json=payload, timeout=TIMEOUT_S)
    return requests.post(
        url,
        params=params,
        data={"payload_json": json.dumps(payload)},
        files={"files[0]": ("ppe_alert.jpg", image_jpg, "image/jpeg")},
        timeout=TIMEOUT_S,
    )


def retry_after(response: requests.Response) -> float:
    """Seconds Discord asks us to wait after a 429 (JSON body first, then the header)."""
    try:
        return max(float(response.json()["retry_after"]), 0.0)
    except (ValueError, KeyError, TypeError):
        try:
            return max(float(response.headers.get("Retry-After", 1)), 0.0)
        except ValueError:
            return 1.0


def describe(response: requests.Response) -> str:
    """'HTTP 404: Unknown Webhook', using Discord's own error message when there is one."""
    try:
        detail = response.json().get("message") or ""
    except ValueError:
        detail = response.text[:100]
    return f"HTTP {response.status_code}: {detail}".rstrip(": ")


def send_to_discord(url: str, content: str, image_jpg: bytes | None = None) -> str | None:
    """Post once and retry once. Returns None on success, or the error (webhook removed)."""
    error = None
    for attempt in (1, 2):
        try:
            response = post_to_discord(url, content, image_jpg)
        except requests.RequestException as exc:  # no connection, DNS failure, timeout
            error, wait = f"{type(exc).__name__}: {exc}", RETRY_DELAY_S
        else:
            if response.ok:
                return None
            error = describe(response)
            if response.status_code == 429:
                wait = retry_after(response)
                if wait > MAX_RETRY_AFTER_S:
                    error += f" (asked to wait {wait:.0f} s; not retried)"
                    break
            elif response.status_code >= 500:  # Discord's side: worth one more try
                wait = RETRY_DELAY_S
            else:  # 400, 401, 404: the request or the webhook is wrong; retrying won't help
                break
        if attempt == 1:
            sleep(wait)
    return redact(error or "unknown error", url)[:MAX_ERROR_CHARS]
