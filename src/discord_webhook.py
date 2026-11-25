"""Post a message to a Discord webhook, with one retry. Shared by the alerts (F7) and the digest (F8).

Messages are a short plain-text line (shown in notifications) plus one embed. Only `requests`
is needed, so the digest can send without OpenCV or numpy.

The webhook URL is a secret: anyone who has it can post to the channel. It is never printed,
logged or stored; every error passes through redact() first.
"""

import json
import re
from datetime import datetime
from time import sleep

import requests

MAX_CONTENT_CHARS = 2000
TIMEOUT_S = (5, 15)  # (connect, read) per attempt
RETRY_DELAY_S = 2
MAX_RETRY_AFTER_S = 30  # a longer 429 wait would break the "alert within 1 minute" target
MAX_ERROR_CHARS = 300
IMAGE_FILENAME = "ppe_alert.jpg"  # an embed shows it as attachment://ppe_alert.jpg

# Discord's embed limits (https://discord.com/developers/docs/resources/message#embed-object-embed-limits).
EMBED_LIMITS = {"title": 256, "description": 4096, "field_name": 256, "field_value": 1024,
                "footer": 2048, "fields": 25, "total": 6000}

# Severity colours, also used by the web app's badge.
COLOURS = {"High": 0xD93025, "Medium": 0xE37400, "None": 0x188038}
GREY = 0x80868B

AUTOMATED_NOTE = "Automated summary, check against the figures above."
STANDARD_NOTE = "Standard summary, built from the figures above."


def summary_note(automated: bool) -> str:
    return AUTOMATED_NOTE if automated else STANDARD_NOTE


def redact(text: str, url: str | None = None) -> str:
    """Remove the webhook URL and its token from text before it is stored or shown."""
    if url:
        text = text.replace(url, "<webhook>")
    # urllib3 errors quote the path alone: "... with url: /api/webhooks/123/abc?wait=true".
    return re.sub(r"webhooks/\d+/[\w-]+", "webhooks/<redacted>", text)


def escape_markdown(text: str) -> str:
    """Photo names like 'site_1_*.jpg' must stay literal, not turn into italics."""
    return re.sub(r"([\\*_~`|>])", r"\\\1", text)


def shorten_name(name: str, max_chars: int = 40) -> str:
    """Keep the start, the end of the stem and the extension: 'source-56088ecf7582…2bc0a071.jpg'.

    The end is kept because files from one camera often share a long prefix and differ only
    in their last characters (a time, a counter).
    """
    if len(name) <= max_chars:
        return name
    dot = name.rfind(".")
    ext = name[dot:] if 0 < dot and len(name) - dot <= 6 else ""
    stem = name[: len(name) - len(ext)]
    room = max_chars - len(ext) - 1  # 1 for the ellipsis
    tail = min(8, room // 3)
    return stem[: room - tail] + "…" + (stem[-tail:] if tail else "") + ext


def footer_time(moment: datetime, pkt) -> str:
    return moment.astimezone(pkt).strftime("%d %b %Y, %H:%M PKT")


def footer_text(automated: bool, moment: datetime, pkt) -> str:
    """'Automated summary, check against the figures above · 10 Nov 2026, 18:00 PKT'."""
    return f"{summary_note(automated).rstrip('.')} · {footer_time(moment, pkt)}"


def embed_problems(content: str, embeds: list[dict]) -> list[str]:
    """Every Discord limit the message breaks; empty when it can be sent."""
    problems = []
    if len(content) > MAX_CONTENT_CHARS:
        problems.append(f"content is {len(content)} characters (limit {MAX_CONTENT_CHARS})")
    total = 0
    for embed in embeds:
        fields = embed.get("fields", [])
        sizes = {
            "title": len(embed.get("title", "")),
            "description": len(embed.get("description", "")),
            "footer": len(embed.get("footer", {}).get("text", "")),
            "fields": len(fields),
        }
        for key, size in sizes.items():
            if size > EMBED_LIMITS[key]:
                problems.append(f"embed {key} is {size} (limit {EMBED_LIMITS[key]})")
        for field in fields:
            if len(field["name"]) > EMBED_LIMITS["field_name"]:
                problems.append(f"field name {field['name'][:30]!r} is too long")
            if len(field["value"]) > EMBED_LIMITS["field_value"]:
                problems.append(f"field {field['name']!r} value is {len(field['value'])} (limit 1024)")
            if not field["value"].strip():
                problems.append(f"field {field['name']!r} is empty (Discord rejects empty values)")
        total += sizes["title"] + sizes["description"] + sizes["footer"]
        total += sum(len(f["name"]) + len(f["value"]) for f in fields)
    if total > EMBED_LIMITS["total"]:
        problems.append(f"embeds total {total} characters (limit {EMBED_LIMITS['total']})")
    return problems


def post_to_discord(
    url: str, content: str, image_jpg: bytes | None = None, embeds: list[dict] | None = None
) -> requests.Response:
    """One webhook call: text, optional embeds, and the image as an attached file if there is one."""
    payload = {
        "content": content,
        "embeds": embeds or [],
        # Never ping anyone, even if a photo name or summary contains "@everyone".
        "allowed_mentions": {"parse": []},
    }
    params = {"wait": "true"}  # Discord answers 200 only once the message exists
    if image_jpg is None:
        return requests.post(url, params=params, json=payload, timeout=TIMEOUT_S)
    return requests.post(
        url,
        params=params,
        data={"payload_json": json.dumps(payload)},
        files={"files[0]": (IMAGE_FILENAME, image_jpg, "image/jpeg")},
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
    """'HTTP 404: Unknown Webhook', using Discord's own message when there is one."""
    try:
        detail = response.json().get("message") or ""
    except ValueError:
        detail = response.text[:100]
    return f"HTTP {response.status_code}: {detail}".rstrip(": ")


def send_to_discord(
    url: str, content: str, image_jpg: bytes | None = None, embeds: list[dict] | None = None
) -> str | None:
    """Post once and retry once. Returns None on success, or the error (webhook removed)."""
    error = None
    for attempt in (1, 2):
        try:
            response = post_to_discord(url, content, image_jpg, embeds)
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
