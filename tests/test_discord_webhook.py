"""Tests for the shared Discord helpers: name shortening, embed limits, mentions, the summary note."""

import pytest

from src import discord_webhook as dw


# --- Shortening photo names: keep the start and the extension ---

@pytest.mark.parametrize(
    "name, limit, expected",
    [
        ("site.jpg", 40, "site.jpg"),  # short enough: unchanged
        ("source-56088ecf75821c1b93f4015a2bc0a071.jpg", 30, "source-56088ecf75…2bc0a071.jpg"),
        ("IMG_20261110_181500_HDR_panorama_final.jpeg", 24, "IMG_20261110…_final.jpeg"),
        ("no_extension_but_a_very_long_name_indeed", 20, "no_extension_…indeed"),
        ("archive.tar.gz", 10, "arch…ar.gz"),
    ],
)
def test_shorten_name(name, limit, expected):
    """Start, then the last characters of the stem, then the extension."""
    assert dw.shorten_name(name, limit) == expected


def test_names_sharing_a_long_prefix_stay_distinguishable():
    a = dw.shorten_name("site_north_gate_camera_2026-11-10_0800.jpg", 30)
    b = dw.shorten_name("site_north_gate_camera_2026-11-10_0815.jpg", 30)
    assert (a, b) == ("site_north_gate_c…-10_0800.jpg", "site_north_gate_c…-10_0815.jpg")
    assert a != b and len(a) == len(b) == 30


def test_shortened_names_are_exactly_the_limit():
    assert len(dw.shorten_name("source-56088ecf75821c1b93f4015a2bc0a071.jpg", 30)) == 30


def test_shortened_names_fit_the_limit():
    for length in range(1, 120):
        name = "x" * length + ".png"
        assert len(dw.shorten_name(name, 40)) <= 40
        assert dw.shorten_name(name, 40).endswith(".png")


def test_a_dot_far_from_the_end_is_not_an_extension():
    assert dw.shorten_name("v1.2_site_photo_from_the_north_gate_camera", 20) == "v1.2_site_pho…camera"


# --- Embed limits ---

def embed(**overrides):
    base = {"title": "t", "description": "d", "fields": [{"name": "n", "value": "v"}], "footer": {"text": "f"}}
    return base | overrides


def test_a_normal_message_has_no_problems():
    assert dw.embed_problems("hello", [embed()]) == []


@pytest.mark.parametrize(
    "content, overrides, expected",
    [
        ("x" * 2001, {}, "content is 2001 characters"),
        ("", {"title": "t" * 257}, "embed title is 257"),
        ("", {"description": "d" * 4097}, "embed description is 4097"),
        ("", {"footer": {"text": "f" * 2049}}, "embed footer is 2049"),
        ("", {"fields": [{"name": "n", "value": "v"}] * 26}, "embed fields is 26"),
        ("", {"fields": [{"name": "n" * 257, "value": "v"}]}, "field name"),
        ("", {"fields": [{"name": "n", "value": "v" * 1025}]}, "value is 1025"),
        ("", {"fields": [{"name": "n", "value": "  "}]}, "is empty"),
        ("", {"description": "d" * 4000, "fields": [{"name": "n", "value": "v" * 1000}] * 3}, "embeds total"),
    ],
)
def test_each_limit_is_checked(content, overrides, expected):
    problems = dw.embed_problems(content, [embed(**overrides)])
    assert any(expected in p for p in problems), problems


# --- Mentions never ping; embeds and the image are sent together ---

class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append(kwargs)

        class OK:
            ok = True
        return OK()


def test_text_only_message_is_json_with_mentions_off(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(dw.requests, "post", rec)
    assert dw.send_to_discord("https://example.invalid/hook", "@everyone hi", embeds=[embed(title="@here")]) is None
    payload = rec.calls[0]["json"]
    assert payload["allowed_mentions"] == {"parse": []}
    assert payload["content"] == "@everyone hi" and payload["embeds"][0]["title"] == "@here"


def test_image_is_attached_under_the_name_the_embed_uses(monkeypatch):
    import json

    rec = Recorder()
    monkeypatch.setattr(dw.requests, "post", rec)
    dw.send_to_discord("https://example.invalid/hook", "c", b"\xff\xd8jpg", [embed(image={"url": "attachment://ppe_alert.jpg"})])
    call = rec.calls[0]
    assert call["files"]["files[0]"][0] == dw.IMAGE_FILENAME == "ppe_alert.jpg"
    payload = json.loads(call["data"]["payload_json"])
    assert payload["allowed_mentions"] == {"parse": []}
    assert payload["embeds"][0]["image"]["url"] == f"attachment://{dw.IMAGE_FILENAME}"


# --- The disclosure note ---

def test_summary_note_says_when_text_is_automated():
    assert dw.summary_note(True) == "Automated summary, check against the figures above."
    assert dw.summary_note(False) == "Standard summary, built from the figures above."
