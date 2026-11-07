import re
from pathlib import Path

import defect_inspector

STATIC = Path(defect_inspector.__file__).parent / "static"
INDEX = STATIC / "index.html"


def static_files():
    return [p for p in STATIC.rglob("*") if p.is_file()]


def test_no_external_assets():
    for f in static_files():
        text = f.read_text(encoding="utf-8")
        assert not re.search(r'(src|href)="https?://', text)
        assert not re.search(r'@import|url\(\s*["\']?https?:', text)


def test_every_status_of_the_loop_has_a_look():
    text = INDEX.read_text(encoding="utf-8")
    for status in ("NO_PROFILE", "CAMERA_LOST", "EMPTY", "MOVING", "ERROR"):
        assert status + ":" in text


def test_page_uses_every_number_from_the_state():
    text = INDEX.read_text(encoding="utf-8")
    for field in ("state.score", "state.threshold", "state.latency_ms", "state.message", "state.label"):
        assert field in text


def test_stale_image_note_really_hides():
    """A display rule on the note would otherwise beat the hidden attribute and show it always."""
    assert ".view-note[hidden] { display: none; }" in INDEX.read_text(encoding="utf-8")


def test_page_asks_for_no_icon_it_does_not_have():
    assert 'rel="icon" href="data:,"' in INDEX.read_text(encoding="utf-8")


def test_verdict_is_announced_and_motion_can_be_turned_off():
    text = INDEX.read_text(encoding="utf-8")
    assert 'aria-live="polite"' in text and "prefers-reduced-motion" in text
