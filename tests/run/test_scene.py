import cv2
import numpy as np
import pytest

from defect_inspector.profile import Profile
from defect_inspector.scene import SceneGate, gate_for_profile

TABLE = np.full((120, 160, 3), 220, np.uint8)
PIECE = TABLE.copy()
PIECE[30:90, 50:110] = 40
ELSEWHERE = TABLE.copy()
ELSEWHERE[10:70, 90:150] = 40

# Like the station: a gray sheet, and a nut that covers about a tenth of the zone.
GRAY_TABLE = np.full((400, 400, 3), 90, np.uint8)
SMALL_NUT = GRAY_TABLE.copy()
cv2.circle(SMALL_NUT, (200, 200), 72, (200, 200, 200), -1)
cv2.circle(SMALL_NUT, (200, 200), 34, (60, 60, 60), -1)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def noisy(frame, seed, amount=3):
    noise = np.random.RandomState(seed).randint(-amount, amount + 1, frame.shape)
    return np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)


def shifted(frame, dx):
    return np.roll(frame, dx, axis=1)


def feed(gate, clock, frames, fps=30):
    results = []
    for frame in frames:
        results.append(gate.update(frame))
        clock.t += 1.0 / fps
    return results


def test_empty_table_is_empty():
    assert SceneGate(TABLE).update(TABLE) == "EMPTY"


def test_camera_noise_on_the_table_is_still_empty():
    gate = SceneGate(TABLE)
    assert {gate.update(noisy(TABLE, seed)) for seed in range(5)} == {"EMPTY"}


def test_small_nut_on_a_gray_sheet_is_present():
    # The mean difference over the zone stays low for a small piece; the share of changed pixels does not.
    assert SceneGate(GRAY_TABLE).update(SMALL_NUT) != "EMPTY"


@pytest.mark.parametrize("gain", [0.8, 1.2])
def test_light_change_on_the_empty_table_is_still_empty(gain):
    darker = np.clip(GRAY_TABLE.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    assert SceneGate(GRAY_TABLE).update(darker) == "EMPTY"


def test_light_change_does_not_hide_the_piece():
    darker = np.clip(SMALL_NUT.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    assert SceneGate(GRAY_TABLE).update(darker) != "EMPTY"


def test_piece_becomes_stable_only_after_it_stays_still_long_enough():
    clock = Clock()
    gate = SceneGate(TABLE, still_seconds=0.5, now=clock)
    results = feed(gate, clock, [PIECE] * 20)
    assert results[:15] == ["MOVING"] * 15 and results[-1] == "STABLE"


def test_noise_does_not_stop_a_still_piece_from_being_stable():
    clock = Clock()
    gate = SceneGate(TABLE, still_seconds=0.5, now=clock)
    assert feed(gate, clock, [noisy(PIECE, seed) for seed in range(30)])[-1] == "STABLE"


def test_slow_motion_never_counts_as_still():
    # One pixel per frame is under any frame-to-frame threshold, but it adds up.
    clock = Clock()
    gate = SceneGate(GRAY_TABLE, still_seconds=0.5, now=clock)
    results = feed(gate, clock, [shifted(SMALL_NUT, dx) for dx in range(60)])
    assert "STABLE" not in results


def test_moving_the_piece_restarts_the_wait():
    clock = Clock()
    gate = SceneGate(TABLE, still_seconds=0.2, now=clock)
    feed(gate, clock, [PIECE] * 10)
    results = feed(gate, clock, [ELSEWHERE] * 10)
    assert results[:6] == ["MOVING"] * 6 and results[-1] == "STABLE"


def test_reset_forgets_the_still_frames():
    clock = Clock()
    gate = SceneGate(TABLE, still_seconds=0.2, now=clock)
    feed(gate, clock, [PIECE] * 10)
    gate.reset()
    assert gate.update(PIECE) == "MOVING"


def profile_at(folder, roi=None):
    return Profile("x", "X", folder, 32, 0.5, [0.4] * 3, [0.2] * 3, 4, roi, None, {})


def test_gate_comes_from_the_table_photo_in_the_profile(tmp_path):
    cv2.imwrite(str(tmp_path / "background.jpg"), TABLE)
    gate = gate_for_profile(profile_at(tmp_path))
    assert gate.update(TABLE) == "EMPTY" and gate.update(PIECE) == "MOVING"


def test_profile_without_a_table_photo_has_no_gate(tmp_path):
    assert gate_for_profile(profile_at(tmp_path)) is None


def test_gate_looks_only_at_the_zone_of_interest(tmp_path):
    frame = np.zeros((120, 160, 3), np.uint8)
    frame[20:60, 20:60] = 220
    cv2.imwrite(str(tmp_path / "background.png"), frame)
    (tmp_path / "background.jpg").write_bytes((tmp_path / "background.png").read_bytes())
    gate = gate_for_profile(profile_at(tmp_path, roi=[20, 20, 40, 40]))
    assert gate.update(TABLE[:40, :40]) == "EMPTY"


def test_unreadable_table_photo_is_an_error(tmp_path):
    (tmp_path / "background.jpg").write_bytes(b"broken")
    with pytest.raises(ValueError, match="background"):
        gate_for_profile(profile_at(tmp_path))
