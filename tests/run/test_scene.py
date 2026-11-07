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


def noisy(frame, seed, amount=3):
    noise = np.random.RandomState(seed).randint(-amount, amount + 1, frame.shape)
    return np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)


def test_empty_table_is_empty():
    assert SceneGate(TABLE).update(TABLE) == "EMPTY"


def test_camera_noise_on_the_table_is_still_empty():
    gate = SceneGate(TABLE)
    assert {gate.update(noisy(TABLE, seed)) for seed in range(5)} == {"EMPTY"}


def test_piece_becomes_stable_only_after_enough_still_frames():
    gate = SceneGate(TABLE, stable_frames=5)
    results = [gate.update(PIECE) for _ in range(6)]
    assert results == ["MOVING"] * 5 + ["STABLE"]


def test_noise_does_not_stop_a_still_piece_from_being_stable():
    gate = SceneGate(TABLE, stable_frames=5)
    assert [gate.update(noisy(PIECE, seed)) for seed in range(6)][-1] == "STABLE"


def test_moving_the_piece_restarts_the_wait():
    gate = SceneGate(TABLE, stable_frames=2)
    for _ in range(3):
        gate.update(PIECE)
    assert gate.update(ELSEWHERE) == "MOVING"
    assert gate.update(ELSEWHERE) == "MOVING"
    assert gate.update(ELSEWHERE) == "STABLE"


def test_reset_forgets_the_still_frames():
    gate = SceneGate(TABLE, stable_frames=2)
    for _ in range(3):
        gate.update(PIECE)
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
