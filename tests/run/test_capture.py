import cv2
import numpy as np
import pytest

from capture import (
    build_station,
    capture_shots,
    next_shots,
    parse_types,
    plan_session,
    shot_path,
    write_station,
)
from defect_inspector.camera import CameraSettings, ReplayCamera, load_station

FRAME = np.zeros((8, 8, 3), np.uint8)
SETTINGS = CameraSettings(0, 640, 480, 30, -5, 4000)


def save_shots(root, kind, piece, count, defect_type=None):
    for shot in range(1, count + 1):
        path = shot_path(root, kind, piece, shot, defect_type)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), FRAME)


@pytest.fixture
def folder_with_two_shots(tmp_path):
    save_shots(tmp_path, "good", 7, 2)
    return tmp_path


def test_shot_path_shape(tmp_path):
    assert shot_path(tmp_path, "defect", 2, 1, "marker").as_posix().endswith("defect/marker/piece-02/shot-01.jpg")
    assert shot_path(tmp_path, "good", 7, 3).as_posix().endswith("good/piece-07/shot-03.jpg")


def test_defect_shot_needs_a_type(tmp_path):
    with pytest.raises(ValueError, match="type"):
        shot_path(tmp_path, "defect", 2, 1)


def test_next_shots_never_overwrite(folder_with_two_shots):
    assert next_shots(folder_with_two_shots, "good", 7, 1)[0].name == "shot-03.jpg"


def test_next_shots_continue_after_a_gap(folder_with_two_shots):
    shot_path(folder_with_two_shots, "good", 7, 1).unlink()
    assert next_shots(folder_with_two_shots, "good", 7, 1)[0].name == "shot-03.jpg"


def test_parse_types():
    assert parse_types("zgarietura=5,marker=5,lovitura=4") == [("zgarietura", 5), ("marker", 5), ("lovitura", 4)]


@pytest.mark.parametrize("text", ["marker", "marker=x", "good=5", "a b=5", "marker=0"])
def test_bad_types_are_refused(text):
    with pytest.raises(ValueError):
        parse_types(text)


def test_good_session_goes_through_every_piece(tmp_path):
    steps = plan_session(tmp_path, "good", pieces=2, shots=3)
    assert [(s.piece, s.shots_done, s.shots_total) for s in steps] == [(1, 0, 3), (2, 0, 3)]
    assert len(steps[0].paths) == 3


def test_defect_session_numbers_pieces_within_each_type(tmp_path):
    steps = plan_session(tmp_path, "defect", types=[("marker", 2), ("lovitura", 1)], shots=3)
    assert [(s.defect_type, s.piece) for s in steps] == [("marker", 1), ("marker", 2), ("lovitura", 1)]


def test_stopped_session_continues_where_it_was(tmp_path):
    save_shots(tmp_path, "good", 1, 3)
    save_shots(tmp_path, "good", 2, 1)
    steps = plan_session(tmp_path, "good", pieces=3, shots=3)
    assert [(s.piece, s.shots_done) for s in steps] == [(2, 1), (3, 0)]
    assert steps[0].paths[0].name == "shot-02.jpg" and len(steps[0].paths) == 2


def test_capture_shots_writes_each_path(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cv2.imwrite(str(source / "0.png"), FRAME)
    paths = [tmp_path / "out" / "shot-01.jpg", tmp_path / "out" / "shot-02.jpg"]
    capture_shots(ReplayCamera(source, fps=1000), paths)
    assert all(p.exists() for p in paths)


def test_capture_shots_gives_up_without_frames(tmp_path):
    class NoFrames:
        def latest(self):
            return None

    with pytest.raises(RuntimeError, match="camera"):
        capture_shots(NoFrames(), [tmp_path / "shot-01.jpg"], timeout_s=0.01)


def test_station_round_trip(tmp_path):
    write_station(tmp_path, build_station(SETTINGS, [1, 2, 3, 4]), FRAME)
    station = load_station(tmp_path / "station.json")
    assert station.camera == SETTINGS and station.roi == [1, 2, 3, 4]
    assert cv2.imread(str(tmp_path / "background.jpg")) is not None
