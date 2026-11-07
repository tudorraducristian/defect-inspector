"""The windowed parts of capture.py, with the keys scripted and no real window or camera."""
import json

import cv2
import numpy as np
import pytest

import capture
from capture import main, plan_session, run_session, run_setup, shot_path
from defect_inspector.camera import CameraSettings, load_station

FRAME = np.full((60, 80, 3), 120, np.uint8)
SETTINGS = CameraSettings(1, 80, 60, 30, None, None)
ENTER, ESC, NOTHING = 13, 27, -1


class Screen:
    """Stands in for the OpenCV windows: hands out the scripted keys, keeps what was shown."""

    def __init__(self, monkeypatch, keys, roi=(1, 2, 30, 40)):
        self.keys = list(keys)
        self.shown = []
        self.closed = False
        monkeypatch.setattr(capture.cv2, "imshow", lambda name, image: self.shown.append(image.copy()))
        monkeypatch.setattr(capture.cv2, "waitKey", self.next_key)
        monkeypatch.setattr(capture.cv2, "destroyAllWindows", self.close)
        monkeypatch.setattr(capture.cv2, "selectROI", lambda *a, **k: roi)

    def next_key(self, delay):
        if not self.keys:
            raise AssertionError("the tool kept waiting after the last scripted key")
        return self.keys.pop(0)

    def close(self):
        self.closed = True


class StillCamera:
    def __init__(self, answers=None):
        self.answers = list(answers or [])

    def latest(self):
        if self.answers:
            return self.answers.pop(0)
        return FRAME, 1


class FakeCapture:
    def __init__(self, settings):
        self.settings = settings
        self.released = False

    def read(self):
        return True, FRAME.copy()

    def get(self, prop):
        return {cv2.CAP_PROP_EXPOSURE: -6.0, cv2.CAP_PROP_WB_TEMPERATURE: 4600.0}[prop]

    def release(self):
        self.released = True


@pytest.fixture
def opened(monkeypatch):
    captures = []

    def fake_open(settings):
        captures.append(FakeCapture(settings))
        return captures[-1]

    monkeypatch.setattr(capture, "open_camera", fake_open)
    return captures


@pytest.fixture(autouse=True)
def no_wait_between_shots(monkeypatch):
    monkeypatch.setattr(capture, "MIN_SECONDS_BETWEEN_SHOTS", 0.0)


def test_session_takes_one_photo_per_enter(monkeypatch, tmp_path):
    Screen(monkeypatch, [NOTHING, ENTER, ENTER, NOTHING, ENTER, ENTER])
    assert run_session(StillCamera(), plan_session(tmp_path, "good", pieces=2, shots=2), None) is True
    assert sorted(p.name for p in tmp_path.rglob("*.jpg")) == ["shot-01.jpg", "shot-01.jpg", "shot-02.jpg", "shot-02.jpg"]


def test_esc_stops_the_session_and_keeps_the_photos_taken(monkeypatch, tmp_path):
    Screen(monkeypatch, [ENTER, ESC])
    assert run_session(StillCamera(), plan_session(tmp_path, "good", pieces=2, shots=2), None) is False
    assert [p.name for p in tmp_path.rglob("*.jpg")] == ["shot-01.jpg"]


def test_enter_held_down_takes_a_single_photo(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "MIN_SECONDS_BETWEEN_SHOTS", 60.0)
    Screen(monkeypatch, [ENTER, ENTER, ENTER, ESC])
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=3), None)
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


def test_photo_has_no_text_or_zone_drawn_on_it(monkeypatch, tmp_path):
    screen = Screen(monkeypatch, [ENTER])
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=1), [5, 5, 20, 20])
    saved = cv2.imread(str(shot_path(tmp_path, "good", 1, 1)))
    assert np.abs(saved.astype(int) - FRAME.astype(int)).max() <= 2
    assert not np.array_equal(screen.shown[0], FRAME)


def test_window_is_closed_at_the_end(monkeypatch, tmp_path):
    screen = Screen(monkeypatch, [ENTER])
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=1), None)
    assert screen.closed


def test_lost_camera_shows_a_message_and_esc_still_stops(monkeypatch, tmp_path):
    screen = Screen(monkeypatch, [NOTHING, ESC])
    camera = StillCamera([None, None, None])
    assert run_session(camera, plan_session(tmp_path, "good", pieces=1, shots=1), None) is False
    assert screen.shown and not list(tmp_path.rglob("*.jpg"))


def test_enter_while_the_camera_is_lost_takes_no_photo(monkeypatch, tmp_path):
    Screen(monkeypatch, [ENTER, ENTER])
    camera = StillCamera([None])
    run_session(camera, plan_session(tmp_path, "good", pieces=1, shots=1), None)
    assert len(list(tmp_path.rglob("*.jpg"))) == 1


def test_setup_writes_the_station_with_locked_settings(monkeypatch, tmp_path, opened):
    Screen(monkeypatch, [NOTHING, ENTER, ENTER])
    assert run_setup(SETTINGS, tmp_path) is True
    station = load_station(tmp_path / "station.json")
    assert station.roi == [1, 2, 30, 40]
    assert (station.camera.exposure, station.camera.white_balance) == (-6.0, 4600.0)
    assert station.camera.index == 1
    assert cv2.imread(str(tmp_path / "background.jpg")) is not None


def test_setup_takes_the_table_photo_with_the_settings_already_locked(monkeypatch, tmp_path, opened):
    Screen(monkeypatch, [ENTER, ENTER])
    run_setup(SETTINGS, tmp_path)
    assert len(opened) == 2 and opened[0].released
    assert opened[1].settings.exposure == -6.0 and opened[1].released


def test_esc_during_setup_writes_nothing(monkeypatch, tmp_path, opened):
    Screen(monkeypatch, [ESC])
    assert run_setup(SETTINGS, tmp_path) is False
    assert not (tmp_path / "station.json").exists() and opened[-1].released


def test_empty_zone_writes_nothing(monkeypatch, tmp_path, opened):
    Screen(monkeypatch, [ENTER], roi=(0, 0, 0, 0))
    assert run_setup(SETTINGS, tmp_path) is False
    assert not (tmp_path / "station.json").exists()


def write_station(root):
    root.mkdir(parents=True)
    (root / "station.json").write_text(json.dumps({"roi": None, "camera": {"index": 0}}), encoding="utf-8")


@pytest.mark.parametrize("arguments", [
    ["--session", "good"],
    ["--session", "defect"],
    ["--kind", "good", "--headless"],
    ["--kind", "defect", "--piece", "1", "--headless"],
    ["--piece", "1"],
    [],
])
def test_incomplete_command_is_refused_before_the_camera_starts(tmp_path, arguments, monkeypatch):
    write_station(tmp_path / "x")

    def no_camera(*args, **kwargs):
        raise AssertionError("the camera must not start")

    monkeypatch.setattr(capture, "CameraWorker", no_camera)
    with pytest.raises(SystemExit) as stopped:
        main(["--object", "x", "--data", str(tmp_path)] + arguments)
    assert stopped.value.code == 2


def test_session_without_a_setup_says_what_to_do(tmp_path, capsys):
    assert main(["--object", "x", "--data", str(tmp_path), "--session", "good", "--pieces", "1"]) == 2
    assert "--setup" in capsys.readouterr().err
