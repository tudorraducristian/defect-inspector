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
        self.window_size = None
        monkeypatch.setattr(capture.cv2, "namedWindow", lambda name, flags: setattr(self, "flags", flags))
        monkeypatch.setattr(capture.cv2, "resizeWindow", lambda name, w, h: setattr(self, "window_size", (w, h)))

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


CAMERA_VALUES = {cv2.CAP_PROP_EXPOSURE: -6.0, cv2.CAP_PROP_WB_TEMPERATURE: 4600.0,
                 cv2.CAP_PROP_FOCUS: 30.0, cv2.CAP_PROP_GAIN: 64.0}
# What OpenCV gives for a control the camera does not have.
NO_CONTROLS = {prop: -1.0 for prop in CAMERA_VALUES}


class FakeCapture:
    values = CAMERA_VALUES

    def __init__(self, settings):
        self.settings = settings
        self.released = False

    def read(self):
        return True, FRAME.copy()

    def get(self, prop):
        return self.values[prop]

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


def test_setup_locks_focus_and_gain_too(monkeypatch, tmp_path, opened):
    Screen(monkeypatch, [ENTER, ENTER])
    run_setup(SETTINGS, tmp_path)
    camera = load_station(tmp_path / "station.json").camera
    assert (camera.focus, camera.gain) == (30.0, 64.0)
    assert (opened[1].settings.focus, opened[1].settings.gain) == (30.0, 64.0)


def test_setup_leaves_unlocked_what_the_camera_does_not_have(monkeypatch, tmp_path, opened):
    monkeypatch.setattr(FakeCapture, "values", NO_CONTROLS)
    Screen(monkeypatch, [ENTER, ENTER])
    run_setup(SETTINGS, tmp_path)
    camera = load_station(tmp_path / "station.json").camera
    assert (camera.exposure, camera.white_balance, camera.focus, camera.gain) == (None, None, None, None)


def test_setup_reports_what_the_camera_holds_after_locking(monkeypatch, tmp_path, opened, capsys):
    values = dict(CAMERA_VALUES)
    values[cv2.CAP_PROP_FOCUS] = -1.0
    monkeypatch.setattr(FakeCapture, "values", values)
    Screen(monkeypatch, [ENTER, ENTER])
    run_setup(SETTINGS, tmp_path)
    out = capsys.readouterr().out
    assert "expunere -6" in out and "balans alb 4600" in out and "castig 64" in out
    assert "focalizare: nu se poate bloca" in out


def test_setup_refuses_to_replace_a_station_that_already_has_photos(tmp_path, monkeypatch, capsys):
    write_station(tmp_path / "x")
    shot = shot_path(tmp_path / "x", "good", 1, 1)
    shot.parent.mkdir(parents=True)
    cv2.imwrite(str(shot), FRAME)
    ran = []
    monkeypatch.setattr(capture, "run_setup", lambda settings, root, lock_from_camera=True: ran.append(root) or True)
    assert main(["--object", "x", "--data", str(tmp_path), "--setup"]) == 2
    assert "--force" in capsys.readouterr().err and not ran
    assert main(["--object", "x", "--data", str(tmp_path), "--setup", "--force"]) == 0
    assert ran


def shown_text(monkeypatch):
    texts = []
    real = capture.cv2.putText
    monkeypatch.setattr(capture.cv2, "putText", lambda image, text, *a: texts.append(text) or real(image, text, *a))
    return texts


def test_session_says_when_to_turn_the_piece_over(monkeypatch, tmp_path):
    texts = shown_text(monkeypatch)
    Screen(monkeypatch, [ENTER] * 5)
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=5), None, flip_after=3)
    face_a = [t for t in texts if "Fata A" in t]
    face_b = [t for t in texts if "Intoarce piulita" in t]
    # Drawn twice per shown frame (outline, then letters): 3 shots on face A, 2 on face B.
    assert len(face_a) == 6 and len(face_b) == 4


def test_session_without_turning_says_nothing_about_faces(monkeypatch, tmp_path):
    texts = shown_text(monkeypatch)
    Screen(monkeypatch, [ENTER] * 3)
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=3), None)
    assert not [t for t in texts if "Fata" in t or "Intoarce" in t]


def test_resumed_piece_still_knows_its_face(monkeypatch, tmp_path):
    for shot in (1, 2, 3):
        path = shot_path(tmp_path, "good", 1, shot)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), FRAME)
    texts = shown_text(monkeypatch)
    Screen(monkeypatch, [ENTER] * 2)
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=5), None, flip_after=3)
    assert not [t for t in texts if "Fata A" in t] and [t for t in texts if "Intoarce piulita" in t]


def test_setup_with_measured_values_keeps_them_instead_of_asking_the_camera(monkeypatch, tmp_path, opened):
    # Some webcams report their defaults, not what they use, so a lock read from them is wrong.
    Screen(monkeypatch, [ENTER, ENTER])
    run_setup(SETTINGS._replace(exposure=-5.0, focus=325.0), tmp_path, lock_from_camera=False)
    camera = load_station(tmp_path / "station.json").camera
    assert (camera.exposure, camera.focus, camera.white_balance, camera.gain) == (-5.0, 325.0, None, None)
    assert opened[0].settings.focus == 325.0


def test_measured_values_on_the_command_line_reach_the_setup(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(capture, "run_setup", lambda settings, root, lock_from_camera=True:
                        calls.append((settings, lock_from_camera)) or True)
    main(["--object", "x", "--data", str(tmp_path), "--setup", "--camera", "1", "--exposure", "-5", "--focus", "325"])
    main(["--object", "x", "--data", str(tmp_path), "--setup", "--camera", "1"])
    (fixed, fixed_lock), (read, read_lock) = calls
    assert (fixed.exposure, fixed.focus, fixed.white_balance, fixed_lock) == (-5.0, 325.0, None, False)
    assert (read.exposure, read_lock) == (None, True)


def test_windows_fit_a_laptop_screen_whatever_the_camera_size(monkeypatch, tmp_path, opened):
    # A 1920x1080 frame shown at full size runs off a laptop screen, zone selection included.
    screen = Screen(monkeypatch, [ESC])
    run_setup(SETTINGS, tmp_path)
    assert screen.flags == cv2.WINDOW_NORMAL and screen.window_size[0] <= 1280
    screen = Screen(monkeypatch, [ESC])
    run_session(StillCamera(), plan_session(tmp_path, "good", pieces=1, shots=1), None)
    assert screen.flags == cv2.WINDOW_NORMAL and screen.window_size[0] <= 1280


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
