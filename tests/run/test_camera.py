import json
import time

import cv2
import numpy as np
import pytest

from defect_inspector import camera as camera_module
from defect_inspector.camera import CameraSettings, CameraWorker, ReplayCamera, load_station, open_camera

SETTINGS = CameraSettings(index=0, width=640, height=480, fps=30, exposure=None, white_balance=None)
FRAME = np.zeros((4, 4, 3), np.uint8)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class FakeCapture:
    """Answers read() from a list of results; True gives a frame, False a failure."""

    def __init__(self, reads):
        self.reads = list(reads)
        self.open_count = 0
        self.release_count = 0

    def open(self, settings):
        self.open_count += 1
        return self

    def isOpened(self):
        return True

    def read(self):
        ok = self.reads.pop(0) if self.reads else True
        return (ok, FRAME.copy()) if ok else (False, None)

    def release(self):
        self.release_count += 1


@pytest.fixture
def fake_clock():
    return FakeClock()


def run_worker(capture, clock, ticks):
    worker = CameraWorker(SETTINGS, open_capture=capture.open, now=clock)
    for _ in range(ticks):
        worker._run_once()
    return worker


def advance(worker, clock, seconds):
    clock.t += seconds
    worker._run_once()
    return worker


def test_latest_returns_newest_frame(fake_clock):
    assert run_worker(FakeCapture([True] * 3), fake_clock, ticks=3).latest()[1] == 3


def test_read_failure_marks_lost_and_reopens(fake_clock):
    capture = FakeCapture([False])
    worker = run_worker(capture, fake_clock, ticks=1)
    assert worker.is_lost() and capture.release_count == 1
    worker = advance(worker, fake_clock, seconds=3)
    assert not worker.is_lost() and capture.open_count == 2


def test_reopening_waits_between_attempts(fake_clock):
    capture = FakeCapture([False])
    worker = run_worker(capture, fake_clock, ticks=1)
    advance(worker, fake_clock, seconds=0.5)
    assert capture.open_count == 1


def test_latest_is_none_while_lost(fake_clock):
    capture = FakeCapture([True, False, False])
    worker = run_worker(capture, fake_clock, ticks=2)
    assert worker.latest() is None


def test_frame_older_than_the_limit_reads_as_lost(fake_clock):
    """A read() stuck inside the driver must not look like a live, frozen camera."""
    worker = run_worker(FakeCapture([True]), fake_clock, ticks=1)
    fake_clock.t += 10
    assert worker.is_lost() and worker.latest() is None


def test_open_that_fails_marks_lost(fake_clock):
    def refuse(settings):
        raise RuntimeError("no camera")

    worker = CameraWorker(SETTINGS, open_capture=refuse, now=fake_clock)
    worker._run_once()
    assert worker.is_lost() and worker.latest() is None


def test_read_that_raises_marks_lost(fake_clock):
    capture = FakeCapture([])

    def explode():
        raise cv2.error("driver")

    capture.read = explode
    assert run_worker(capture, fake_clock, ticks=1).is_lost()


def test_worker_thread_delivers_a_frame_and_stops():
    worker = CameraWorker(SETTINGS, open_capture=FakeCapture([]).open)
    worker.start()
    try:
        for _ in range(1000):
            if worker.latest() is not None:
                break
            time.sleep(0.005)
        assert worker.latest() is not None
    finally:
        worker.stop()
    assert not worker._thread.is_alive()


def test_configure_reopens_with_the_new_settings_on_the_same_camera(fake_clock):
    opened = []
    capture = FakeCapture([])

    def open_capture(settings):
        opened.append(settings)
        return capture

    worker = CameraWorker(SETTINGS._replace(index=2), open_capture=open_capture, now=fake_clock)
    worker._run_once()
    worker.configure(SETTINGS._replace(index=0, width=1280))
    worker._run_once()
    assert capture.release_count == 1 and len(opened) == 2
    assert opened[1].width == 1280 and opened[1].index == 2


def test_configure_with_the_same_settings_does_not_reopen(fake_clock):
    capture = FakeCapture([])
    worker = CameraWorker(SETTINGS, open_capture=capture.open, now=fake_clock)
    worker._run_once()
    worker.configure(SETTINGS)
    worker._run_once()
    assert capture.open_count == 1


def test_replay_ignores_camera_settings(tmp_path):
    cv2.imwrite(str(tmp_path / "0.png"), FRAME)
    camera = ReplayCamera(tmp_path)
    camera.configure(SETTINGS)
    assert camera.latest() is not None


@pytest.fixture
def image_folder(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / "{}.png".format(i)), np.full((4, 4, 3), i * 50, np.uint8))
    return tmp_path


def test_replay_cycles_through_folder(image_folder):
    camera = ReplayCamera(image_folder)
    assert camera.latest() is not None and not camera.is_lost()


def test_replay_moves_on_with_time_and_wraps(image_folder, fake_clock):
    camera = ReplayCamera(image_folder, fps=1, now=fake_clock)
    seen = []
    for _ in range(4):
        frame, frame_id = camera.latest()
        seen.append((int(frame[0, 0, 0]), frame_id))
        fake_clock.t += 1
    assert seen == [(0, 0), (50, 1), (100, 2), (0, 3)]


def test_replay_repeats_the_id_within_one_frame_time(image_folder, fake_clock):
    camera = ReplayCamera(image_folder, fps=1, now=fake_clock)
    first = camera.latest()[1]
    fake_clock.t += 0.2
    assert camera.latest()[1] == first


def test_replay_finds_images_in_subfolders(tmp_path):
    (tmp_path / "scratch").mkdir()
    cv2.imwrite(str(tmp_path / "scratch" / "0.png"), FRAME)
    assert ReplayCamera(tmp_path).latest() is not None


def test_replay_of_empty_folder_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no images"):
        ReplayCamera(tmp_path)


def test_load_station_reads_camera_and_roi(tmp_path):
    path = tmp_path / "station.json"
    path.write_text(json.dumps({
        "roi": [1, 2, 30, 40],
        "camera": {"index": 1, "width": 1280, "height": 720, "fps": 30, "exposure": -5, "white_balance": 4000},
    }), encoding="utf-8")
    station = load_station(path)
    assert station.roi == [1, 2, 30, 40]
    assert station.camera == CameraSettings(1, 1280, 720, 30, -5, 4000)


class RecordingVideoCapture:
    last = None

    def __init__(self, index, api):
        self.index, self.api, self.props = index, api, {}
        RecordingVideoCapture.last = self

    def set(self, prop, value):
        self.props[prop] = value
        return True


@pytest.fixture
def recording_capture(monkeypatch):
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", RecordingVideoCapture)
    return RecordingVideoCapture


@pytest.mark.parametrize("platform, api", [("win32", cv2.CAP_DSHOW), ("linux", cv2.CAP_V4L2)])
def test_open_camera_picks_the_driver_of_the_platform(recording_capture, monkeypatch, platform, api):
    monkeypatch.setattr(camera_module.sys, "platform", platform)
    assert open_camera(SETTINGS).api == api


def test_open_camera_asks_for_mjpg_and_size(recording_capture):
    props = open_camera(SETTINGS).props
    assert props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert (props[cv2.CAP_PROP_FRAME_WIDTH], props[cv2.CAP_PROP_FRAME_HEIGHT]) == (640, 480)


def test_open_camera_leaves_automatic_settings_when_none_are_given(recording_capture):
    props = open_camera(SETTINGS).props
    assert cv2.CAP_PROP_EXPOSURE not in props and cv2.CAP_PROP_WB_TEMPERATURE not in props


def test_open_camera_locks_exposure_and_white_balance_when_given(recording_capture, monkeypatch):
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    props = open_camera(SETTINGS._replace(exposure=-5, white_balance=4000)).props
    assert props[cv2.CAP_PROP_AUTO_EXPOSURE] == 1 and props[cv2.CAP_PROP_EXPOSURE] == -5
    assert props[cv2.CAP_PROP_AUTO_WB] == 0 and props[cv2.CAP_PROP_WB_TEMPERATURE] == 4000
