"""The camera, in its own thread, because opening it and reading from it both block.

Only the newest frame is kept. A frame that is too old reads as no frame at all, so a
stuck camera can never look like a live one. Runs on Python 3.6, on the Jetson Nano.
Adapted from CameraWorker in jetson-car-logger.
"""
import json
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_CAMERA = {"index": 0, "width": 1280, "height": 720, "fps": 30, "exposure": None, "white_balance": None,
                  "focus": None, "gain": None}
# OpenCV gives "manual exposure" and "automatic exposure" different values on each driver.
MANUAL_EXPOSURE = {"win32": 0.25}
MANUAL_EXPOSURE_LINUX = 1
AUTO_EXPOSURE = {"win32": 0.75}
AUTO_EXPOSURE_LINUX = 3

CameraSettings = namedtuple("CameraSettings", ["index", "width", "height", "fps", "exposure", "white_balance",
                                               "focus", "gain"])
# Stations written before focus and gain were kept still load.
CameraSettings.__new__.__defaults__ = (None, None)
Station = namedtuple("Station", ["camera", "roi"])


def settings_from_dict(fields):
    merged = dict(DEFAULT_CAMERA)
    merged.update(fields or {})
    return CameraSettings(**{name: merged[name] for name in CameraSettings._fields})


def load_station(path):
    fields = json.loads(Path(path).read_text(encoding="utf-8"))
    return Station(settings_from_dict(fields.get("camera")), fields.get("roi"))


def open_camera(settings):
    api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
    capture = cv2.VideoCapture(settings.index, api)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    capture.set(cv2.CAP_PROP_FPS, settings.fps)
    # A camera keeps manual mode after it is closed, so "not locked" has to turn automatic back on.
    if settings.exposure is None:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE.get(sys.platform, AUTO_EXPOSURE_LINUX))
    else:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, MANUAL_EXPOSURE.get(sys.platform, MANUAL_EXPOSURE_LINUX))
        capture.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
    if settings.white_balance is None:
        capture.set(cv2.CAP_PROP_AUTO_WB, 1)
    else:
        capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        capture.set(cv2.CAP_PROP_WB_TEMPERATURE, settings.white_balance)
    if settings.focus is None:
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, settings.focus)
    if settings.gain is not None:
        capture.set(cv2.CAP_PROP_GAIN, settings.gain)
    return capture


class CameraWorker:
    def __init__(self, settings, open_capture=None, now=None, stale_after_s=2.0, reopen_backoff_s=2.0,
                 keep_index=True):
        self._settings = settings
        self._keep_index = keep_index
        self._open_capture = open_capture or open_camera
        self._now = now or time.monotonic
        self._stale_after_s = stale_after_s
        self._reopen_backoff_s = reopen_backoff_s
        self._lock = threading.Lock()
        self._cap = None
        self._frame = None
        self._frame_id = 0
        self._last_frame_at = None
        self._last_attempt_at = None
        self._failed = False
        self._reopen = False
        self._running = False
        self._thread = None

    def _is_stale(self):
        return self._last_frame_at is not None and self._now() - self._last_frame_at > self._stale_after_s

    def is_lost(self):
        with self._lock:
            return self._failed or self._is_stale()

    def latest(self):
        """Return (frame, frame_id), or None while there is no fresh frame."""
        with self._lock:
            if self._failed or self._frame is None or self._is_stale():
                return None
            return self._frame, self._frame_id

    def _fail(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._last_attempt_at = self._now()
        with self._lock:
            self._failed = True

    def configure(self, settings):
        """Use these settings; the reopening happens in the camera thread.

        A camera chosen by hand stays; otherwise the settings also say which camera to use.
        """
        if self._keep_index:
            settings = settings._replace(index=self._settings.index)
        with self._lock:
            if settings != self._settings:
                self._settings = settings
                self._reopen = True

    def _ensure_open(self):
        with self._lock:
            reopen, self._reopen = self._reopen, False
        if reopen and self._cap is not None:
            self._cap.release()
            self._cap = None
            self._last_attempt_at = None
        if self._cap is not None:
            if self._cap.isOpened():
                return True
            self._fail()
            return False
        if self._last_attempt_at is not None and self._now() - self._last_attempt_at < self._reopen_backoff_s:
            return False
        try:
            self._cap = self._open_capture(self._settings)
        except Exception:
            self._fail()
            return False
        self._last_attempt_at = self._now()
        if not self._cap.isOpened():
            self._fail()
            return False
        return True

    def _run_once(self):
        """One iteration: open if needed, then read once. Sleeps nowhere, so it can be tested."""
        if not self._ensure_open():
            return False
        try:
            ok, frame = self._cap.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            self._fail()
            return False
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            self._last_frame_at = self._now()
            self._failed = False
        return True

    def _loop(self):
        while self._running:
            if not self._run_once():
                time.sleep(0.05)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class ReplayCamera:
    """Plays a folder of images in a circle, with the same interface as CameraWorker."""

    def __init__(self, folder, fps=5, now=None):
        self._paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self._paths:
            raise ValueError("no images in {}".format(folder))
        self._fps = fps
        self._now = now or time.monotonic
        self._started_at = self._now()
        self._lock = threading.Lock()
        self._cached_id = None
        self._cached_frame = None

    def start(self):
        pass

    def stop(self):
        pass

    def configure(self, settings):
        pass

    def is_lost(self):
        return False

    def latest(self):
        frame_id = int((self._now() - self._started_at) * self._fps)
        with self._lock:
            if frame_id != self._cached_id:
                self._cached_frame = cv2.imread(str(self._paths[frame_id % len(self._paths)]))
                self._cached_id = frame_id
            if self._cached_frame is None:
                return None
            return self._cached_frame, frame_id
