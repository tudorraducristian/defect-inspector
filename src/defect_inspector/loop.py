"""The inspection loop: camera, scene gate, one verdict per piece, and the state for the page.

The model runs in the thread of this loop, outside the lock, so the page keeps reading the
state and the live image while the model works. Runs on Python 3.6, on the Jetson Nano.
"""
import logging
import threading
import time

import cv2

from defect_inspector.postprocessing import crop_roi, render_overlay
from defect_inspector.scene import EMPTY, MOVING, STABLE, SceneGate

log = logging.getLogger("defect_inspector.loop")

NO_PROFILE = "NO_PROFILE"
CAMERA_LOST = "CAMERA_LOST"
RESULT = "RESULT"
ERROR = "ERROR"

MESSAGES = {
    NO_PROFILE: "Alege un obiect.",
    CAMERA_LOST: "Camera nu dă imagine. Reîncerc singur.",
    EMPTY: "Pune o piesă sub cameră.",
    MOVING: "Aștept să stea nemișcată piesa.",
    RESULT: "",
}
MESSAGE_ANALYZING = "Analizez piesa."
JPEG_QUALITY = 80
ZONE_COLOR = (0, 200, 0)
STEP_PAUSE_S = 0.02


class InspectionLoop:
    def __init__(self, camera, gate=None):
        self._camera = camera
        self._gate = gate
        self._inspector = None
        self._profile = None
        self._lock = threading.Lock()
        # Goes up with each change of profile or table photo, so a verdict that was being
        # computed for the previous one is dropped.
        self._generation = 0
        self._last_frame_id = None
        self._inspected = False
        self._heat = None
        self._frame_id = None
        self._status = NO_PROFILE
        self._message = MESSAGES[NO_PROFILE]
        self._label = None
        self._score = None
        self._latency_ms = None
        self._running = False
        self._thread = None

    def _publish(self, status, message=None):
        """Change the status; any status but RESULT means there is no verdict on screen."""
        with self._lock:
            self._status = status
            self._message = MESSAGES[status] if message is None else message
            self._label = self._score = self._latency_ms = self._heat = None

    def state(self):
        with self._lock:
            profile = self._profile
            return {
                "status": self._status,
                "label": self._label,
                "score": self._score,
                "threshold": profile.threshold if profile else None,
                "latency_ms": self._latency_ms,
                "profile_id": profile.id if profile else None,
                "frame_id": self._frame_id,
                "message": self._message,
            }

    def jpeg(self):
        """The newest camera frame, with the suspect zone colored; None if there is no frame."""
        latest = self._camera.latest()
        if latest is None:
            return None
        frame = latest[0]
        with self._lock:
            heat = self._heat
            threshold = self._profile.threshold if self._profile else None
            roi = self._profile.roi if self._profile else None
        if heat is not None and heat.shape == frame.shape[:2]:
            frame = render_overlay(frame, heat, threshold)
        elif roi:
            frame = frame.copy()  # the camera's own frame is shared and must stay clean
        if roi:
            # The piece is placed by looking at this image, so the zone the model sees is drawn on it.
            x, y, w, h = roi
            thickness = max(2, frame.shape[1] // 400)
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), ZONE_COLOR, thickness)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return encoded.tobytes() if ok else None

    def latest_frame(self):
        latest = self._camera.latest()
        return None if latest is None else latest[0]

    def configure_camera(self, settings):
        self._camera.configure(settings)

    def set_inspector(self, inspector, profile, gate):
        with self._lock:
            self._inspector, self._profile, self._gate = inspector, profile, gate
            self._generation += 1
            self._inspected = False
            self._last_frame_id = None
        self._publish(MOVING)

    def set_background(self, frame):
        """A new photo of the empty table, for when the light has changed."""
        with self._lock:
            profile = self._profile
        if profile is None:
            raise ValueError("no active profile")
        gate = SceneGate(crop_roi(frame, profile.roi))
        with self._lock:
            self._gate = gate
            self._generation += 1
            self._inspected = False
        self._publish(MOVING)

    def _inspect(self, inspector, frame, generation):
        self._publish(MOVING, MESSAGE_ANALYZING)
        try:
            verdict = inspector.inspect(frame)
        except Exception as error:
            log.exception("inspection failed")
            with self._lock:
                current = generation == self._generation
            if current:
                self._publish(ERROR, "Modelul a dat o eroare: {}".format(error))
            return
        with self._lock:
            if generation != self._generation:
                return
            self._status = RESULT
            self._message = MESSAGES[RESULT]
            self._label = verdict.label
            self._score = verdict.score
            self._latency_ms = verdict.latency_ms
            self._heat = verdict.heat

    def _wants_verdict(self, gate, profile, frame):
        """Ask the gate; return True once for each time a piece settles."""
        scene = gate.update(crop_roi(frame, profile.roi))
        if scene != STABLE:
            self._inspected = False
            self._publish(scene)
            return False
        if self._inspected:
            return False
        self._inspected = True
        return True

    def step(self):
        latest = self._camera.latest()
        with self._lock:
            inspector, profile, gate = self._inspector, self._profile, self._gate
            generation = self._generation
        if latest is None:
            if gate is not None:
                gate.reset()
            self._inspected = False
            self._last_frame_id = None
            self._publish(CAMERA_LOST)
            return
        frame, frame_id = latest
        with self._lock:
            self._frame_id = frame_id
        if inspector is None:
            self._publish(NO_PROFILE)
            return
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        if gate is None:
            self._inspect(inspector, frame, generation)
            return
        try:
            wanted = self._wants_verdict(gate, profile, frame)
        except Exception as error:
            log.exception("scene gate failed")
            self._publish(ERROR, "Nu pot citi imaginea: {}".format(error))
            return
        if wanted:
            self._inspect(inspector, frame, generation)

    def _loop(self):
        while self._running:
            try:
                self.step()
            except Exception as error:
                log.exception("inspection step failed")
                self._publish(ERROR, str(error))
            time.sleep(STEP_PAUSE_S)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
