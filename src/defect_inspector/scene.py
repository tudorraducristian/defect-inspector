"""Decide when there is something to judge: a piece that is present and still.

An empty table or a hand under the camera would look like a defect, because the model has
only ever seen good pieces. Runs on Python 3.6, on the Jetson Nano.
"""
import cv2
import numpy as np

from defect_inspector.postprocessing import crop_roi
from defect_inspector.profile import BACKGROUND_FILE

EMPTY = "EMPTY"
MOVING = "MOVING"
STABLE = "STABLE"
COMPARE_SIZE = (64, 64)


def _small_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, COMPARE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


class SceneGate:
    def __init__(self, background, present_threshold=12.0, moving_threshold=6.0, stable_frames=5):
        self._background = _small_gray(background)
        self._present_threshold = present_threshold
        self._moving_threshold = moving_threshold
        self._stable_frames = stable_frames
        self._previous = None
        self._still = 0

    def reset(self):
        self._previous = None
        self._still = 0

    def update(self, frame):
        small = _small_gray(frame)
        previous, self._previous = self._previous, small
        if np.abs(small - self._background).mean() < self._present_threshold:
            self._still = 0
            return EMPTY
        if previous is None or np.abs(small - previous).mean() >= self._moving_threshold:
            self._still = 0
            return MOVING
        self._still += 1
        return STABLE if self._still >= self._stable_frames else MOVING


def gate_for_profile(profile):
    """Build the gate from the table photo of the profile; a profile without one has no gate."""
    path = profile.path / BACKGROUND_FILE
    if not path.exists():
        return None
    background = cv2.imread(str(path))
    if background is None:
        raise ValueError("cannot read {} of profile {}".format(BACKGROUND_FILE, profile.id))
    return SceneGate(crop_roi(background, profile.roi))
