"""Decide when there is something to judge: a piece that is present and still.

An empty table or a hand under the camera would look like a defect, because the model has
only ever seen good pieces. Runs on Python 3.6, on the Jetson Nano.

Both tests first remove any change of light over the whole zone (the median difference), so a
cloud or a lamp does not count as a piece or as movement. A piece is present when enough pixels
differ from the table photo; on the station a nut covers about a tenth of the zone, while its mean
difference stays under any threshold that light alone would not cross. A piece is still when it
has stayed close to the frame where it stopped for long enough; comparing with that frame rather
than with the previous one catches slow movement, which is small from one frame to the next.
"""
import time

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


def _difference(a, b):
    """Per-pixel difference, without the change of light over the whole zone."""
    diff = a - b
    return np.abs(diff - np.median(diff))


class SceneGate:
    # Measured on the station photos of 7 November 2026: a nut changes 9-10% of the pixels by more
    # than 20 gray levels; camera noise between two frames gives a mean difference of 0.6-0.7.
    def __init__(self, background, changed_level=20.0, present_fraction=0.03, moving_level=2.0,
                 still_seconds=0.7, now=None):
        self._background = _small_gray(background)
        self._changed_level = changed_level
        self._present_fraction = present_fraction
        self._moving_level = moving_level
        self._still_seconds = still_seconds
        self._now = now or time.monotonic
        self._anchor = None
        self._still_since = None

    def reset(self):
        self._anchor = None
        self._still_since = None

    def update(self, frame):
        small = _small_gray(frame)
        changed = (_difference(small, self._background) > self._changed_level).mean()
        if changed < self._present_fraction:
            self.reset()
            return EMPTY
        if self._anchor is None or _difference(small, self._anchor).mean() >= self._moving_level:
            self._anchor = small
            self._still_since = self._now()
            return MOVING
        if self._now() - self._still_since >= self._still_seconds:
            return STABLE
        return MOVING


def gate_for_profile(profile):
    """Build the gate from the table photo of the profile; a profile without one has no gate."""
    path = profile.path / BACKGROUND_FILE
    if not path.exists():
        return None
    background = cv2.imread(str(path))
    if background is None:
        raise ValueError("cannot read {} of profile {}".format(BACKGROUND_FILE, profile.id))
    return SceneGate(crop_roi(background, profile.roi))
