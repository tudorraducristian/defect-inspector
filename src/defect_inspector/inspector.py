"""From one camera frame to one verdict.

Runs on Python 3.6, on the Jetson Nano.
"""
import time
from collections import namedtuple

import numpy as np

from defect_inspector.postprocessing import crop_roi, map_to_image, score_from_map
from defect_inspector.preprocessing import to_model_input

Verdict = namedtuple("Verdict", ["label", "score", "threshold", "heat", "latency_ms"])


class Inspector:
    def __init__(self, profile, backend):
        self.profile = profile
        self.backend = backend

    def _heat_on_frame(self, cells, frame_bgr):
        """Scale the map to the zone of interest and place it on a frame-sized map."""
        profile = self.profile
        crop_height, crop_width = crop_roi(frame_bgr, profile.roi).shape[:2]
        crop_heat = map_to_image(cells, crop_width, crop_height, profile.map_padding)
        if profile.roi is None:
            return crop_heat
        x, y = profile.roi[0], profile.roi[1]
        heat = np.zeros(frame_bgr.shape[:2], np.float32)
        heat[y:y + crop_height, x:x + crop_width] = crop_heat
        return heat

    def inspect(self, frame_bgr):
        started = time.perf_counter()
        profile = self.profile
        x = to_model_input(crop_roi(frame_bgr, profile.roi), profile.image_size, profile.mean, profile.std)
        cells = self.backend.run(x)
        score = score_from_map(cells)
        heat = self._heat_on_frame(cells, frame_bgr)
        label = "DEFECT" if score > profile.threshold else "OK"
        latency_ms = (time.perf_counter() - started) * 1000.0
        return Verdict(label, score, profile.threshold, heat, latency_ms)
