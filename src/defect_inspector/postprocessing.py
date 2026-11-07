"""Turn the model's anomaly map into a score, a picture-sized map and a colored overlay.

Runs on Python 3.6, on the Jetson Nano.
"""
import cv2
import numpy as np

OVERLAY_COLOR_BGR = (0, 0, 255)
OVERLAY_OPACITY = 0.5


def score_from_map(heat):
    """The image score is the highest cell of the map, as in training's evaluation."""
    return float(np.max(heat))


def map_to_image(heat, width, height, padding=4):
    """Pad the small map with zeros, as the model loses a border, then scale it to the image."""
    cells = np.squeeze(np.asarray(heat, np.float32))
    padded = np.pad(cells, padding, mode="constant")
    return cv2.resize(padded, (width, height), interpolation=cv2.INTER_LINEAR)


def render_overlay(frame_bgr, heat_image, threshold):
    """Return a copy of the frame with only the cells above the threshold colored."""
    out = frame_bgr.copy()
    suspect = heat_image > threshold
    if suspect.any():
        color = np.array(OVERLAY_COLOR_BGR, np.float32)
        blended = (1.0 - OVERLAY_OPACITY) * out[suspect] + OVERLAY_OPACITY * color
        out[suspect] = blended.astype(np.uint8)
    return out


def crop_roi(frame, roi):
    """Cut the zone of interest, given as [x, y, width, height]; None keeps the frame."""
    if roi is None:
        return frame
    x, y, width, height = roi
    frame_height, frame_width = frame.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("roi {} does not fit in a {}x{} frame".format(roi, frame_width, frame_height))
    return frame[y:y + height, x:x + width]
