"""Image preparation shared by training and inference, so the model sees the same input.

Runs on Python 3.6, on the Jetson Nano.
"""
import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_rgb01(frame_bgr, image_size):
    """Resize a BGR frame to a square; return RGB floats in [0, 1], height x width x 3."""
    resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def to_model_input(frame_bgr, image_size, mean, std):
    """Return the normalized batch of one image, with shape (1, 3, size, size)."""
    rgb = to_rgb01(frame_bgr, image_size)
    normalized = (rgb - np.array(mean, np.float32)) / np.array(std, np.float32)
    return normalized.transpose(2, 0, 1)[np.newaxis]
