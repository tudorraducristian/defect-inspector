import numpy as np

from defect_inspector.preprocessing import IMAGENET_MEAN, IMAGENET_STD, to_model_input, to_rgb01


def test_model_input_shape_and_range():
    x = to_model_input(np.zeros((480, 640, 3), np.uint8), 256, IMAGENET_MEAN, IMAGENET_STD)
    assert x.shape == (1, 3, 256, 256) and x.dtype == np.float32


def test_bgr_becomes_rgb():
    frame = np.zeros((8, 8, 3), np.uint8)
    frame[..., 2] = 255  # red in BGR
    assert to_rgb01(frame, 8)[0, 0, 0] == 1.0


def test_model_input_applies_mean_and_std():
    frame = np.full((8, 8, 3), 255, np.uint8)
    x = to_model_input(frame, 8, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    assert np.allclose(x, 1.0)
