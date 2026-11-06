from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from defect_train import evaluate as evaluate_module
from defect_train.data import Splits
from defect_train.evaluate import (
    evaluate,
    image_auroc,
    propose_threshold,
    rates_at_threshold,
    score_paths,
)


class FakeScorer:
    """Stands in for a network: the score of an image is written in its file name."""

    def __init__(self):
        self.test_offset = 0.0

    def raise_test_scores(self, by):
        self.test_offset = by

    def score(self, paths):
        return np.array([float(p.stem) + (self.test_offset if p.parts[0] == "test" else 0.0)
                         for p in paths])


@pytest.fixture
def fake_net_scoring_by_path(monkeypatch):
    monkeypatch.setattr(evaluate_module, "score_paths",
                        lambda net, paths, image_size, device: net.score(paths))
    return FakeScorer()


@pytest.fixture
def splits():
    return Splits(
        train=[Path("train/0.1.png")],
        val=[Path("val/0.2.png"), Path("val/0.4.png")],
        test=[(Path("test/0.1.png"), "good"), (Path("test/0.3.png"), "good"),
              (Path("test/0.5.png"), "scratch"), (Path("test/0.9.png"), "bent")],
    )


def test_auroc_perfect_separation():
    assert image_auroc([0.1, 0.2], [0.8, 0.9]) == 1.0


def test_rates_at_threshold():
    r = rates_at_threshold([0.1, 0.6], [0.5, 0.9, 0.7, 0.4], 0.45)
    assert r == {"detected_rate": 0.75, "false_alarm_rate": 0.5}


def test_score_equal_to_threshold_is_not_defect():
    assert rates_at_threshold([0.5], [0.5], 0.5) == {"detected_rate": 0.0, "false_alarm_rate": 0.0}


def test_threshold_is_above_every_validation_score_even_negative():
    assert propose_threshold([-0.3, -0.1]) > -0.1
    assert propose_threshold([0.2, 0.4]) == pytest.approx(0.44)


def test_threshold_ignores_test_scores(fake_net_scoring_by_path, splits):
    a = evaluate(fake_net_scoring_by_path, splits, 256, "cpu")["threshold"]
    fake_net_scoring_by_path.raise_test_scores(by=100.0)
    assert evaluate(fake_net_scoring_by_path, splits, 256, "cpu")["threshold"] == a


def test_evaluate_reports_every_metric(fake_net_scoring_by_path, splits):
    metrics = evaluate(fake_net_scoring_by_path, splits, 256, "cpu")
    assert metrics["image_auroc"] == 1.0
    assert metrics["threshold"] == pytest.approx(0.44)
    assert metrics["per_defect"] == {"bent": 1.0, "scratch": 1.0}
    assert metrics["detected_rate"] == 1.0 and metrics["false_alarm_rate"] == 0.0
    assert metrics["counts"] == {"train": 1, "val": 2, "test_good": 2, "test_defect": 2}


class BrightnessNet(torch.nn.Module):
    """Its map is the image itself, so the score is the brightest normalized pixel."""

    def forward(self, image):
        return image[:, :1]


def test_score_paths_is_the_maximum_of_the_map(tmp_path):
    dark, bright = tmp_path / "dark.png", tmp_path / "bright.png"
    cv2.imwrite(str(dark), np.zeros((16, 16, 3), np.uint8))
    image = np.zeros((16, 16, 3), np.uint8)
    image[0, 0] = 255
    image[:8] = 40
    cv2.imwrite(str(bright), image)
    scores = score_paths(BrightnessNet(), [dark, bright], 16, "cpu")
    assert scores.shape == (2,) and scores[1] > scores[0]
