import cv2
import numpy as np
import pytest

from defect_inspector.camera import CameraWorker, ReplayCamera
from run_server import build_camera, parse_args


def test_default_stays_on_this_machine():
    assert parse_args([]).host == "127.0.0.1"


def test_camera_and_replay_exclude_each_other():
    with pytest.raises(SystemExit):
        parse_args(["--camera", "1", "--replay", "folder"])


def test_default_source_is_the_camera():
    assert isinstance(build_camera(parse_args(["--camera", "1"])), CameraWorker)


def test_replay_folder_becomes_the_source(tmp_path):
    cv2.imwrite(str(tmp_path / "0.png"), np.zeros((4, 4, 3), np.uint8))
    assert isinstance(build_camera(parse_args(["--replay", str(tmp_path)])), ReplayCamera)
