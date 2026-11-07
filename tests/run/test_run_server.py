import cv2
import numpy as np
import pytest

from defect_inspector.camera import CameraSettings, CameraWorker, ReplayCamera
from run_server import build_camera, parse_args


def test_default_stays_on_this_machine():
    assert parse_args([]).host == "127.0.0.1"


def test_camera_and_replay_exclude_each_other():
    with pytest.raises(SystemExit):
        parse_args(["--camera", "1", "--replay", "folder"])


def test_default_source_is_the_camera():
    assert isinstance(build_camera(parse_args(["--camera", "1"])), CameraWorker)


def test_camera_chosen_by_hand_stays_when_an_object_is_chosen():
    camera = build_camera(parse_args(["--camera", "1"]))
    camera.configure(CameraSettings(3, 640, 480, 30, None, None))
    assert camera._settings.index == 1 and camera._settings.width == 640


def test_without_a_choice_the_object_says_which_camera_took_its_photos():
    camera = build_camera(parse_args([]))
    assert camera._settings.index == 0
    camera.configure(CameraSettings(3, 640, 480, 30, None, None))
    assert camera._settings.index == 3


def test_replay_folder_becomes_the_source(tmp_path):
    cv2.imwrite(str(tmp_path / "0.png"), np.zeros((4, 4, 3), np.uint8))
    assert isinstance(build_camera(parse_args(["--replay", str(tmp_path)])), ReplayCamera)
