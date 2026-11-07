import json

import numpy as np
import pytest

from defect_inspector.backends import load_backend
from defect_inspector.inspector import Inspector
from defect_inspector.postprocessing import crop_roi, map_to_image, render_overlay, score_from_map
from defect_inspector.preprocessing import to_model_input
from defect_inspector.profile import ProfileError, list_profiles, load_profile

FRAME = np.full((64, 64, 3), 120, np.uint8)
GOOD_FIELDS = {
    "schema_version": 1, "object_id": "good-one", "object_name": "Good one", "image_size": 32,
    "threshold": 0.5, "mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2], "map_padding": 4,
    "metrics": {"image_auroc": 0.9}, "roi": None, "camera": None,
}


class FakeBackend:
    def __init__(self, peak):
        self.peak = peak

    def run(self, x):
        return np.full((1, 1, 56, 56), self.peak, np.float32)


class RecordingBackend(FakeBackend):
    def __init__(self):
        super().__init__(0.0)
        self.last_input = None

    def run(self, x):
        self.last_input = x
        return super().run(x)


def write_profile_folder(folder, **changes):
    folder.mkdir(parents=True)
    (folder / "profile.json").write_text(json.dumps(dict(GOOD_FIELDS, **changes)), encoding="utf-8")
    (folder / "model.onnx").write_bytes(b"model")
    return folder


@pytest.fixture
def profile(tmp_path):
    return load_profile(write_profile_folder(tmp_path / "good-one"))


@pytest.fixture
def profiles_root_with_one_broken(tmp_path):
    write_profile_folder(tmp_path / "good-one")
    broken = tmp_path / "broken-one"
    broken.mkdir()
    (broken / "profile.json").write_text("{not json", encoding="utf-8")
    return tmp_path


@pytest.fixture
def profile_folder_schema_2(tmp_path):
    return write_profile_folder(tmp_path / "future", schema_version=2)


def test_verdict_above_threshold_is_defect(profile):
    v = Inspector(profile._replace(threshold=0.5), FakeBackend(peak=0.9)).inspect(FRAME)
    assert v.label == "DEFECT" and v.score == pytest.approx(0.9)


def test_verdict_equal_to_threshold_is_ok(profile):
    assert Inspector(profile._replace(threshold=0.25), FakeBackend(peak=0.25)).inspect(FRAME).label == "OK"


def test_verdict_reports_threshold_and_latency(profile):
    v = Inspector(profile, FakeBackend(peak=0.1)).inspect(FRAME)
    assert v.threshold == profile.threshold and v.latency_ms >= 0.0


def test_score_is_the_highest_cell_of_the_map():
    heat = np.zeros((1, 1, 56, 56))
    heat[0, 0, 7, 9] = 0.75
    assert score_from_map(heat) == pytest.approx(0.75)


def test_overlay_leaves_good_piece_unchanged():
    assert np.array_equal(render_overlay(FRAME, np.zeros(FRAME.shape[:2]), 0.5), FRAME)


def test_overlay_colors_only_the_suspect_zone():
    heat = np.zeros(FRAME.shape[:2])
    heat[10:20, 10:20] = 1.0
    out = render_overlay(FRAME, heat, 0.5)
    assert not np.array_equal(out[10:20, 10:20], FRAME[10:20, 10:20])
    assert np.array_equal(out[40:, 40:], FRAME[40:, 40:])


def test_overlay_does_not_change_the_frame_it_receives():
    frame = FRAME.copy()
    render_overlay(frame, np.ones(frame.shape[:2]), 0.5)
    assert np.array_equal(frame, FRAME)


def test_map_is_padded_then_resized():
    image = map_to_image(np.ones((1, 1, 56, 56)), 256, 256)
    assert image.shape == (256, 256)
    assert image[0, 0] == 0.0 and image[128, 128] == pytest.approx(1.0)


def test_crop_roi_is_x_y_width_height():
    frame = np.zeros((20, 40, 3), np.uint8)
    assert crop_roi(frame, [10, 5, 8, 4]).shape == (4, 8, 3)


def test_crop_without_roi_keeps_the_frame():
    assert crop_roi(FRAME, None) is FRAME


def test_roi_outside_the_frame_is_refused():
    with pytest.raises(ValueError, match="roi"):
        crop_roi(FRAME, [60, 60, 10, 10])


def test_load_profile_reads_every_field(tmp_path):
    folder = write_profile_folder(tmp_path / "good-one", roi=[1, 2, 3, 4], camera={"index": 0})
    p = load_profile(folder)
    assert (p.id, p.name, p.path, p.image_size, p.threshold) == ("good-one", "Good one", folder, 32, 0.5)
    assert (p.mean, p.std, p.map_padding) == ([0.4] * 3, [0.2] * 3, 4)
    assert (p.roi, p.camera, p.metrics) == ([1, 2, 3, 4], {"index": 0}, {"image_auroc": 0.9})


def test_broken_profile_does_not_hide_good_ones(profiles_root_with_one_broken):
    profiles, errors = list_profiles(profiles_root_with_one_broken)
    assert [p.id for p in profiles] == ["good-one"] and "broken-one" in errors[0]


def test_empty_folder_is_reported_by_name(tmp_path):
    write_profile_folder(tmp_path / "good-one")
    (tmp_path / "empty-one").mkdir()
    profiles, errors = list_profiles(tmp_path)
    assert [p.id for p in profiles] == ["good-one"] and "empty-one" in errors[0]


def test_missing_model_is_refused(tmp_path):
    folder = write_profile_folder(tmp_path / "no-model")
    (folder / "model.onnx").unlink()
    with pytest.raises(ProfileError, match="model.onnx"):
        load_profile(folder)


def test_missing_field_is_named(tmp_path):
    folder = write_profile_folder(tmp_path / "no-threshold")
    fields = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    del fields["threshold"]
    (folder / "profile.json").write_text(json.dumps(fields), encoding="utf-8")
    with pytest.raises(ProfileError, match="threshold"):
        load_profile(folder)


def test_two_folders_with_the_same_id_keep_the_first(tmp_path):
    write_profile_folder(tmp_path / "a-copy")
    write_profile_folder(tmp_path / "b-copy")
    profiles, errors = list_profiles(tmp_path)
    assert [p.path.name for p in profiles] == ["a-copy"] and "b-copy" in errors[0]


def test_missing_profiles_folder_is_an_error_not_a_crash(tmp_path):
    profiles, errors = list_profiles(tmp_path / "nothing-here")
    assert profiles == [] and "nothing-here" in errors[0]


def test_newer_schema_is_refused(profile_folder_schema_2):
    with pytest.raises(ProfileError, match="schema"):
        load_profile(profile_folder_schema_2)


def test_inspector_applies_roi(profile):
    backend = RecordingBackend()
    p = profile._replace(roi=[0, 0, 10, 10])
    Inspector(p, backend).inspect(FRAME)
    expected = to_model_input(FRAME[:10, :10], p.image_size, p.mean, p.std)
    assert np.array_equal(backend.last_input, expected)


def test_heat_has_frame_size_and_stays_inside_roi(profile):
    p = profile._replace(roi=[8, 8, 32, 32])
    heat = Inspector(p, FakeBackend(peak=1.0)).inspect(FRAME).heat
    assert heat.shape == FRAME.shape[:2]
    assert heat[24, 24] == pytest.approx(1.0) and heat[50, 50] == 0.0


def test_unknown_backend_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nope"):
        load_backend("nope", tmp_path / "model.onnx")
