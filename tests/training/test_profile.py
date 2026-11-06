import json

import numpy as np
import pytest
import torch

from defect_train import export
from defect_train.architecture import build_random_net
from defect_train.export import (
    PROFILE_FIELDS,
    ExportMismatch,
    to_anomaly_net,
    verify_onnx,
    write_profile,
)
from defect_train.train_loop import TrainedParts

METRICS = {"threshold": 0.5, "image_auroc": 0.9}


def raise_export_mismatch(*args, **kwargs):
    raise ExportMismatch("differs")


@pytest.fixture
def net():
    return build_random_net(256)


@pytest.fixture
def images():
    return np.random.rand(1, 3, 256, 256).astype(np.float32)


@pytest.fixture
def written_profile(tmp_path, net, images):
    return write_profile(net, tmp_path / "p", "x", "X", 256, METRICS, images)


def test_profile_json_has_every_field(written_profile):
    data = json.loads((written_profile / "profile.json").read_text())
    assert set(data) == PROFILE_FIELDS and data["schema_version"] == 1
    assert data["threshold"] == 0.5 and data["map_padding"] == 4
    assert data["roi"] is None and data["camera"] is None


def test_exported_model_equals_original(written_profile, net, images):
    assert verify_onnx(net, written_profile / "model.onnx", images) < 1e-4


def test_mismatch_leaves_no_profile(tmp_path, monkeypatch, net, images):
    monkeypatch.setattr(export, "verify_onnx", raise_export_mismatch)
    with pytest.raises(ExportMismatch):
        write_profile(net, tmp_path / "p", "x", "X", 256, METRICS, images)
    assert not (tmp_path / "p" / "profile.json").exists()
    assert not (tmp_path / "p" / "model.onnx").exists()


def test_failed_export_keeps_the_previous_profile(tmp_path, monkeypatch, net, images):
    write_profile(net, tmp_path / "p", "old", "Old", 256, METRICS, images)
    monkeypatch.setattr(export, "verify_onnx", raise_export_mismatch)
    with pytest.raises(ExportMismatch):
        write_profile(net, tmp_path / "p", "new", "New", 256, METRICS, images)
    assert json.loads((tmp_path / "p" / "profile.json").read_text())["object_id"] == "old"


def test_station_supplies_roi_and_camera(tmp_path, net, images):
    station = {"roi": [0, 0, 10, 10], "camera": {"exposure": 100}}
    path = write_profile(net, tmp_path / "p", "x", "X", 256, METRICS, images, station=station)
    data = json.loads((path / "profile.json").read_text())
    assert data["roi"] == [0, 0, 10, 10] and data["camera"] == {"exposure": 100}


def test_to_anomaly_net_uses_the_trained_normalization():
    base = build_random_net(256)
    parts = TrainedParts(base.teacher, base.student, base.autoencoder,
                         torch.full((1, 384, 1, 1), 2.0), torch.full((1, 384, 1, 1), 3.0),
                         {"st_start": 0.1, "st_end": 0.2, "ae_start": 0.3, "ae_end": 0.4})
    net = to_anomaly_net(parts)
    assert net.teacher_mean.flatten()[0] == 2.0 and net.teacher_std.flatten()[0] == 3.0
    assert net.q_st_end.item() == pytest.approx(0.2) and net.q_ae_start.item() == pytest.approx(0.3)
    assert not net.training
