from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from defect_train.architecture import build_random_net
from defect_train.export import (
    ExportMismatch,
    export_onnx,
    export_probe_models,
    verify_onnx,
)


def test_output_shape_full_256():
    assert build_random_net(256)(torch.zeros(1, 3, 256, 256)).shape == (1, 1, 56, 56)


@pytest.mark.parametrize("size, side", [(256, 56), (192, 40), (160, 32)])
def test_output_shape_without_autoencoder(size, side):
    net = build_random_net(size, use_autoencoder=False)
    assert net(torch.zeros(1, 3, size, size)).shape == (1, 1, side, side)


def test_probe_writes_all_models(tmp_path):
    names = {Path(p).name for p in export_probe_models(tmp_path)}
    assert names == {"probe_full_256.onnx", "probe_st_256.onnx",
                     "probe_st_192.onnx", "probe_st_160.onnx"}


def test_export_opset_ir_and_no_dropout(tmp_path):
    path = tmp_path / "m.onnx"
    export_onnx(build_random_net(256), path, 256)
    model = onnx.load(path)
    assert model.opset_import[0].version == 11
    assert model.ir_version <= 7
    assert "Dropout" not in {n.op_type for n in model.graph.node}


# An untrained map is around 1e-4, so two unrelated untrained models differ by about
# 1e-5. These two tests need a tolerance far below that to mean anything.
UNTRAINED_TOL = 1e-8


def test_onnx_matches_torch(tmp_path):
    net, path = build_random_net(256), tmp_path / "m.onnx"
    export_onnx(net, path, 256)
    images = np.random.rand(2, 3, 256, 256).astype(np.float32)
    assert verify_onnx(net, path, images, tol=UNTRAINED_TOL) < UNTRAINED_TOL


def test_verify_raises_when_models_differ(tmp_path):
    path = tmp_path / "m.onnx"
    export_onnx(build_random_net(256), path, 256)
    images = np.random.rand(1, 3, 256, 256).astype(np.float32)
    with pytest.raises(ExportMismatch):
        verify_onnx(build_random_net(256), path, images, tol=UNTRAINED_TOL)


def test_autoencoder_rejects_other_image_sizes():
    with pytest.raises(ValueError, match="256"):
        build_random_net(192, use_autoencoder=True)
