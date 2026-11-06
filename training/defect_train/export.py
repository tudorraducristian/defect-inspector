"""Export an AnomalyNet to ONNX and check that the exported file matches it."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from defect_train.architecture import build_random_net

PROBE_MODELS = (
    ("probe_full_256.onnx", 256, True),
    ("probe_st_256.onnx", 256, False),
    ("probe_st_192.onnx", 192, False),
    ("probe_st_160.onnx", 160, False),
)


class ExportMismatch(Exception):
    """The ONNX model does not produce the same output as the PyTorch model."""


def export_onnx(net, path, image_size, opset=11):
    net.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(net, (dummy,), str(path), opset_version=opset, dynamo=False,
                      input_names=["image"], output_names=["map"])


def verify_onnx(net, path, images, tol=1e-4):
    """Return the largest absolute difference; raise ExportMismatch above tol."""
    net.eval()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for image in images:
        batch = image[np.newaxis]
        with torch.no_grad():
            expected = net(torch.from_numpy(batch)).numpy()
        actual = session.run(["map"], {"image": batch})[0]
        worst = max(worst, float(np.abs(expected - actual).max()))
    if worst > tol:
        raise ExportMismatch("largest difference {} is above {}".format(worst, tol))
    return worst


def export_probe_models(folder):
    """Write the untrained models used by the speed probe; return their paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, image_size, use_autoencoder in PROBE_MODELS:
        path = folder / name
        export_onnx(build_random_net(image_size, use_autoencoder), path, image_size)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="folder for the probe models")
    args = parser.parse_args()
    for path in export_probe_models(args.probe):
        print(path)


if __name__ == "__main__":
    main()
