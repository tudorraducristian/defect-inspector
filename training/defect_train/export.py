"""Export an AnomalyNet to ONNX and check that the exported file matches it."""
import argparse
import copy
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from defect_inspector.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from defect_train.architecture import AnomalyNet, build_random_net

MODEL_SOURCE = "nelson1425/EfficientAD@fcab5146f84ae17597044ad5ddf1656ccf805401"
# Each side of the map is padded by this many cells before it is scaled up to the image.
MAP_PADDING = 4
PROFILE_FIELDS = {"schema_version", "object_id", "object_name", "image_size", "threshold",
                  "mean", "std", "map_padding", "metrics", "source", "created_at", "roi", "camera"}

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


def to_anomaly_net(parts):
    """Combine the trained parts into the network that outputs the normalized map."""
    q = parts.quantiles
    net = AnomalyNet(parts.teacher, parts.student, parts.autoencoder,
                     parts.teacher_mean, parts.teacher_std,
                     q["st_start"], q["st_end"], q["ae_start"], q["ae_end"])
    return net.eval()


def write_profile(net, out_dir, object_id, object_name, image_size, metrics, images, station=None):
    """Export, verify, and only then put model.onnx and profile.json in out_dir.

    The work happens in a neighboring folder, so a failed export leaves no new profile
    and keeps an older one intact.
    """
    out_dir = Path(out_dir)
    staging = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    station = station or {}
    try:
        cpu_net = copy.deepcopy(net).cpu().eval()
        export_onnx(cpu_net, staging / "model.onnx", image_size)
        verify_onnx(cpu_net, staging / "model.onnx", images)
        profile = {
            "schema_version": 1,
            "object_id": object_id,
            "object_name": object_name,
            "image_size": image_size,
            "threshold": metrics["threshold"],
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "map_padding": MAP_PADDING,
            "metrics": metrics,
            "source": MODEL_SOURCE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "roi": station.get("roi"),
            "camera": station.get("camera"),
        }
        (staging / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(staging, out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, help="folder for the probe models")
    args = parser.parse_args()
    for path in export_probe_models(args.probe):
        print(path)


if __name__ == "__main__":
    main()
