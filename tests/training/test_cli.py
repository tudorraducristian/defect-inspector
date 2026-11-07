import json

import onnxruntime
import pytest

from defect_train.cli import main, require_gpu


def run_cli(dataset, teacher, out, train_steps, image_size=256, no_autoencoder=False):
    argv = ["--dataset-root", str(dataset), "--object-id", "tiny", "--object-name", "Tiny",
            "--teacher", str(teacher), "--out", str(out), "--train-steps", str(train_steps),
            "--image-size", str(image_size), "--num-workers", "0"]
    if no_autoencoder:
        argv.append("--no-autoencoder")
    main(argv)


def test_cli_end_to_end(tmp_path, tiny_dataset, random_teacher):
    run_cli(tiny_dataset, random_teacher, tmp_path, train_steps=2)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert {"image_auroc", "threshold", "per_defect", "detected_rate", "false_alarm_rate"} <= set(metrics)
    assert (tmp_path / "checkpoint.pt").exists()
    onnxruntime.InferenceSession(str(tmp_path / "profile" / "model.onnx"))


def test_cli_without_autoencoder(tmp_path, tiny_dataset, random_teacher):
    run_cli(tiny_dataset, random_teacher, tmp_path, train_steps=1, image_size=192, no_autoencoder=True)
    assert json.loads((tmp_path / "profile" / "profile.json").read_text())["image_size"] == 192


def test_cli_station_goes_into_the_profile(tmp_path, tiny_dataset, random_teacher):
    (tiny_dataset / "station.json").write_text(json.dumps({"roi": [0, 0, 8, 8], "camera": {"index": 1}}))
    (tiny_dataset / "background.jpg").write_bytes(b"table")
    main(["--dataset-root", str(tiny_dataset), "--object-id", "tiny", "--object-name", "Tiny",
          "--teacher", str(random_teacher), "--out", str(tmp_path), "--train-steps", "1",
          "--num-workers", "0", "--station", str(tiny_dataset / "station.json")])
    profile = json.loads((tmp_path / "profile" / "profile.json").read_text())
    assert profile["roi"] == [0, 0, 8, 8] and profile["camera"] == {"index": 1}
    assert (tmp_path / "profile" / "background.jpg").read_bytes() == b"table"


def test_station_of_the_dataset_is_used_without_the_flag(tmp_path, tiny_dataset, random_teacher):
    # A notebook from before --station would otherwise give a profile with no zone, no camera and no gate.
    (tiny_dataset / "station.json").write_text(json.dumps({"roi": [0, 0, 8, 8], "camera": {"index": 1}}))
    (tiny_dataset / "background.jpg").write_bytes(b"table")
    run_cli(tiny_dataset, random_teacher, tmp_path, train_steps=1)
    profile = json.loads((tmp_path / "profile" / "profile.json").read_text())
    assert profile["roi"] == [0, 0, 8, 8] and (tmp_path / "profile" / "background.jpg").exists()


def test_same_command_resumes(tmp_path, tiny_dataset, random_teacher, capsys):
    run_cli(tiny_dataset, random_teacher, tmp_path, train_steps=2)
    capsys.readouterr()
    run_cli(tiny_dataset, random_teacher, tmp_path, train_steps=2)
    assert "step" not in capsys.readouterr().out


class FakeCuda:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = FakeCuda(cuda)


def test_require_gpu_stops_without_cuda():
    with pytest.raises(SystemExit, match="plac"):
        require_gpu(FakeTorch(cuda=False))


def test_require_gpu_passes_with_cuda():
    require_gpu(FakeTorch(cuda=True))
