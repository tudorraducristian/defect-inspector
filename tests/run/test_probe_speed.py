import json
import sys
import types

import numpy as np
import pytest

from probe_speed import (
    BackendUnavailable,
    ProviderInactive,
    assert_provider_active,
    main,
    make_runner,
    measure_trtexec,
    ram_used_mb,
    rss_mb,
    summarize,
)


def test_summarize_excludes_warmup():
    r = summarize([9.0, 9.0, 0.1, 0.1, 0.2], warmup=2)
    assert r["runs"] == 3 and r["median_ms"] == 100.0


def test_summarize_keeps_order_and_compares_ends():
    r = summarize([9.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], warmup=1)
    assert r["times_ms"] == pytest.approx([100, 200, 300, 400, 500, 600])
    assert r["first5_ms"] == pytest.approx(300.0)
    assert r["last5_ms"] == pytest.approx(400.0)


def test_summarize_p90_uses_nearest_rank():
    times = [i / 1000.0 for i in range(1, 11)]
    assert summarize(times, warmup=0)["p90_ms"] == pytest.approx(9.0)


def test_summarize_rejects_no_measured_runs():
    with pytest.raises(ValueError):
        summarize([0.1, 0.2], warmup=2)


def test_ram_used_reads_meminfo(tmp_path):
    f = tmp_path / "meminfo"
    f.write_text("MemTotal: 4000000 kB\nMemFree: 1 kB\nMemAvailable: 1000000 kB\n")
    assert ram_used_mb(f) == pytest.approx(2929.7, abs=0.1)


def test_ram_used_is_none_without_meminfo(tmp_path):
    assert ram_used_mb(tmp_path / "missing") is None


def test_rss_reads_status(tmp_path):
    f = tmp_path / "status"
    f.write_text("Name: python\nVmRSS:\t  204800 kB\n")
    assert rss_mb(f) == pytest.approx(200.0)


def test_rss_is_none_without_status(tmp_path):
    assert rss_mb(tmp_path / "missing") is None


class FakeSession:
    def __init__(self, providers):
        self._p = providers

    def get_providers(self):
        return self._p


def test_gpu_backend_rejects_silent_cpu_fallback():
    with pytest.raises(ProviderInactive):
        assert_provider_active(FakeSession(["CPUExecutionProvider"]), "ort-gpu")


def test_gpu_backend_accepts_cuda():
    assert_provider_active(FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"]), "ort-gpu")


def test_trt_backend_rejects_cuda_only():
    with pytest.raises(ProviderInactive):
        assert_provider_active(FakeSession(["CUDAExecutionProvider", "CPUExecutionProvider"]), "ort-trt")


def test_trt_backend_accepts_tensorrt():
    assert_provider_active(FakeSession(["TensorrtExecutionProvider", "CPUExecutionProvider"]), "ort-trt")


def test_cpu_backend_accepts_cpu():
    assert_provider_active(FakeSession(["CPUExecutionProvider"]), "ort-cpu")


def fake_onnxruntime(active_providers, requested):
    class Input:
        name = "image"
        shape = [1, 3, 8, 8]

    class Session(FakeSession):
        def __init__(self, path, providers):
            requested.extend(providers)
            super().__init__(active_providers)

        def get_inputs(self):
            return [Input()]

        def run(self, outputs, feed):
            self.feed = feed
            return [None]

    module = types.ModuleType("onnxruntime")
    module.InferenceSession = Session
    return module


def test_make_runner_fails_when_gpu_falls_back_to_cpu(monkeypatch):
    requested = []
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CPUExecutionProvider"], requested))
    with pytest.raises(ProviderInactive):
        make_runner("m.onnx", "ort-gpu")
    assert requested == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_make_runner_requests_tensorrt_first_for_ort_trt(monkeypatch):
    requested = []
    active = ["TensorrtExecutionProvider", "CPUExecutionProvider"]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(active, requested))
    run = make_runner("m.onnx", "ort-trt")
    run()
    assert requested[0] == "TensorrtExecutionProvider"
    assert run.providers == active


def test_make_runner_rejects_unknown_backend():
    with pytest.raises(ValueError, match="ort-cpu"):
        make_runner("m.onnx", "nope")


FAKE_TRTEXEC = """
import json, sys, time
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if "=" in a)
if "fail" in args["onnx"]:
    print("Error: cannot parse the model")
    sys.exit(1)
if "slow" in args["onnx"]:
    time.sleep(0.7)
count = int(args["iterations"])
times = [{"latencyMs": 10.0 + i} for i in range(count)]
with open(args["exportTimes"], "w") as out:
    json.dump(times, out)
"""


@pytest.fixture
def fake_trtexec(tmp_path):
    script = tmp_path / "fake_trtexec.py"
    script.write_text(FAKE_TRTEXEC)
    return [sys.executable, str(script)]


def test_trtexec_times_exclude_warmup_and_keep_order(fake_trtexec):
    result = measure_trtexec("m.onnx", runs=3, warmup=2, command=fake_trtexec)
    assert result["runs"] == 3
    assert result["times_ms"] == pytest.approx([12.0, 13.0, 14.0])
    assert result["providers"] == ["trtexec"]
    assert result["load_s"] >= 0


def test_trtexec_memory_ignores_the_reading_after_the_process_ended(fake_trtexec, monkeypatch):
    rss = iter([100.0])
    ram = iter([2000.0])
    # A finished process has no VmRSS line, so its reading comes back as None.
    monkeypatch.setattr("probe_speed.rss_mb", lambda path: next(rss, None))
    monkeypatch.setattr("probe_speed.ram_used_mb", lambda: next(ram, 500.0))
    result = measure_trtexec("slow.onnx", runs=3, warmup=2, command=fake_trtexec)
    assert result["rss_mb"] == 100.0
    assert result["ram_used_mb"] == 2000.0


def test_trtexec_failure_keeps_its_message(fake_trtexec):
    with pytest.raises(RuntimeError, match="cannot parse"):
        measure_trtexec("fail.onnx", runs=3, warmup=2, command=fake_trtexec)


def test_trtexec_missing_gives_clear_error(tmp_path):
    with pytest.raises(BackendUnavailable, match="trtexec"):
        measure_trtexec("m.onnx", runs=3, warmup=2, command=[str(tmp_path / "trtexec")])


def test_main_writes_one_json_line(monkeypatch, capsys):
    def fake_make_runner(path, backend):
        def run():
            pass
        run.providers = ["fake"]
        return run

    monkeypatch.setattr("probe_speed.make_runner", fake_make_runner)
    code = main(["--onnx", "folder/m.onnx", "--backend", "ort-cpu", "--runs", "6", "--warmup", "2"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["backend"] == "ort-cpu" and result["model"] == "m.onnx"
    assert result["runs"] == 6 and len(result["times_ms"]) == 6
    assert result["providers"] == ["fake"]
    assert set(result) >= {"load_s", "median_ms", "p90_ms", "first5_ms", "last5_ms",
                           "rss_mb", "ram_used_mb"}


def test_main_reports_a_failing_backend_as_data(monkeypatch, capsys):
    monkeypatch.setattr("probe_speed.TRTEXEC", "missing-trtexec-binary")
    code = main(["--onnx", "m.onnx", "--backend", "tensorrt"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["backend"] == "tensorrt" and result["model"] == "m.onnx"
    assert "trtexec" in result["error"]


def test_real_numpy_is_used_for_the_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime(["CPUExecutionProvider"], []))
    run = make_runner("m.onnx", "ort-cpu")
    assert run.image.dtype == np.float32 and run.image.shape == (1, 3, 8, 8)
