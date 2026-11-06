"""Measure how long one ONNX model takes on one inference backend.

Prints one JSON line. A backend that does not work is reported the same way, with an
"error" field and exit code 1, so that a full comparison can go on without it.
Runs on Python 3.6, on the Jetson Nano.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ORT_PROVIDERS = {
    "ort-cpu": ["CPUExecutionProvider"],
    "ort-gpu": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "ort-trt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
}
REQUIRED_PROVIDER = {
    "ort-gpu": "CUDAExecutionProvider",
    "ort-trt": "TensorrtExecutionProvider",
}
BACKENDS = sorted(ORT_PROVIDERS) + ["tensorrt"]
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


class ProviderInactive(Exception):
    """onnxruntime started without the accelerator that was asked for."""


class BackendUnavailable(Exception):
    """The program needed by the backend cannot be started."""


def summarize(times_s, warmup):
    """Turn per-image times in seconds into milliseconds, without the warmup images."""
    measured = [t * 1000.0 for t in times_s[warmup:]]
    if not measured:
        raise ValueError("no measured runs left after {} warmup runs".format(warmup))
    rank = int(math.ceil(0.9 * len(measured))) - 1
    return {
        "runs": len(measured),
        "median_ms": statistics.median(measured),
        "p90_ms": sorted(measured)[rank],
        "first5_ms": statistics.median(measured[:5]),
        "last5_ms": statistics.median(measured[-5:]),
        "times_ms": measured,
    }


def assert_provider_active(session, backend):
    """Fail if onnxruntime fell back to the CPU, which it does without saying so."""
    required = REQUIRED_PROVIDER.get(backend)
    if required is not None and required not in session.get_providers():
        raise ProviderInactive("{} needs {}, but the session has {}".format(
            backend, required, session.get_providers()))


def _read_kb_field(path, field):
    """Return the value of a "Field: 123 kB" line, or None if the file is missing."""
    try:
        with open(str(path)) as source:
            for line in source:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def ram_used_mb(path="/proc/meminfo"):
    """Return system memory in use, as free -m does; the Nano's GPU shares this memory."""
    total = _read_kb_field(path, "MemTotal")
    available = _read_kb_field(path, "MemAvailable")
    if total is None or available is None:
        return None
    return (total - available) / 1024.0


def rss_mb(path="/proc/self/status"):
    """Return this process's resident memory, or None where /proc does not exist."""
    value = _read_kb_field(path, "VmRSS")
    return None if value is None else value / 1024.0


def _random_image(shape):
    return np.random.rand(*[int(d) for d in shape]).astype(np.float32)


def _ort_runner(path, backend):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(path), providers=ORT_PROVIDERS[backend])
    assert_provider_active(session, backend)
    model_input = session.get_inputs()[0]
    image = _random_image(model_input.shape)
    name = model_input.name

    def run():
        session.run(None, {name: image})

    run.providers = session.get_providers()
    run.image = image
    return run


def make_runner(path, backend):
    """Load the model and return a function that runs it once on a fixed random image."""
    if backend in ORT_PROVIDERS:
        return _ort_runner(path, backend)
    raise ValueError("unknown backend {}, choose from {}".format(backend, BACKENDS))


def measure_trtexec(path, runs, warmup, command=None):
    """Measure with trtexec, NVIDIA's TensorRT tool; TensorRT 8.2 has no tensorrt.lite.

    trtexec builds the engine itself, so its first images count as warmup here too.
    Memory is the last reading before it exits, while it is still running the model.
    """
    with tempfile.TemporaryDirectory() as folder:
        times_path = os.path.join(folder, "times.json")
        log_path = os.path.join(folder, "trtexec.log")
        arguments = (command or [TRTEXEC]) + [
            "--onnx={}".format(path),
            "--iterations={}".format(warmup + runs),
            "--duration=0",
            "--warmUp=0",
            "--avgRuns=1",
            "--exportTimes={}".format(times_path),
        ]
        start = time.perf_counter()
        with open(log_path, "w") as log:
            try:
                process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT)
            except OSError as exc:
                raise BackendUnavailable("trtexec cannot be started: {}".format(exc))
            process_rss = used_ram = None
            while process.poll() is None:
                sample = rss_mb("/proc/{}/status".format(process.pid))
                if sample is not None:  # None means that it has just finished.
                    process_rss, used_ram = sample, ram_used_mb()
                time.sleep(0.2)
        wall_s = time.perf_counter() - start
        if process.returncode != 0:
            with open(log_path) as log:
                tail = " | ".join(log.read().splitlines()[-5:])
            raise RuntimeError("trtexec exited with {}: {}".format(process.returncode, tail))
        with open(times_path) as source:
            times_s = [entry["latencyMs"] / 1000.0 for entry in json.load(source)]

    result = {"load_s": wall_s - sum(times_s), "providers": ["trtexec"]}
    result.update(summarize(times_s, warmup))
    result["rss_mb"] = process_rss
    result["ram_used_mb"] = used_ram
    return result


def _measure(path, backend, runs, warmup):
    if backend == "tensorrt":
        return measure_trtexec(path, runs, warmup)
    start = time.perf_counter()
    run = make_runner(path, backend)
    load_s = time.perf_counter() - start

    times_s = []
    for _ in range(warmup + runs):
        start = time.perf_counter()
        run()
        times_s.append(time.perf_counter() - start)

    result = {"load_s": load_s, "providers": run.providers}
    result.update(summarize(times_s, warmup))
    # Read while the model is still loaded, so that its memory is counted.
    result["rss_mb"] = rss_mb()
    result["ram_used_mb"] = ram_used_mb()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--backend", required=True, help="one of {}".format(BACKENDS))
    parser.add_argument("--runs", type=int, default=30, help="measured images, after warmup")
    parser.add_argument("--warmup", type=int, default=5, help="images run before measuring")
    args = parser.parse_args(argv)

    result = {"backend": args.backend, "model": os.path.basename(args.onnx)}
    code = 0
    try:
        result.update(_measure(args.onnx, args.backend, args.runs, args.warmup))
    except Exception as exc:  # Any backend failure is a result to record, not a crash.
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
        code = 1
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    sys.exit(main())
