"""Run the exported model. The library is imported only when it is asked for.

Runs on Python 3.6, on the Jetson Nano.
"""


class OrtBackend:
    def __init__(self, onnx_path, providers):
        import onnxruntime
        self._session = onnxruntime.InferenceSession(str(onnx_path), providers=providers)

    def run(self, x):
        return self._session.run(["map"], {"image": x})[0]


def load_backend(name, onnx_path):
    if name == "ort-cpu":
        return OrtBackend(onnx_path, ["CPUExecutionProvider"])
    raise ValueError("unknown backend {}, available: ort-cpu".format(name))
