# defect-inspector

A visual quality-inspection station for the NVIDIA Jetson Nano. The model learns
only from good parts and flags what looks different, using EfficientAD (small).

Work in progress. Results will be added here once they have been measured.

## Layout

- `training/defect_train/`: model architecture and ONNX export. Never runs on the Nano.
- `tools/`: command-line tools. Python 3.6 compatible, because they run on the Nano.
- `tests/`: automated tests, run with `python -m pytest`.

## Third-party code

See `NOTICE`.
