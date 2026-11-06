# defect-inspector

A visual quality-inspection station for the NVIDIA Jetson Nano. The model learns
only from good parts and flags what looks different, using EfficientAD (small).

Work in progress. Results will be added here once they have been measured.

## Layout

- `training/`: training code and the Colab notebook. Never runs on the Nano.
- `src/defect_inspector/`: code shared with the Nano. Python 3.6 compatible.
- `tools/`: command-line tools. Python 3.6 compatible.
- `tests/`: automated tests, run with `python -m pytest`.

## Training

Open `training/train.ipynb` in Google Colab, with a GPU, and press `Run all`. The dataset
archive goes in Google Drive, in `defect-inspector/datasets/`. A stopped session continues
from its last checkpoint when you press `Run all` again.

## Data and third-party code

MVTec AD is not included in this repository. It is licensed under CC BY-NC-SA 4.0, for
non-commercial use; get it from
[the MVTec site](https://www.mvtec.com/company/research/datasets/mvtec-ad).

The code is under the MIT License. See `NOTICE` for the third-party code it includes.
