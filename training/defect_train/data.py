"""Dataset loading. The same folder format serves MVTec AD and own parts:
train/good/, optional val/good/, test/good/, test/<defect>/.
"""
from pathlib import Path
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from defect_inspector.preprocessing import IMAGENET_MEAN, IMAGENET_STD, to_model_input, to_rgb01

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# OpenCV's own threads can deadlock inside forked DataLoader workers.
cv2.setNumThreads(0)


class DatasetError(Exception):
    """The dataset is missing something, or a file in it cannot be read."""


class Splits(NamedTuple):
    train: List[Path]
    val: List[Path]
    test: List[Tuple[Path, str]]


def list_images(folder) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
                  key=lambda p: p.name.lower())


def _split_train_val(good, val_fraction, seed):
    """Hold out a fixed part of the good images, chosen the same way on every run."""
    order = np.random.RandomState(seed).permutation(len(good))
    train_size = int((1 - val_fraction) * len(good))
    if train_size == 0 or train_size == len(good):
        raise DatasetError("need more good images in train/good to hold out a validation part")
    return ([good[i] for i in sorted(order[:train_size])],
            [good[i] for i in sorted(order[train_size:])])


def load_splits(root, val_fraction=0.1, seed=42) -> Splits:
    root = Path(root)
    good = list_images(root / "train" / "good")
    if not good:
        raise DatasetError("no images in {}".format(root / "train" / "good"))

    val = list_images(root / "val" / "good")
    train = good
    if not val:
        train, val = _split_train_val(good, val_fraction, seed)

    test_root = root / "test"
    kinds = sorted(p.name for p in test_root.iterdir() if p.is_dir()) if test_root.is_dir() else []
    test = [(path, kind) for kind in kinds for path in list_images(test_root / kind)]
    if not any(kind == "good" for _, kind in test):
        raise DatasetError("no good test images in {}".format(test_root / "good"))
    if not any(kind != "good" for _, kind in test):
        raise DatasetError("no defect images in {}, need test/<defect>/".format(test_root))
    return Splits(train, val, test)


def _read(path):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise DatasetError("cannot read image {}".format(path))
    return frame


_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
_jitter = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2),
])


def _to_tensor(rgb01):
    return torch.from_numpy(rgb01.transpose(2, 0, 1))


class TrainImages(Dataset):
    """Return (clean, augmented); like train_transform in EfficientAD, see NOTICE."""

    def __init__(self, paths, image_size):
        self.paths = list(paths)
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        frame = _read(self.paths[index])
        clean = to_model_input(frame, self.image_size, IMAGENET_MEAN, IMAGENET_STD)[0]
        rgb = _to_tensor(to_rgb01(frame, self.image_size))
        return torch.from_numpy(clean), _normalize(_jitter(rgb))


class PenaltyImages(Dataset):
    """Unrelated images for the student's penalty; like penalty_transform in EfficientAD."""

    def __init__(self, root, image_size):
        self.paths = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise DatasetError("no penalty images in {}".format(root))
        self.image_size = image_size
        self._gray = transforms.RandomGrayscale(0.3)
        self._crop = transforms.CenterCrop(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rgb = _to_tensor(to_rgb01(_read(self.paths[index]), 2 * self.image_size))
        return _normalize(self._crop(self._gray(rgb)))
