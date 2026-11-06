"""Small synthetic datasets, in the folder format used for MVTec AD and for own parts."""
import cv2
import numpy as np
import pytest


def write_images(folder, count, seed=0, size=64):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    for index in range(count):
        image = rng.randint(0, 256, (size, size, 3), dtype=np.uint8)
        cv2.imwrite(str(folder / "{:03d}.png".format(index)), image)


def make_dataset(root, seed=0):
    write_images(root / "train" / "good", 10, seed)
    write_images(root / "test" / "good", 3, seed + 1)
    write_images(root / "test" / "scratch", 2, seed + 2)
    write_images(root / "test" / "bent", 2, seed + 3)
    write_images(root / "ground_truth" / "scratch", 2, seed + 4)
    return root


@pytest.fixture
def tiny_dataset(tmp_path):
    return make_dataset(tmp_path / "dataset")


@pytest.fixture
def other_dataset(tmp_path):
    root = make_dataset(tmp_path / "other", seed=50)
    cv2.imwrite(str(root / "train" / "good" / "added.png"), np.zeros((64, 64, 3), np.uint8))
    return root


@pytest.fixture
def tiny_dataset_with_val(tiny_dataset):
    write_images(tiny_dataset / "val" / "good", 2, 60)
    return tiny_dataset


@pytest.fixture
def dataset_without_defects(tmp_path):
    root = tmp_path / "dataset"
    write_images(root / "train" / "good", 5)
    write_images(root / "test" / "good", 2)
    return root


@pytest.fixture
def tiny_image_file(tmp_path):
    write_images(tmp_path / "image", 1, size=300)
    return tmp_path / "image" / "000.png"


@pytest.fixture
def gray_image_file(tmp_path):
    path = tmp_path / "gray.png"
    cv2.imwrite(str(path), np.full((64, 64), 128, np.uint8))
    return path


@pytest.fixture
def penalty_folder(tmp_path):
    root = tmp_path / "penalty"
    write_images(root / "train" / "class_a", 2, 70)
    write_images(root / "val" / "class_b", 1, 71)
    return root
