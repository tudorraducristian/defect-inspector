import cv2
import numpy as np
import pytest

from defect_inspector.preprocessing import IMAGENET_MEAN, IMAGENET_STD, to_model_input
from defect_train.data import DatasetError, PenaltyImages, TrainImages, list_images, load_splits


def test_training_and_runtime_preprocess_identically(tiny_image_file):
    clean, _ = TrainImages([tiny_image_file], 256)[0]
    frame = cv2.imread(str(tiny_image_file))
    expected = to_model_input(frame, 256, IMAGENET_MEAN, IMAGENET_STD)[0]
    assert np.abs(clean.numpy() - expected).max() < 1e-6


def test_splits_are_disjoint_and_deterministic(tiny_dataset):
    a, b = load_splits(tiny_dataset), load_splits(tiny_dataset)
    assert a.train == b.train and not set(a.train) & set(a.val)
    assert len(a.train) == 9 and len(a.val) == 1


def test_uses_val_folder_when_present(tiny_dataset_with_val):
    splits = load_splits(tiny_dataset_with_val)
    assert all("/val/" in p.as_posix() for p in splits.val)
    assert len(splits.train) == 10


def test_test_split_has_defect_types_and_ignores_ground_truth(tiny_dataset):
    test = load_splits(tiny_dataset).test
    assert {kind for _, kind in test} == {"good", "scratch", "bent"}
    assert not any("ground_truth" in path.as_posix() for path, _ in test)


def test_gray_image_is_loaded_as_three_channels(gray_image_file):
    clean, _ = TrainImages([gray_image_file], 256)[0]
    assert clean.shape == (3, 256, 256)


def test_augmented_has_the_same_shape(tiny_image_file):
    clean, augmented = TrainImages([tiny_image_file], 64)[0]
    assert augmented.shape == clean.shape


def test_unreadable_image_raises_clear_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="broken.png"):
        TrainImages([path], 64)[0]


def test_missing_defects_raise_clear_error(dataset_without_defects):
    with pytest.raises(DatasetError, match="defect"):
        load_splits(dataset_without_defects)


def test_missing_training_images_raise_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="train"):
        load_splits(tmp_path)


def test_list_images_is_sorted_and_keeps_only_images(tmp_path):
    for name in ("b.PNG", "a.jpg", "c.jpeg", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpg", "b.PNG", "c.jpeg"]


def test_penalty_images_finds_nested_files_and_crops(penalty_folder):
    images = PenaltyImages(penalty_folder, 32)
    assert len(images) == 3 and images[0].shape == (3, 32, 32)


def test_empty_penalty_folder_raises_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="penalty"):
        PenaltyImages(tmp_path, 32)
