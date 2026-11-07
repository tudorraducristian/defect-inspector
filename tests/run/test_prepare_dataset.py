import json
import re
import tarfile

import cv2
import numpy as np
import pytest

from capture import shot_path
from prepare_dataset import DatasetError, DatasetTooSmall, make_tar, prepare

FRAME = np.full((40, 60, 3), 90, np.uint8)
ROI = [10, 5, 20, 16]


def make_raw(root, good_pieces, roi=ROI, defect_types=("marker", "lovitura"), with_station=True):
    root.mkdir(parents=True, exist_ok=True)
    shots = [("good", piece, None) for piece in range(1, good_pieces + 1)]
    shots += [("defect", piece, kind) for kind in defect_types for piece in (1, 2)]
    for kind, piece, defect_type in shots:
        for shot in (1, 2):
            path = shot_path(root, kind, piece, shot, defect_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), FRAME)
    cv2.imwrite(str(root / "background.jpg"), FRAME)
    if with_station:
        (root / "station.json").write_text(json.dumps({"roi": roi, "camera": {"index": 0}}), encoding="utf-8")
    return root


def pieces_in(folder):
    found = set()
    for path in folder.rglob("*.png"):
        found.add(re.search(r"piece-\d+", path.name).group())
    return found


@pytest.fixture
def raw_dataset(tmp_path_factory):
    return make_raw(tmp_path_factory.mktemp("raw"), 25)


@pytest.fixture
def raw_dataset_40_pieces(tmp_path_factory):
    return make_raw(tmp_path_factory.mktemp("raw"), 40)


@pytest.fixture
def raw_dataset_12_pieces(tmp_path_factory):
    return make_raw(tmp_path_factory.mktemp("raw"), 12)


def test_every_piece_stays_on_one_side(raw_dataset, tmp_path):
    prepare(raw_dataset, tmp_path)
    assert pieces_in(tmp_path / "train")
    assert pieces_in(tmp_path / "test" / "good")
    assert not pieces_in(tmp_path / "train") & pieces_in(tmp_path / "test" / "good")
    assert not pieces_in(tmp_path / "train") & pieces_in(tmp_path / "val")
    assert not pieces_in(tmp_path / "val") & pieces_in(tmp_path / "test" / "good")


def test_all_shots_of_a_piece_are_kept(raw_dataset, tmp_path):
    summary = prepare(raw_dataset, tmp_path)
    assert summary["images"]["train"] == 2 * summary["train_pieces"]
    assert len(list((tmp_path / "train" / "good").iterdir())) == summary["images"]["train"]


def test_defect_pieces_only_in_test(raw_dataset, tmp_path):
    prepare(raw_dataset, tmp_path)
    assert (tmp_path / "test" / "marker").exists() and not (tmp_path / "train" / "marker").exists()
    assert len(list((tmp_path / "test" / "marker").iterdir())) == 4


def test_ten_good_pieces_are_held_for_test(raw_dataset_40_pieces, tmp_path):
    summary = prepare(raw_dataset_40_pieces, tmp_path)
    assert summary["test_good_pieces"] == 10 and summary["val_pieces"] == 3 and summary["train_pieces"] == 27


def test_too_few_pieces_is_refused(raw_dataset_12_pieces, tmp_path):
    with pytest.raises(DatasetTooSmall):
        prepare(raw_dataset_12_pieces, tmp_path)


def test_no_defects_is_refused(tmp_path_factory, tmp_path):
    raw = make_raw(tmp_path_factory.mktemp("raw"), 25, defect_types=())
    with pytest.raises(DatasetTooSmall, match="defect"):
        prepare(raw, tmp_path)


def test_split_is_deterministic_and_skips_background(raw_dataset, tmp_path):
    a = prepare(raw_dataset, tmp_path / "a")
    b = prepare(raw_dataset, tmp_path / "b")
    assert a == b and not list((tmp_path / "a").rglob("train/**/background.jpg"))


def test_images_are_cut_to_the_zone_the_inspector_will_use(raw_dataset, tmp_path):
    prepare(raw_dataset, tmp_path)
    image = cv2.imread(str(next((tmp_path / "train" / "good").iterdir())))
    assert image.shape[:2] == (ROI[3], ROI[2])


def test_station_without_zone_keeps_whole_images(tmp_path_factory, tmp_path):
    raw = make_raw(tmp_path_factory.mktemp("raw"), 25, roi=None)
    prepare(raw, tmp_path)
    assert cv2.imread(str(next((tmp_path / "train" / "good").iterdir()))).shape == FRAME.shape


def test_station_and_background_and_split_go_along(raw_dataset, tmp_path):
    summary = prepare(raw_dataset, tmp_path)
    assert (tmp_path / "background.jpg").exists()
    assert json.loads((tmp_path / "station.json").read_text(encoding="utf-8"))["roi"] == ROI
    assert json.loads((tmp_path / "split.json").read_text(encoding="utf-8")) == summary


def test_missing_station_is_refused(tmp_path_factory, tmp_path):
    raw = make_raw(tmp_path_factory.mktemp("raw"), 25, with_station=False)
    with pytest.raises(DatasetError, match="station.json"):
        prepare(raw, tmp_path)


def test_unreadable_image_is_named(raw_dataset, tmp_path):
    bad = shot_path(raw_dataset, "good", 3, 1)
    bad.write_bytes(b"not an image")
    with pytest.raises(DatasetError, match="piece-03"):
        prepare(raw_dataset, tmp_path)


def test_folder_with_old_files_is_refused(raw_dataset, tmp_path):
    (tmp_path / "old.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DatasetError, match="not empty"):
        prepare(raw_dataset, tmp_path)


def test_defect_folder_named_good_is_refused(raw_dataset, tmp_path):
    path = raw_dataset / "defect" / "good" / "piece-01" / "shot-01.jpg"
    path.parent.mkdir(parents=True)
    cv2.imwrite(str(path), FRAME)
    with pytest.raises(DatasetError, match="good"):
        prepare(raw_dataset, tmp_path)


def test_tar_holds_the_dataset_format(raw_dataset, tmp_path):
    prepare(raw_dataset, tmp_path / "ready")
    archive = make_tar(tmp_path / "ready", tmp_path / "object.tar")
    with tarfile.open(str(archive)) as tar:
        names = tar.getnames()
    assert any(re.search(r"train/good/piece-\d+_shot-\d+\.png$", name) for name in names)
    assert any(name.endswith("station.json") for name in names)
