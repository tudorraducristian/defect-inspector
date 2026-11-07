"""Turn the raw photos of one object into the dataset format used by training.

Input, written by capture.py: good/piece-NN/shot-NN.jpg, defect/<type>/piece-NN/shot-NN.jpg,
station.json, background.jpg.
Output: train/good/, val/good/, test/good/, test/<type>/, plus split.json, station.json and
background.jpg. All photos of a piece stay on one side, and images are cut to the zone of
interest, because that is what the inspector will look at. PNG, so no quality is lost twice.
Runs on Python 3.6.
"""
import argparse
import json
import shutil
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

from defect_inspector.postprocessing import crop_roi

MIN_TRAIN_PIECES = 10


class DatasetError(Exception):
    """The raw photos cannot be used; the message says why."""


class DatasetTooSmall(DatasetError):
    """There are too few pieces for a fair split."""


def _pieces(folder):
    """Map each piece folder name to its photos; folders without photos do not count."""
    if not folder.is_dir():
        return {}
    pieces = {}
    for piece in sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("piece-")):
        shots = sorted(piece.glob("shot-*.jpg"))
        if shots:
            pieces[piece.name] = shots
    return pieces


def _copy_cut(shots, piece, target, roi):
    target.mkdir(parents=True, exist_ok=True)
    for shot in shots:
        frame = cv2.imread(str(shot))
        if frame is None:
            raise DatasetError("cannot read {}/{}".format(piece, shot.name))
        try:
            cut = crop_roi(frame, roi)
        except ValueError as error:
            raise DatasetError("{}/{}: {}".format(piece, shot.name, error))
        cv2.imwrite(str(target / "{}_{}.png".format(piece, shot.stem)), cut)
    return len(shots)


def _read_station(src):
    path = src / "station.json"
    if not path.exists():
        raise DatasetError("station.json is missing from {}; run capture.py --setup first".format(src))
    return json.loads(path.read_text(encoding="utf-8"))


def _read_defects(src):
    defect_root = src / "defect"
    kinds = sorted(p.name for p in defect_root.iterdir() if p.is_dir()) if defect_root.is_dir() else []
    if "good" in kinds:
        raise DatasetError("a defect type cannot be named good")
    defects = {kind: _pieces(defect_root / kind) for kind in kinds}
    defects = {kind: pieces for kind, pieces in defects.items() if pieces}
    if not defects:
        raise DatasetTooSmall("no defect photos in {}".format(defect_root))
    return defects


def _split_good(names, test_good_pieces, val_fraction, seed):
    """Same seed, same split; the order comes from the shuffled list of piece names."""
    needed = test_good_pieces + MIN_TRAIN_PIECES
    if len(names) < needed:
        raise DatasetTooSmall("{} good pieces, at least {} are needed".format(len(names), needed))
    order = [names[i] for i in np.random.RandomState(seed).permutation(len(names))]
    test = order[:test_good_pieces]
    rest = order[test_good_pieces:]
    val_count = max(1, int(round(val_fraction * len(rest))))
    return sorted(rest[val_count:]), sorted(rest[:val_count]), sorted(test)


def prepare(src, dst, test_good_pieces=10, val_fraction=0.1, seed=42):
    src, dst = Path(src), Path(dst)
    station = _read_station(src)
    defects = _read_defects(src)
    good = _pieces(src / "good")
    train, val, test = _split_good(sorted(good), test_good_pieces, val_fraction, seed)
    if dst.exists() and any(dst.iterdir()):
        raise DatasetError("{} is not empty; a new split must not mix with an old one".format(dst))

    roi = station.get("roi")
    images = {"train": 0, "val": 0, "test_good": 0, "test_defect": 0}
    for name in train:
        images["train"] += _copy_cut(good[name], name, dst / "train" / "good", roi)
    for name in val:
        images["val"] += _copy_cut(good[name], name, dst / "val" / "good", roi)
    for name in test:
        images["test_good"] += _copy_cut(good[name], name, dst / "test" / "good", roi)
    for kind, pieces in sorted(defects.items()):
        for name, shots in sorted(pieces.items()):
            images["test_defect"] += _copy_cut(shots, name, dst / "test" / kind, roi)

    summary = {
        "seed": seed,
        "train_pieces": len(train),
        "val_pieces": len(val),
        "test_good_pieces": len(test),
        "defect_pieces": {kind: len(pieces) for kind, pieces in sorted(defects.items())},
        "images": images,
        "pieces": {"train": train, "val": val, "test_good": test},
    }
    (dst / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy(str(src / "station.json"), str(dst / "station.json"))
    if (src / "background.jpg").exists():
        shutil.copy(str(src / "background.jpg"), str(dst / "background.jpg"))
    return summary


def make_tar(folder, archive):
    """Pack the dataset for Google Drive; photos do not compress further, so no compression."""
    folder, archive = Path(folder), Path(archive)
    with tarfile.open(str(archive), "w") as tar:
        tar.add(str(folder), arcname=".")
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--test-good-pieces", type=int, default=10)
    args = parser.parse_args(argv)
    ready = args.data / (args.object + "-ready")
    try:
        summary = prepare(args.data / args.object, ready, test_good_pieces=args.test_good_pieces)
    except DatasetError as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in summary.items() if k != "pieces"}, indent=2))
    print(make_tar(ready, args.data / (args.object + ".tar")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
