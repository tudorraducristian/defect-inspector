"""Read inspection profiles: one folder per object, written by training.

Runs on Python 3.6, on the Jetson Nano.
"""
import json
from collections import namedtuple
from pathlib import Path

SCHEMA_VERSION = 1
MODEL_FILE = "model.onnx"
PROFILE_FILE = "profile.json"
BACKGROUND_FILE = "background.jpg"
REQUIRED_FIELDS = ("schema_version", "object_id", "object_name", "image_size", "threshold",
                   "mean", "std", "map_padding", "metrics")

Profile = namedtuple("Profile", ["id", "name", "path", "image_size", "threshold", "mean", "std",
                                 "map_padding", "roi", "camera", "metrics"])


class ProfileError(Exception):
    """A profile folder cannot be used; the message says why."""


def _read_fields(folder):
    path = folder / PROFILE_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        raise ProfileError("{} is missing or cannot be read".format(PROFILE_FILE))
    except ValueError as error:
        raise ProfileError("{} is not valid JSON: {}".format(PROFILE_FILE, error))


def load_profile(folder):
    folder = Path(folder)
    fields = _read_fields(folder)
    if not isinstance(fields, dict):
        raise ProfileError("{} must hold an object".format(PROFILE_FILE))
    missing = [name for name in REQUIRED_FIELDS if name not in fields]
    if missing:
        raise ProfileError("{} lacks {}".format(PROFILE_FILE, ", ".join(missing)))
    if fields["schema_version"] != SCHEMA_VERSION:
        raise ProfileError("schema_version {} is not supported, this version reads {}".format(
            fields["schema_version"], SCHEMA_VERSION))
    if not (folder / MODEL_FILE).is_file():
        raise ProfileError("{} is missing".format(MODEL_FILE))
    return Profile(
        id=fields["object_id"],
        name=fields["object_name"],
        path=folder,
        image_size=fields["image_size"],
        threshold=fields["threshold"],
        mean=fields["mean"],
        std=fields["std"],
        map_padding=fields["map_padding"],
        roi=fields.get("roi"),
        camera=fields.get("camera"),
        metrics=fields["metrics"],
    )


def list_profiles(root):
    """Return (usable profiles, messages); one bad folder never hides the others."""
    root = Path(root)
    if not root.is_dir():
        return [], ["profiles folder {} does not exist".format(root)]
    profiles, errors, seen = [], [], set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        try:
            profile = load_profile(folder)
        except ProfileError as error:
            errors.append("{}: {}".format(folder.name, error))
            continue
        if profile.id in seen:
            errors.append("{}: object_id {} is already used by another folder".format(folder.name, profile.id))
            continue
        seen.add(profile.id)
        profiles.append(profile)
    return profiles, errors
