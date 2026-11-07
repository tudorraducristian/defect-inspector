"""Set up the station and take the photos of one object.

  capture.py --object X --setup                                  camera, zone, empty table
  capture.py --object X --session good --pieces 40 --shots 5     one window, Enter per photo
  capture.py --object X --session defect --types marker=5 --shots 3
  capture.py --object X --kind good --piece 7 --shots 1 --headless   no window, for the Nano

Photos stay as the camera gave them, whole; prepare_dataset.py cuts them to the zone.
A stopped session continues where it was. Runs on Python 3.6. The windows need OpenCV
with GUI support; --headless does not.
"""
import argparse
import json
import re
import sys
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from defect_inspector.camera import CameraWorker, load_station, open_camera, settings_from_dict

JPEG_QUALITY = 95
SHOT_NAME = re.compile(r"^shot-(\d+)\.jpg$")
TYPE_NAME = re.compile(r"^[a-z0-9_-]+$")
KEY_ENTER = (10, 13)
KEY_ESC = 27
MIN_SECONDS_BETWEEN_SHOTS = 0.4
WINDOW = "defect-inspector"
NO_CAMERA_SIZE = (360, 640, 3)
NO_CAMERA_LINES = ["Camera nu da imagine.", "Reincerc singur. Esc: opresc."]

SessionStep = namedtuple("SessionStep", ["kind", "defect_type", "piece", "shots_done", "shots_total", "paths"])


def shot_path(root, kind, piece, shot, defect_type=None):
    if kind == "defect" and not defect_type:
        raise ValueError("a defect photo needs a defect type")
    folder = Path(root) / kind
    if kind == "defect":
        folder = folder / defect_type
    return folder / "piece-{:02d}".format(piece) / "shot-{:02d}.jpg".format(shot)


def _existing_shots(root, kind, piece, defect_type=None):
    folder = shot_path(root, kind, piece, 1, defect_type).parent
    if not folder.is_dir():
        return []
    matches = (SHOT_NAME.match(p.name) for p in folder.iterdir())
    return sorted(int(m.group(1)) for m in matches if m)


def next_shots(root, kind, piece, count, defect_type=None):
    """The next free paths for a piece, always after its highest photo, so nothing is overwritten."""
    existing = _existing_shots(root, kind, piece, defect_type)
    first = existing[-1] + 1 if existing else 1
    return [shot_path(root, kind, piece, shot, defect_type) for shot in range(first, first + count)]


def parse_types(text):
    """Read zgarietura=5,marker=5 into [("zgarietura", 5), ("marker", 5)]."""
    types = []
    for part in text.split(","):
        name, _, count = part.partition("=")
        if not TYPE_NAME.match(name) or name == "good" or not count.isdigit() or int(count) < 1:
            raise ValueError("expected name=count, with a name other than good, got {}".format(part))
        types.append((name, int(count)))
    return types


def plan_session(root, kind, pieces=None, shots=1, types=None):
    """List the photos still to take; pieces that already have enough are left out."""
    if kind == "good":
        wanted = [(None, piece) for piece in range(1, pieces + 1)]
    else:
        wanted = [(name, piece) for name, count in types for piece in range(1, count + 1)]
    steps = []
    for defect_type, piece in wanted:
        done = len(_existing_shots(root, kind, piece, defect_type))
        if done < shots:
            paths = next_shots(root, kind, piece, shots - done, defect_type)
            steps.append(SessionStep(kind, defect_type, piece, done, shots, paths))
    return steps


def save_image(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise RuntimeError("cannot write {}".format(path))


def capture_shots(camera, paths, timeout_s=10.0):
    """Without a window: save one new frame per path."""
    last_id = None
    for path in paths:
        deadline = time.monotonic() + timeout_s
        while True:
            latest = camera.latest()
            if latest is not None and latest[1] != last_id:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("the camera gives no new frames")
            time.sleep(0.01)
        save_image(path, latest[0])
        last_id = latest[1]


def build_station(settings, roi):
    return {"roi": roi, "camera": dict(settings._asdict())}


def write_station(root, station, background):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    save_image(root / "background.jpg", background)
    (root / "station.json").write_text(json.dumps(station, indent=2), encoding="utf-8")


def _locked(settings, capture):
    """Keep what the camera settled on, so light changes do not change the image later."""
    exposure = capture.get(cv2.CAP_PROP_EXPOSURE)
    white_balance = capture.get(cv2.CAP_PROP_WB_TEMPERATURE)
    return settings._replace(exposure=exposure if exposure else None,
                             white_balance=white_balance if white_balance > 0 else None)


def _show(frame, lines, roi=None):
    preview = frame.copy()
    if roi:
        cv2.rectangle(preview, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
    for index, line in enumerate(lines):
        y = 30 + 30 * index
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WINDOW, preview)
    return cv2.waitKey(30) & 0xFF


def _wait_for_enter(capture, lines):
    """Show the camera until Enter; return the frame at that moment, or None on Esc."""
    while True:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("the camera gives no frames")
        key = _show(frame, lines)
        if key in KEY_ENTER:
            return frame
        if key == KEY_ESC:
            return None


def run_setup(settings, root):
    capture = open_camera(settings)
    try:
        if _wait_for_enter(capture, ["Aseaza camera si lumina.", "Enter: blochez setarile. Esc: renunt."]) is None:
            return False
        # Reopened with the locked settings, so the table photo looks like the later photos.
        settings = _locked(settings, capture)
        capture.release()
        capture = open_camera(settings)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("the camera gives no frames")
        x, y, width, height = cv2.selectROI(WINDOW, frame, showCrosshair=False)
        if width == 0 or height == 0:
            return False
        background = _wait_for_enter(capture, ["Lasa masa goala.", "Enter: salvez poza mesei. Esc: renunt."])
        if background is None:
            return False
        write_station(root, build_station(settings, [int(x), int(y), int(width), int(height)]), background)
        return True
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_setup_headless(settings, root, roi):
    capture = open_camera(settings)
    try:
        # The first frames still have the exposure moving.
        for _ in range(30):
            ok, background = capture.read()
            if not ok:
                raise RuntimeError("the camera gives no frames")
        write_station(root, build_station(_locked(settings, capture), roi), background)
    finally:
        capture.release()


def _step_lines(step, shot_index, position, total):
    name = "Piesa {}".format(step.piece) if step.kind == "good" else "{} piesa {}".format(step.defect_type, step.piece)
    return ["{} ({} din {})".format(name, position, total),
            "Poza {} din {}".format(step.shots_done + shot_index + 1, step.shots_total),
            "Enter: poza. Esc: opresc."]


def run_session(camera, steps, roi):
    """One window for the whole session; return False if Tudor stopped it with Esc."""
    last_shot_at = 0.0
    try:
        for position, step in enumerate(steps, 1):
            for shot_index, path in enumerate(step.paths):
                lines = _step_lines(step, shot_index, position, len(steps))
                while True:
                    latest = camera.latest()
                    if latest is None:
                        if _show(np.zeros(NO_CAMERA_SIZE, np.uint8), NO_CAMERA_LINES) == KEY_ESC:
                            return False
                        continue
                    key = _show(latest[0], lines, roi)
                    if key == KEY_ESC:
                        return False
                    if key in KEY_ENTER and time.monotonic() - last_shot_at > MIN_SECONDS_BETWEEN_SHOTS:
                        save_image(path, latest[0])
                        last_shot_at = time.monotonic()
                        break
        return True
    finally:
        cv2.destroyAllWindows()


def _parse_roi(text):
    values = [int(v) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("expected --roi x,y,width,height")
    return values


def _check_arguments(parser, args):
    """Refuse an incomplete command before the camera starts; return the parsed defect types."""
    if args.setup:
        return None
    if args.headless:
        if not args.kind or not args.piece:
            parser.error("--headless needs --kind and --piece")
        if args.kind == "defect" and not args.defect_type:
            parser.error("--kind defect needs --defect-type")
        return None
    if args.session == "good" and not args.pieces:
        parser.error("--session good needs --pieces")
    if args.session == "defect" and not args.types:
        parser.error("--session defect needs --types")
    if not args.session:
        parser.error("choose --setup, --session or --headless")
    try:
        return parse_types(args.types) if args.types else None
    except ValueError as error:
        parser.error(str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--session", choices=["good", "defect"])
    parser.add_argument("--pieces", type=int)
    parser.add_argument("--types")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--kind", choices=["good", "defect"])
    parser.add_argument("--defect-type")
    parser.add_argument("--piece", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--roi", help="x,y,width,height, with --setup --headless")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)
    root = args.data / args.object
    types = _check_arguments(parser, args)

    if args.setup:
        settings = settings_from_dict({"index": args.camera, "width": args.width,
                                       "height": args.height, "fps": args.fps})
        if args.headless:
            run_setup_headless(settings, root, _parse_roi(args.roi) if args.roi else None)
            return 0
        return 0 if run_setup(settings, root) else 1

    station_path = root / "station.json"
    if not station_path.exists():
        print("Lipseste {}; ruleaza intai --setup.".format(station_path), file=sys.stderr)
        return 2
    station = load_station(station_path)
    camera = CameraWorker(station.camera)
    camera.start()
    try:
        if args.headless:
            capture_shots(camera, next_shots(root, args.kind, args.piece, args.shots, args.defect_type))
            return 0
        steps = plan_session(root, args.session, pieces=args.pieces, shots=args.shots, types=types)
        return 0 if run_session(camera, steps, station.roi) else 1
    finally:
        camera.stop()


if __name__ == "__main__":
    sys.exit(main())
