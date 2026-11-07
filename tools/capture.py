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
WINDOW_SIZE = (1280, 720)
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


# Each control the setup locks: the setting, the OpenCV property, the name shown to Tudor, and which
# values are real. OpenCV gives -1 for a control the camera does not have.
LOCKED_CONTROLS = [
    ("exposure", cv2.CAP_PROP_EXPOSURE, "expunere", lambda v: v not in (0, -1)),
    ("white_balance", cv2.CAP_PROP_WB_TEMPERATURE, "balans alb", lambda v: v > 0),
    ("focus", cv2.CAP_PROP_FOCUS, "focalizare", lambda v: v >= 0),
    ("gain", cv2.CAP_PROP_GAIN, "castig", lambda v: v >= 0),
]


def _locked(settings, capture):
    """Keep what the camera settled on, so light changes do not change the image later."""
    values = {}
    for name, prop, _, is_real in LOCKED_CONTROLS:
        value = capture.get(prop)
        values[name] = value if is_real(value) else None
    return settings._replace(**values)


def _report_locked(settings, capture):
    """Print what the reopened camera holds, so a lock that did not take is seen at once."""
    print("Camera, dupa blocare:")
    for name, prop, label, _ in LOCKED_CONTROLS:
        if getattr(settings, name) is None:
            print("  {}: nu se poate bloca".format(label))
        else:
            print("  {} {:g} (cerut {:g})".format(label, capture.get(prop), getattr(settings, name)))


def _open_window():
    """A window that scales the frame down, so 1920x1080 still fits a laptop screen."""
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, *WINDOW_SIZE)


def _show(frame, lines, roi=None):
    preview = frame.copy()
    # Text and frame are scaled down together, so the text grows with the frame.
    scale = max(1.0, frame.shape[1] / float(WINDOW_SIZE[0]))
    if roi:
        cv2.rectangle(preview, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), int(2 * scale))
    for index, line in enumerate(lines):
        y = int((30 + 30 * index) * scale)
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0), int(4 * scale))
        cv2.putText(preview, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (255, 255, 255), int(2 * scale))
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


def run_setup(settings, root, lock_from_camera=True):
    capture = open_camera(settings)
    try:
        _open_window()
        if _wait_for_enter(capture, ["Aseaza camera si lumina.", "Enter: blochez setarile. Esc: renunt."]) is None:
            return False
        # Reopened with the locked settings, so the table photo looks like the later photos.
        if lock_from_camera:
            settings = _locked(settings, capture)
        capture.release()
        capture = open_camera(settings)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("the camera gives no frames")
        _report_locked(settings, capture)
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


def _step_lines(step, shot_index, position, total, flip_after=None):
    name = "Piesa {}".format(step.piece) if step.kind == "good" else "{} piesa {}".format(step.defect_type, step.piece)
    shot = step.shots_done + shot_index + 1
    lines = ["{} ({} din {})".format(name, position, total), "Poza {} din {}".format(shot, step.shots_total)]
    if flip_after:
        # Counted from the shot number, so a resumed piece still shows the right face.
        lines.append("Fata A" if shot <= flip_after else "Intoarce piulita: fata B")
    return lines + ["Enter: poza. Esc: opresc."]


def run_session(camera, steps, roi, flip_after=None):
    """One window for the whole session; return False if Tudor stopped it with Esc."""
    last_shot_at = 0.0
    try:
        _open_window()
        for position, step in enumerate(steps, 1):
            for shot_index, path in enumerate(step.paths):
                lines = _step_lines(step, shot_index, position, len(steps), flip_after)
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
    parser.add_argument("--flip-after", type=int, help="turn the piece over after this many photos")
    parser.add_argument("--force", action="store_true", help="with --setup, replace a station that has photos")
    # Measured values, for a camera that does not report what it uses. Given ones are locked, the rest stay
    # automatic, and nothing is read from the camera.
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--focus", type=float)
    parser.add_argument("--white-balance", type=float)
    args = parser.parse_args(argv)
    root = args.data / args.object
    types = _check_arguments(parser, args)

    if args.setup:
        # A new zone or new camera values would apply to the photos already taken, which saw the old ones.
        if not args.force and any((root / kind).is_dir() and any((root / kind).rglob("*.jpg"))
                                  for kind in ("good", "defect")):
            print("{} are deja poze; o statie noua le-ar strica. Adauga --force daca vrei asta.".format(root),
                  file=sys.stderr)
            return 2
        measured = {"exposure": args.exposure, "focus": args.focus, "white_balance": args.white_balance}
        lock_from_camera = all(value is None for value in measured.values())
        settings = settings_from_dict(dict(measured, index=args.camera, width=args.width,
                                           height=args.height, fps=args.fps))
        if args.headless:
            run_setup_headless(settings, root, _parse_roi(args.roi) if args.roi else None)
            return 0
        return 0 if run_setup(settings, root, lock_from_camera) else 1

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
        return 0 if run_session(camera, steps, station.roi, args.flip_after) else 1
    finally:
        camera.stop()


if __name__ == "__main__":
    sys.exit(main())
