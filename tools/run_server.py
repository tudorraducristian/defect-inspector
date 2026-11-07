"""Start the inspection station: the camera, the inspection loop and the web page.

Runs on Python 3.6, on the Jetson Nano. On the Nano, give --host 0.0.0.0 to reach the page
from the laptop; the page has no login, so the default stays on this machine only.
"""
import argparse
import logging

import uvicorn

from defect_inspector.camera import CameraWorker, ReplayCamera, settings_from_dict
from defect_inspector.loop import InspectionLoop
from defect_inspector.server import create_app

# Slow enough to see each verdict, as one image takes up to a few seconds on the processor.
REPLAY_FPS = 0.2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="profiles")
    parser.add_argument("--backend", default="ort-cpu")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, default=0)
    source.add_argument("--replay", help="folder of images to play instead of the camera")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def build_camera(args):
    if args.replay:
        return ReplayCamera(args.replay, fps=REPLAY_FPS)
    return CameraWorker(settings_from_dict({"index": args.camera}))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    camera = build_camera(args)
    loop = InspectionLoop(camera)
    camera.start()
    loop.start()
    try:
        uvicorn.run(create_app(loop, args.profiles, args.backend), host=args.host, port=args.port)
    finally:
        loop.stop()
        camera.stop()


if __name__ == "__main__":
    main()
