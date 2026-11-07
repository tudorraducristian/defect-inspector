"""The web server: the page, the state, the choice of object and the live image.

It must also run with fastapi 0.67 on the Jetson Nano, under Python 3.6, so the routes return
plain dicts, use no Pydantic models and no lifespan, and the loop starts in run_server.py.
"""
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from defect_inspector.backends import load_backend
from defect_inspector.camera import settings_from_dict
from defect_inspector.inspector import Inspector
from defect_inspector.profile import MODEL_FILE, list_profiles
from defect_inspector.scene import gate_for_profile

STATIC = Path(__file__).parent / "static"
STREAM_BOUNDARY = "frame"
STREAM_PERIOD_S = 0.1
JPEG_START = b"\xff\xd8"


def _placeholder_jpeg():
    """Shown instead of the last image, so a lost camera never looks like a still scene."""
    image = np.full((360, 640, 3), 40, np.uint8)
    cv2.putText(image, "Camera nu da imagine", (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return cv2.imencode(".jpg", image)[1].tobytes()


def _stream(loop, limit):
    placeholder = _placeholder_jpeg()
    sent = 0
    while limit is None or sent < limit:
        image = loop.jpeg() or placeholder
        header = "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n".format(STREAM_BOUNDARY, len(image))
        yield header.encode("ascii") + image + b"\r\n"
        sent += 1
        if limit is None or sent < limit:
            time.sleep(STREAM_PERIOD_S)


def _describe(profile):
    return {"id": profile.id, "name": profile.name, "threshold": profile.threshold,
            "image_size": profile.image_size, "metrics": profile.metrics}


def _activate(loop, profile, backend_name):
    """Prepare everything first, so a failure leaves the object in use untouched."""
    gate = gate_for_profile(profile)
    backend = load_backend(backend_name, profile.path / MODEL_FILE)
    if profile.camera:
        loop.configure_camera(settings_from_dict(profile.camera))
    loop.set_inspector(Inspector(profile, backend), profile, gate)


def create_app(loop, profiles_root, backend_name, stream_limit=None):
    """stream_limit ends the live stream after that many images; the tests need it to finish."""
    app = FastAPI(title="defect-inspector")

    @app.get("/", response_class=HTMLResponse)
    def page():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        return loop.state()

    @app.get("/api/profiles")
    def profiles():
        usable, errors = list_profiles(profiles_root)
        return {"profiles": [_describe(p) for p in usable], "errors": errors,
                "active": loop.state()["profile_id"]}

    @app.post("/api/profiles/{profile_id}/activate")
    def activate(profile_id):
        usable, errors = list_profiles(profiles_root)
        chosen = [p for p in usable if p.id == profile_id]
        if not chosen:
            problems = [e for e in errors if e.startswith(profile_id + ":")]
            if problems:
                raise HTTPException(status_code=409, detail=problems[0])
            raise HTTPException(status_code=404, detail="unknown profile {}".format(profile_id))
        try:
            _activate(loop, chosen[0], backend_name)
        except Exception as error:
            raise HTTPException(status_code=409, detail="{}: {}".format(profile_id, error))
        return loop.state()

    @app.post("/api/background")
    def background():
        frame = loop.latest_frame()
        if frame is None:
            raise HTTPException(status_code=409, detail="the camera gives no image")
        try:
            loop.set_background(frame)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))
        return loop.state()

    @app.get("/stream.mjpg")
    def stream():
        return StreamingResponse(_stream(loop, stream_limit),
                                 media_type="multipart/x-mixed-replace; boundary={}".format(STREAM_BOUNDARY))

    return app
