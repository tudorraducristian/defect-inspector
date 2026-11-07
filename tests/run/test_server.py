import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from defect_inspector import server as server_module
from defect_inspector.server import create_app

STATE_FIELDS = {"status", "label", "score", "threshold", "latency_ms", "profile_id", "frame_id", "message"}
GOOD_FIELDS = {
    "schema_version": 1, "object_id": "piulita-m10", "object_name": "Piuliță M10", "image_size": 32,
    "threshold": 0.5, "mean": [0.4, 0.4, 0.4], "std": [0.2, 0.2, 0.2], "map_padding": 4,
    "metrics": {"image_auroc": 0.9}, "roi": None, "camera": {"width": 640, "height": 480},
}
FRAME = np.zeros((8, 8, 3), np.uint8)


class FakeLoop:
    def __init__(self, frame=FRAME, jpeg=b"JPEGDATA"):
        self.frame = frame
        self.image = jpeg
        self.profile = None
        self.gate = None
        self.background = None
        self.camera_settings = None
        self.refuse_background = False

    def state(self):
        return {"status": "NO_PROFILE", "label": None, "score": None, "threshold": None, "latency_ms": None,
                "profile_id": self.profile.id if self.profile else None, "frame_id": None, "message": "x"}

    def set_inspector(self, inspector, profile, gate):
        self.profile, self.gate = profile, gate

    def configure_camera(self, settings):
        self.camera_settings = settings

    def latest_frame(self):
        return self.frame

    def set_background(self, frame):
        if self.refuse_background:
            raise ValueError("no active profile")
        self.background = frame

    def jpeg(self):
        return self.image


def write_profile(root, folder="piulita-m10", **changes):
    path = root / folder
    path.mkdir(parents=True)
    (path / "profile.json").write_text(json.dumps(dict(GOOD_FIELDS, **changes)), encoding="utf-8")
    (path / "model.onnx").write_bytes(b"model")
    return path


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(server_module, "load_backend", lambda name, path: object())


@pytest.fixture
def fake_loop():
    return FakeLoop()


@pytest.fixture
def profiles_root(tmp_path):
    write_profile(tmp_path)
    return tmp_path


@pytest.fixture
def client(fake_loop, profiles_root):
    return TestClient(create_app(fake_loop, profiles_root, "ort-cpu"))


@pytest.fixture
def client_with_one_broken_profile(fake_loop, profiles_root):
    broken = profiles_root / "broken-one"
    broken.mkdir()
    (broken / "profile.json").write_text("{not json", encoding="utf-8")
    return TestClient(create_app(fake_loop, profiles_root, "ort-cpu"))


@pytest.fixture
def client_with_camera_lost(profiles_root):
    return TestClient(create_app(FakeLoop(frame=None), profiles_root, "ort-cpu"))


@pytest.fixture
def client_with_finite_stream(fake_loop, profiles_root):
    return TestClient(create_app(fake_loop, profiles_root, "ort-cpu", stream_limit=3))


def test_state_has_every_field(client):
    assert set(client.get("/api/state").json()) == STATE_FIELDS


def test_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")
    assert "/stream.mjpg" in response.text and "/api/state" in response.text


def test_profiles_lists_active_and_errors(client_with_one_broken_profile):
    body = client_with_one_broken_profile.get("/api/profiles").json()
    assert [p["id"] for p in body["profiles"]] == ["piulita-m10"] and body["errors"]
    assert body["profiles"][0]["name"] == "Piuliță M10" and body["active"] is None


def test_profiles_marks_the_active_one(client):
    client.post("/api/profiles/piulita-m10/activate")
    assert client.get("/api/profiles").json()["active"] == "piulita-m10"


def test_activate_unknown_profile_is_404(client):
    assert client.post("/api/profiles/nope/activate").status_code == 404


def test_activate_broken_profile_is_409_with_message(client_with_one_broken_profile):
    r = client_with_one_broken_profile.post("/api/profiles/broken-one/activate")
    assert r.status_code == 409 and r.json()["detail"]


def test_activate_changes_state(client, fake_loop):
    client.post("/api/profiles/piulita-m10/activate")
    assert fake_loop.state()["profile_id"] == "piulita-m10"


def test_activate_returns_the_state(client):
    assert client.post("/api/profiles/piulita-m10/activate").json()["profile_id"] == "piulita-m10"


def test_activate_applies_the_camera_settings_of_the_profile(client, fake_loop):
    client.post("/api/profiles/piulita-m10/activate")
    assert (fake_loop.camera_settings.width, fake_loop.camera_settings.height) == (640, 480)


def test_activate_on_another_kind_of_machine_skips_the_locked_values(fake_loop, tmp_path, monkeypatch):
    import defect_inspector.camera as camera_module
    monkeypatch.setattr(camera_module.sys, "platform", "linux")
    write_profile(tmp_path, camera={"width": 640, "height": 480, "exposure": -4.0, "platform": "win32"})
    client = TestClient(create_app(fake_loop, tmp_path, "ort-cpu"))
    client.post("/api/profiles/piulita-m10/activate")
    assert fake_loop.camera_settings.width == 640 and fake_loop.camera_settings.exposure is None


def test_activate_profile_without_camera_settings_leaves_the_camera(fake_loop, tmp_path):
    write_profile(tmp_path, camera=None)
    client = TestClient(create_app(fake_loop, tmp_path, "ort-cpu"))
    client.post("/api/profiles/piulita-m10/activate")
    assert fake_loop.camera_settings is None


def test_activate_builds_the_gate_from_the_table_photo(fake_loop, tmp_path):
    import cv2
    folder = write_profile(tmp_path)
    cv2.imwrite(str(folder / "background.jpg"), FRAME)
    TestClient(create_app(fake_loop, tmp_path, "ort-cpu")).post("/api/profiles/piulita-m10/activate")
    assert fake_loop.gate is not None


def test_activate_with_a_model_that_cannot_load_is_409(client, monkeypatch, fake_loop):
    def refuse(name, path):
        raise RuntimeError("bad model")

    monkeypatch.setattr(server_module, "load_backend", refuse)
    r = client.post("/api/profiles/piulita-m10/activate")
    assert r.status_code == 409 and "bad model" in r.json()["detail"] and fake_loop.profile is None


def test_activate_with_a_broken_table_photo_is_409(fake_loop, tmp_path):
    folder = write_profile(tmp_path)
    (folder / "background.jpg").write_bytes(b"broken")
    r = TestClient(create_app(fake_loop, tmp_path, "ort-cpu")).post("/api/profiles/piulita-m10/activate")
    assert r.status_code == 409 and "background" in r.json()["detail"]


def test_background_without_frame_is_409(client_with_camera_lost):
    assert client_with_camera_lost.post("/api/background").status_code == 409


def test_background_without_a_profile_is_409(fake_loop, client):
    fake_loop.refuse_background = True
    assert client.post("/api/background").status_code == 409


def test_background_uses_the_live_frame(client, fake_loop):
    assert client.post("/api/background").status_code == 200
    assert fake_loop.background is FRAME


def test_stream_is_multipart(client_with_finite_stream):
    r = client_with_finite_stream.get("/stream.mjpg")
    assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert r.content.count(b"JPEGDATA") == 3


def test_stream_without_a_frame_shows_a_placeholder_not_the_old_image(profiles_root):
    client = TestClient(create_app(FakeLoop(jpeg=None), profiles_root, "ort-cpu", stream_limit=1))
    body = client.get("/stream.mjpg").content
    assert b"\xff\xd8" in body and b"JPEGDATA" not in body
