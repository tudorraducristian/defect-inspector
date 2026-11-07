import itertools
import time

import cv2
import numpy as np
import pytest

from defect_inspector.inspector import Inspector
from defect_inspector.loop import InspectionLoop
from defect_inspector.profile import Profile
from defect_inspector.scene import SceneGate

TABLE = np.full((64, 64, 3), 220, np.uint8)
PIECE = TABLE.copy()
PIECE[16:48, 16:48] = 40
OTHER = TABLE.copy()
OTHER[4:36, 28:60] = 40
STATE_FIELDS = {"status", "label", "score", "threshold", "latency_ms", "profile_id", "frame_id", "message"}


def make_profile(profile_id="nut", threshold=0.5):
    return Profile(profile_id, profile_id, None, 32, threshold, [0.4] * 3, [0.2] * 3, 4, None, None, {})


class FakeCamera:
    def __init__(self, *frames):
        self.frames = itertools.cycle(frames)
        self.lost = False
        self.ids = itertools.count(1)

    def show(self, *frames):
        self.frames = itertools.cycle(frames)

    def lose(self):
        self.lost = True

    def recover(self):
        self.lost = False

    def is_lost(self):
        return self.lost

    def latest(self):
        return None if self.lost else (next(self.frames), next(self.ids))

    def start(self):
        pass

    def stop(self):
        pass


class FakeBackend:
    def __init__(self, peak=0.1):
        self.peak = peak
        self.calls = 0
        self.broken = False
        self.during_run = None

    def heal(self):
        self.broken = False

    def run(self, x):
        self.calls += 1
        if self.during_run:
            self.during_run()
        if self.broken:
            raise RuntimeError("model failed")
        return np.full((1, 1, 56, 56), self.peak, np.float32)


class Rig:
    """The loop with its fake camera and backend at hand; the tests drive it with step()."""

    def __init__(self, frame=PIECE, peak=0.1, gate=True):
        self.camera = FakeCamera(frame)
        self.backend = FakeBackend(peak)
        self.profile = make_profile()
        self.loop = InspectionLoop(self.camera)
        self.loop.set_inspector(Inspector(self.profile, self.backend), self.profile,
                                SceneGate(TABLE) if gate else None)

    def step(self):
        self.loop.step()

    def state(self):
        return self.loop.state()

    def settle(self):
        for _ in range(10):
            self.step()

    step_until_stable = settle


def show_empty(rig):
    rig.camera.show(TABLE)


def show_other_piece(rig):
    rig.camera.show(OTHER)


def settle(rig):
    rig.settle()


@pytest.fixture
def loop_on_empty_table():
    return Rig(frame=TABLE)


@pytest.fixture
def loop_with_changing_frames():
    rig = Rig()
    rig.camera.show(PIECE, OTHER)
    return rig


@pytest.fixture
def loop_with_still_piece():
    return Rig()


@pytest.fixture
def loop_with_failing_backend():
    rig = Rig()
    rig.backend.broken = True
    return rig


@pytest.fixture
def loop_without_profile():
    return InspectionLoop(FakeCamera(PIECE))


@pytest.fixture
def other():
    profile = make_profile("washer")
    return Inspector(profile, FakeBackend()), profile, SceneGate(TABLE)


def test_empty_table_gives_no_verdict(loop_on_empty_table):
    loop_on_empty_table.step()
    assert loop_on_empty_table.state()["status"] == "EMPTY"


def test_moving_hand_gives_no_verdict(loop_with_changing_frames):
    for _ in range(10):
        loop_with_changing_frames.step()
    assert loop_with_changing_frames.state()["status"] == "MOVING"
    assert loop_with_changing_frames.backend.calls == 0


def test_still_piece_is_inspected_exactly_once(loop_with_still_piece):
    for _ in range(20):
        loop_with_still_piece.step()
    assert loop_with_still_piece.state()["status"] == "RESULT"
    assert loop_with_still_piece.backend.calls == 1


def test_result_reports_every_number(loop_with_still_piece):
    settle(loop_with_still_piece)
    state = loop_with_still_piece.state()
    assert state["label"] == "OK" and state["score"] == pytest.approx(0.1)
    assert state["threshold"] == 0.5 and state["latency_ms"] >= 0 and state["profile_id"] == "nut"


def test_state_always_has_every_field(loop_without_profile, loop_with_still_piece):
    assert set(loop_without_profile.state()) == STATE_FIELDS
    settle(loop_with_still_piece)
    assert set(loop_with_still_piece.state()) == STATE_FIELDS


def test_removing_piece_clears_result(loop_with_still_piece):
    settle(loop_with_still_piece)
    show_empty(loop_with_still_piece)
    settle(loop_with_still_piece)
    assert loop_with_still_piece.state()["status"] == "EMPTY"
    assert loop_with_still_piece.state()["label"] is None


def test_touching_the_piece_clears_result(loop_with_still_piece):
    settle(loop_with_still_piece)
    show_other_piece(loop_with_still_piece)
    loop_with_still_piece.step()
    assert loop_with_still_piece.state()["status"] == "MOVING"
    assert loop_with_still_piece.state()["label"] is None


def test_swapping_piece_directly_gives_new_verdict(loop_with_still_piece):
    settle(loop_with_still_piece)
    show_other_piece(loop_with_still_piece)
    settle(loop_with_still_piece)
    assert loop_with_still_piece.backend.calls == 2


def test_camera_lost_is_reported_not_frozen(loop_with_still_piece):
    settle(loop_with_still_piece)
    loop_with_still_piece.camera.lose()
    loop_with_still_piece.step()
    state = loop_with_still_piece.state()
    assert state["status"] == "CAMERA_LOST" and state["label"] is None and state["message"]
    assert loop_with_still_piece.loop.jpeg() is None


def test_camera_coming_back_gives_a_new_verdict(loop_with_still_piece):
    settle(loop_with_still_piece)
    loop_with_still_piece.camera.lose()
    loop_with_still_piece.step()
    loop_with_still_piece.camera.recover()
    settle(loop_with_still_piece)
    assert loop_with_still_piece.state()["status"] == "RESULT"
    assert loop_with_still_piece.backend.calls == 2


def test_backend_error_is_reported_and_loop_survives(loop_with_failing_backend):
    loop_with_failing_backend.step_until_stable()
    state = loop_with_failing_backend.state()
    assert state["status"] == "ERROR" and "model failed" in state["message"]
    assert loop_with_failing_backend.backend.calls == 1
    loop_with_failing_backend.backend.heal()
    show_other_piece(loop_with_failing_backend)
    loop_with_failing_backend.step_until_stable()
    assert loop_with_failing_backend.state()["status"] == "RESULT"


def test_switching_profile_clears_result(loop_with_still_piece, other):
    settle(loop_with_still_piece)
    loop_with_still_piece.loop.set_inspector(*other)
    assert loop_with_still_piece.state()["profile_id"] == other[1].id
    assert loop_with_still_piece.state()["label"] is None


def test_switching_profile_during_inference_drops_the_old_verdict(loop_with_still_piece, other):
    rig = loop_with_still_piece
    rig.backend.peak = 0.9
    rig.backend.during_run = lambda: rig.loop.set_inspector(*other)
    rig.settle()
    assert rig.state()["profile_id"] == "washer" and rig.backend.calls == 1
    assert rig.state()["label"] is None
    rig.settle()
    assert rig.state()["label"] == "OK" and other[0].backend.calls == 1


def test_no_profile_is_reported(loop_without_profile):
    loop_without_profile.step()
    assert loop_without_profile.state()["status"] == "NO_PROFILE"


def test_without_a_gate_every_new_frame_is_inspected():
    rig = Rig(gate=False)
    rig.step()
    rig.step()
    assert rig.backend.calls == 2 and rig.state()["status"] == "RESULT"


def test_without_a_gate_a_repeated_frame_is_not_inspected_again():
    rig = Rig(gate=False)
    rig.camera.latest = lambda: (PIECE, 7)
    rig.step()
    rig.step()
    assert rig.backend.calls == 1


def test_status_is_moving_while_the_model_works():
    rig = Rig(gate=False)
    seen = []
    rig.backend.during_run = lambda: seen.append((rig.state()["status"], rig.state()["label"]))
    rig.step()
    rig.step()
    assert seen == [("MOVING", None), ("MOVING", None)]


def test_message_says_when_the_model_is_working(loop_with_still_piece):
    rig = loop_with_still_piece
    seen = []
    rig.backend.during_run = lambda: seen.append(rig.state()["message"])
    waiting = []
    for _ in range(3):
        rig.step()
        waiting.append(rig.state()["message"])
    rig.settle()
    assert seen == ["Analizez piesa."]
    assert set(waiting) == {"Aștept să stea nemișcată piesa."}


def test_new_table_photo_replaces_the_gate(loop_with_still_piece):
    settle(loop_with_still_piece)
    loop_with_still_piece.loop.set_background(PIECE)
    loop_with_still_piece.step()
    assert loop_with_still_piece.state()["status"] == "EMPTY"
    assert loop_with_still_piece.state()["label"] is None


def test_latest_frame_is_the_camera_frame_or_nothing(loop_with_still_piece):
    assert loop_with_still_piece.loop.latest_frame() is PIECE
    loop_with_still_piece.camera.lose()
    assert loop_with_still_piece.loop.latest_frame() is None


def test_new_table_photo_without_a_profile_is_refused(loop_without_profile):
    with pytest.raises(ValueError, match="profile"):
        loop_without_profile.set_background(PIECE)


def test_configure_camera_reaches_the_camera(loop_with_still_piece):
    seen = []
    loop_with_still_piece.camera.configure = seen.append
    loop_with_still_piece.loop.configure_camera("settings")
    assert seen == ["settings"]


def decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def is_reddish(pixel):
    return int(pixel[2]) > int(pixel[0]) + 40


def test_jpeg_colors_the_zone_of_a_defect():
    rig = Rig(peak=0.9)
    rig.settle()
    assert rig.state()["label"] == "DEFECT"
    assert is_reddish(decode(rig.loop.jpeg())[32, 32])


def test_jpeg_of_a_good_piece_has_no_color():
    rig = Rig(peak=0.1)
    rig.settle()
    assert not is_reddish(decode(rig.loop.jpeg())[32, 32])


def test_jpeg_uses_the_newest_camera_frame_while_the_model_works():
    rig = Rig(gate=False)
    frames = []
    rig.backend.during_run = lambda: frames.append(decode(rig.loop.jpeg()))
    rig.camera.show(PIECE, OTHER)
    rig.step()
    assert np.abs(frames[0].astype(int) - PIECE.astype(int)).mean() > 5


def test_jpeg_is_none_before_the_first_frame():
    camera = FakeCamera(PIECE)
    camera.lose()
    assert InspectionLoop(camera).jpeg() is None


def test_error_in_a_zone_that_does_not_fit_is_reported():
    rig = Rig()
    rig.profile = rig.profile._replace(roi=[60, 60, 30, 30])
    rig.loop.set_inspector(Inspector(rig.profile, rig.backend), rig.profile, SceneGate(TABLE))
    rig.step()
    state = rig.state()
    assert state["status"] == "ERROR" and "roi" in state["message"]


def test_loop_thread_reaches_a_result_and_stops():
    rig = Rig()
    rig.loop.start()
    try:
        for _ in range(500):
            if rig.state()["status"] == "RESULT":
                break
            time.sleep(0.01)
        assert rig.state()["status"] == "RESULT"
    finally:
        rig.loop.stop()
    assert not rig.loop._thread.is_alive()
