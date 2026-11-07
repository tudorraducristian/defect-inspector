import pytest

from live_trial import rates_from_sequence, record_verdicts


def result(label):
    return {"status": "RESULT", "label": label}


MOVING = {"status": "MOVING", "label": None}
EMPTY = {"status": "EMPTY", "label": None}


def run(states, total):
    feed = iter(states)
    seen = []
    labels = record_verdicts(lambda: next(feed), total, seen.append, pause=lambda: None)
    assert labels == seen
    return labels


def test_rates_from_sequence():
    r = rates_from_sequence(["OK", "DEFECT", "DEFECT", "DEFECT", "OK", "DEFECT"], good_count=2)
    assert r == {"false_alarm_rate": 0.5, "detected_rate": 0.75}


def test_wrong_number_of_good_pieces_is_refused():
    with pytest.raises(ValueError):
        rates_from_sequence(["OK"], good_count=1)


def test_one_verdict_stays_one_however_long_it_is_read():
    assert run([MOVING, result("OK"), result("OK"), result("OK"), EMPTY, result("DEFECT")], 2) == ["OK", "DEFECT"]


def test_a_new_piece_after_moving_counts_again():
    assert run([result("OK"), MOVING, result("OK")], 2) == ["OK", "OK"]


def test_other_states_are_not_verdicts():
    errors = [{"status": "ERROR", "label": None}, {"status": "CAMERA_LOST", "label": None}]
    assert run(errors + [result("DEFECT")], 1) == ["DEFECT"]


def test_it_stops_at_the_total():
    assert run([result("OK"), EMPTY, result("OK"), EMPTY, result("OK")], 2) == ["OK", "OK"]
