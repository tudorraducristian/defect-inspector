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
    assert run([MOVING, result("OK"), result("OK"), result("OK"), EMPTY, result("DEFECT"), EMPTY], 2) == ["OK", "DEFECT"]


def test_a_piece_counts_once_with_its_last_verdict_before_the_table_is_empty():
    # A verdict taken while the hand was still in the picture is replaced by the one on the piece alone.
    states = [result("DEFECT"), MOVING, result("OK"), EMPTY, result("DEFECT"), EMPTY]
    assert run(states, 2) == ["OK", "DEFECT"]


def test_a_piece_taken_away_before_any_verdict_counts_nothing():
    assert run([MOVING, EMPTY, MOVING, result("OK"), EMPTY], 1) == ["OK"]


def test_other_states_are_not_verdicts():
    errors = [{"status": "ERROR", "label": None}, {"status": "CAMERA_LOST", "label": None}]
    assert run(errors + [result("DEFECT"), EMPTY], 1) == ["DEFECT"]


def test_it_stops_at_the_total():
    assert run([result("OK"), EMPTY, result("OK"), EMPTY, result("OK")], 2) == ["OK", "OK"]
