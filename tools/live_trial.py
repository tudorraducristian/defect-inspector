"""Measure the station on real pieces: read one verdict per piece from the running server.

Put the good pieces first, then the defective ones, taking each away after its verdict. Prints
the two rates at the end. Runs on Python 3.6.
"""
import argparse
import json
import sys
import time
import urllib.request

POLL_S = 0.2


def rates_from_sequence(labels, good_count):
    """The first good_count labels belong to good pieces, the others to defective ones."""
    if not 0 < good_count < len(labels):
        raise ValueError("need both good and defective pieces, got {} of {}".format(good_count, len(labels)))
    good, defect = labels[:good_count], labels[good_count:]
    return {
        "false_alarm_rate": good.count("DEFECT") / len(good),
        "detected_rate": defect.count("DEFECT") / len(defect),
    }


def record_verdicts(get_state, total, on_verdict, pause=None):
    """Return one label per piece; a verdict counts once, however many times it is read."""
    pause = pause or (lambda: time.sleep(POLL_S))
    labels = []
    showing = False
    while len(labels) < total:
        state = get_state()
        if state["status"] != "RESULT":
            showing = False
        elif not showing:
            showing = True
            labels.append(state["label"])
            on_verdict(state["label"])
        pause()
    return labels


def _read_state(url):
    with urllib.request.urlopen(url + "/api/state", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--good", type=int, required=True)
    parser.add_argument("--defect", type=int, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args(argv)
    total = args.good + args.defect

    def report(label):
        report.count += 1
        kind = "buna" if report.count <= args.good else "stricata"
        print("{} din {}, piesa {}: {}".format(report.count, total, kind, label), flush=True)

    report.count = 0
    labels = record_verdicts(lambda: _read_state(args.url), total, report)
    print(json.dumps(dict(rates_from_sequence(labels, args.good), labels=labels), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
