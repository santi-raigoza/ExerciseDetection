import numpy as np
import pytest
from utils.rep_counter import RepCounter


class MockLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def make_33(overrides=None):
    lms = [MockLandmark() for _ in range(33)]
    if overrides:
        for i, (x, y, z) in overrides.items():
            lms[i] = MockLandmark(x, y, z)
    return lms


# Pushup/squat: elbow/knee angle ~180° = arm/leg straight = "up"
# shoulder(11,12)=(0,0.1,0), elbow(13,14)=(0,0,0), wrist(15,16)=(0,-0.1,0)
STRAIGHT_ARM = {
    11: (0, 0.1, 0), 12: (0, 0.1, 0),
    13: (0, 0, 0),   14: (0, 0, 0),
    15: (0, -0.1, 0), 16: (0, -0.1, 0),
}
# Elbow angle ~63° = arm bent = "down" for pushup
# ba=(0,0.1,0), bc=(0.1,0.05,0) → cosine≈0.447 → ~63°
BENT_ARM = {
    11: (0, 0.1, 0), 12: (0, 0.1, 0),
    13: (0, 0, 0),   14: (0, 0, 0),
    15: (0.1, 0.05, 0), 16: (0.1, 0.05, 0),
}
# Squat: hip(23,24), knee(25,26), ankle(27,28)
STRAIGHT_LEG = {
    23: (0, 0.4, 0), 24: (0, 0.4, 0),
    25: (0, 0.6, 0), 26: (0, 0.6, 0),
    27: (0, 0.8, 0), 28: (0, 0.8, 0),
}
BENT_LEG = {
    23: (0, 0.4, 0), 24: (0, 0.4, 0),
    25: (0, 0.6, 0), 26: (0, 0.6, 0),
    27: (0.1, 0.55, 0), 28: (0.1, 0.55, 0),
}
# Jumping jack: combined wrist+ankle spread normalized by shoulder width
# shoulder_width = 0.2, so ratio = spread / 0.2
JACKS_CLOSED = {
    11: (0.4, 0.3, 0), 12: (0.6, 0.3, 0),   # shoulders (width=0.2)
    15: (0.45, 0.7, 0), 16: (0.55, 0.7, 0),  # wrists close  → ratio=0.5
    27: (0.47, 0.9, 0), 28: (0.53, 0.9, 0),  # ankles close  → ratio=0.3  min=0.3 < 0.7
}
JACKS_OPEN = {
    11: (0.4, 0.3, 0), 12: (0.6, 0.3, 0),   # shoulders (width=0.2)
    15: (0.1, 0.3, 0),  16: (0.9, 0.3, 0),  # wrists spread → ratio=4.0
    27: (0.2, 0.9, 0),  28: (0.8, 0.9, 0),  # ankles spread → ratio=3.0  min=3.0 > 1.3
}


def send(rc, exercise, lms, n=5, confidence=0.9):
    for _ in range(n):
        rc.update(exercise, confidence, lms)


def test_pushup_counts_one_rep():
    rc = RepCounter()
    send(rc, 'pushup', make_33(BENT_ARM))
    assert rc.count == 0
    send(rc, 'pushup', make_33(STRAIGHT_ARM))
    assert rc.count == 1


def test_pushup_two_reps():
    rc = RepCounter()
    send(rc, 'pushup', make_33(BENT_ARM))
    send(rc, 'pushup', make_33(STRAIGHT_ARM))
    send(rc, 'pushup', make_33(BENT_ARM))
    send(rc, 'pushup', make_33(STRAIGHT_ARM))
    assert rc.count == 2


def test_no_rep_when_confidence_below_threshold():
    rc = RepCounter()
    send(rc, 'pushup', make_33(BENT_ARM), confidence=0.5)
    send(rc, 'pushup', make_33(STRAIGHT_ARM), confidence=0.5)
    assert rc.count == 0


def test_rep_count_resets_on_exercise_change():
    rc = RepCounter()
    send(rc, 'pushup', make_33(BENT_ARM))
    send(rc, 'pushup', make_33(STRAIGHT_ARM))
    assert rc.count == 1
    rc.update('squat', 0.9, make_33())
    assert rc.count == 0


def test_squat_counts_one_rep():
    rc = RepCounter()
    send(rc, 'squat', make_33(BENT_LEG))
    assert rc.count == 0
    send(rc, 'squat', make_33(STRAIGHT_LEG))
    assert rc.count == 1


def test_jumping_jack_counts_one_rep():
    rc = RepCounter()
    send(rc, 'jumping_jack', make_33(JACKS_OPEN))
    assert rc.count == 0
    send(rc, 'jumping_jack', make_33(JACKS_CLOSED))
    assert rc.count == 1


def test_pullup_counts_one_rep():
    rc = RepCounter()
    send(rc, 'pullup', make_33(BENT_ARM))
    assert rc.count == 0
    send(rc, 'pullup', make_33(STRAIGHT_ARM))
    assert rc.count == 1


def test_rest_does_not_count():
    rc = RepCounter()
    send(rc, 'rest', make_33(BENT_ARM))
    send(rc, 'rest', make_33(STRAIGHT_ARM))
    assert rc.count == 0
