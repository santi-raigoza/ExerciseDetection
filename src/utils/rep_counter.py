from collections import deque

import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

SMOOTH_N = 5
CONFIDENCE_THRESHOLD = 0.7


def _angle(a, b, c):
    """Angle in degrees at point b, formed by vectors b→a and b→c."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class RepCounter:
    def __init__(self):
        self._exercise = None
        self._count = 0
        self._state = None
        self._history = deque(maxlen=SMOOTH_N)

    def _metric(self, exercise, coords):
        if exercise in ('pushup', 'pullup'):
            left = _angle(coords[LEFT_SHOULDER], coords[LEFT_ELBOW], coords[LEFT_WRIST])
            right = _angle(coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW], coords[RIGHT_WRIST])
            return (left + right) / 2.0
        if exercise == 'squat':
            left = _angle(coords[LEFT_HIP], coords[LEFT_KNEE], coords[LEFT_ANKLE])
            right = _angle(coords[RIGHT_HIP], coords[RIGHT_KNEE], coords[RIGHT_ANKLE])
            return (left + right) / 2.0
        if exercise == 'jumping_jack':
            return abs(coords[LEFT_WRIST][0] - coords[RIGHT_WRIST][0])
        return 0.0

    def update(self, exercise, confidence, landmarks):
        """
        exercise:   predicted class string
        confidence: float in [0, 1]
        landmarks:  list of 33 objects with .x .y .z
        Returns:    current rep count (int)
        """
        if exercise == 'rest' or confidence < CONFIDENCE_THRESHOLD:
            return self._count

        if exercise != self._exercise:
            self._exercise = exercise
            self._count = 0
            self._state = 'up' if exercise in ('pushup', 'squat') else 'down'
            self._history.clear()

        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        self._history.append(self._metric(exercise, coords))
        smoothed = float(np.mean(self._history))

        if exercise in ('pushup', 'squat'):
            if self._state == 'up' and smoothed < 90:
                self._state = 'down'
            elif self._state == 'down' and smoothed > 160:
                self._state = 'up'
                self._count += 1

        elif exercise == 'pullup':
            if self._state == 'down' and smoothed < 90:
                self._state = 'up'
            elif self._state == 'up' and smoothed > 160:
                self._state = 'down'
                self._count += 1

        elif exercise == 'jumping_jack':
            if self._state == 'up' and smoothed < 0.4:
                self._state = 'down'
                self._count += 1
            elif self._state == 'down' and smoothed > 0.6:
                self._state = 'up'

        return self._count

    @property
    def count(self):
        return self._count
