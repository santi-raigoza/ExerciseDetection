# Exercise Detection App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time exercise detection app that classifies pushup, pullup, squat, and jumping jack from webcam pose data and counts reps.

**Architecture:** A three-phase pipeline — `collect.py` records labeled pose windows to CSV, `train.py` trains a sliding-window MLP classifier offline, and `app.py` runs real-time inference with a per-exercise angle-based rep counter.

**Tech Stack:** Python 3.13, MediaPipe Tasks API (PoseLandmarker), OpenCV, scikit-learn (MLPClassifier, StandardScaler), NumPy, pandas, joblib, pytest

---

## File Map

| Path | Responsibility |
|---|---|
| `src/utils/landmarks.py` | `normalize()`, `build_window()`, `POSE_CONNECTIONS` |
| `src/utils/rep_counter.py` | `RepCounter` state machine for all 4 exercises |
| `src/collect.py` | Webcam data collection, `save_windows()` |
| `src/train.py` | `load_data()`, `train()` — offline training pipeline |
| `src/app.py` | Real-time inference loop, overlay rendering |
| `tests/utils/test_landmarks.py` | Unit tests for normalize + build_window |
| `tests/utils/test_rep_counter.py` | Unit tests for RepCounter state machine |
| `tests/test_train.py` | Unit tests for load_data + train |
| `tests/test_collect.py` | Unit test for save_windows |
| `conftest.py` | Adds `src/` to sys.path so tests can import from `src/utils/` |

---

## Task 1: Project Scaffold

**Files:**
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/utils/__init__.py`
- Create: `conftest.py`
- Create: `data/raw/.gitkeep`
- Create: `models/.gitkeep`
- Modify: `.gitignore`
- Modify: `requirements.txt`

- [ ] **Step 1: Create directory structure and empty init files**

```bash
mkdir -p src/utils tests/utils data/raw models
touch src/__init__.py src/utils/__init__.py tests/__init__.py tests/utils/__init__.py
touch data/raw/.gitkeep models/.gitkeep
```

- [ ] **Step 2: Create `conftest.py`** (makes `src/` importable in all tests)

```python
# conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
data/raw/*.csv
models/*.pkl
.DS_Store
```

- [ ] **Step 4: Add `pytest` to `requirements.txt`**

Add this line to the `# Core` section of `requirements.txt`:
```
pytest==8.4.0
```

Then install:
```bash
.venv/bin/pip install pytest==8.4.0
```

Expected output: `Successfully installed pytest-8.4.0`

- [ ] **Step 5: Commit**

```bash
git add src/ tests/ conftest.py data/raw/.gitkeep models/.gitkeep .gitignore requirements.txt
git commit -m "chore: scaffold project structure"
```

---

## Task 2: `utils/landmarks.py` — Normalization and Window Building

**Files:**
- Create: `src/utils/landmarks.py`
- Create: `tests/utils/test_landmarks.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/utils/test_landmarks.py
import numpy as np
import pytest
from utils.landmarks import normalize, build_window


class MockLandmark:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


def make_33(overrides=None):
    """33 landmarks all at origin, with optional index overrides: {idx: (x,y,z)}."""
    lms = [MockLandmark() for _ in range(33)]
    if overrides:
        for i, (x, y, z) in overrides.items():
            lms[i] = MockLandmark(x, y, z)
    return lms


def test_normalize_returns_shape_99():
    result = normalize(make_33())
    assert result.shape == (99,)


def test_normalize_centers_hip_midpoint_at_origin():
    # Left hip (23), right hip (24), shoulders (11, 12) above hips
    lms = make_33({
        23: (0.4, 0.6, 0.0),  # left hip
        24: (0.6, 0.6, 0.0),  # right hip
        11: (0.4, 0.4, 0.0),  # left shoulder
        12: (0.6, 0.4, 0.0),  # right shoulder
    })
    result = normalize(lms)
    reshaped = result.reshape(33, 3)
    hip_mid_normalized = (reshaped[23] + reshaped[24]) / 2.0
    np.testing.assert_allclose(hip_mid_normalized, [0.0, 0.0, 0.0], atol=1e-6)


def test_normalize_scales_so_shoulder_midpoint_is_at_minus_one():
    # Hips at y=0.6, shoulders at y=0.4 → torso_height = 0.2
    # After normalization, shoulder midpoint should be at y = -0.2/0.2 = -1.0
    lms = make_33({
        23: (0.5, 0.6, 0.0),
        24: (0.5, 0.6, 0.0),
        11: (0.5, 0.4, 0.0),
        12: (0.5, 0.4, 0.0),
    })
    result = normalize(lms)
    reshaped = result.reshape(33, 3)
    shoulder_mid = (reshaped[11] + reshaped[12]) / 2.0
    np.testing.assert_allclose(shoulder_mid[:2], [0.0, -1.0], atol=1e-6)


def test_normalize_no_crash_when_torso_height_is_zero():
    # All landmarks at origin — torso height is 0, should not raise
    result = normalize(make_33())
    assert result.shape == (99,)
    assert not np.any(np.isnan(result))


def test_build_window_returns_shape_2970():
    frames = [np.ones(99) for _ in range(30)]
    assert build_window(frames).shape == (2970,)


def test_build_window_preserves_frame_order():
    # Frame i is all i's — after concatenation frame 0 starts at index 0,
    # frame 1 at index 99, frame 29 at index 2871
    frames = [np.full(99, float(i)) for i in range(30)]
    result = build_window(frames)
    assert result[0] == 0.0
    assert result[99] == 1.0
    assert result[2871] == 29.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/utils/test_landmarks.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.landmarks'`

- [ ] **Step 3: Implement `src/utils/landmarks.py`**

```python
# src/utils/landmarks.py
import numpy as np

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def normalize(landmarks):
    """
    landmarks: list of 33 objects with .x, .y, .z (MediaPipe NormalizedLandmark)
    Returns: np.ndarray of shape (99,) — translated to hip midpoint, scaled by torso height
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    hip_mid = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2.0
    shoulder_mid = (coords[LEFT_SHOULDER] + coords[RIGHT_SHOULDER]) / 2.0
    torso_height = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_height < 1e-6:
        torso_height = 1.0
    return ((coords - hip_mid) / torso_height).flatten()


def build_window(frame_buffer):
    """
    frame_buffer: list or deque of 30 np.ndarray(99,)
    Returns: np.ndarray of shape (2970,) — frames concatenated in order
    """
    return np.concatenate(list(frame_buffer))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/utils/test_landmarks.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils/landmarks.py tests/utils/test_landmarks.py
git commit -m "feat: add landmark normalization and window building"
```

---

## Task 3: `utils/rep_counter.py` — Per-Exercise State Machine

**Files:**
- Create: `src/utils/rep_counter.py`
- Create: `tests/utils/test_rep_counter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/utils/test_rep_counter.py
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
# Jumping jack: wrist spread — closed=wrists near hip x, open=wrists far apart
JACKS_CLOSED = {15: (0.45, 0.7, 0), 16: (0.55, 0.7, 0)}  # spread = 0.10 < 0.4
JACKS_OPEN   = {15: (0.1, 0.3, 0),  16: (0.9, 0.3, 0)}   # spread = 0.80 > 0.6


def send(rc, exercise, lms, n=5, confidence=0.9):
    for _ in range(n):
        rc.update(exercise, confidence, lms)


def test_pushup_counts_one_rep():
    rc = RepCounter()
    send(rc, 'pushup', make_33(BENT_ARM))   # go down
    assert rc.count == 0
    send(rc, 'pushup', make_33(STRAIGHT_ARM))  # come up → 1 rep
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
    send(rc, 'jumping_jack', make_33(JACKS_OPEN))   # open
    assert rc.count == 0
    send(rc, 'jumping_jack', make_33(JACKS_CLOSED))  # close → 1 rep
    assert rc.count == 1


def test_pullup_counts_one_rep():
    # Pullup: starts hanging (arms straight/down=STRAIGHT_ARM angle>160),
    # pulls up (arms bent/BENT_ARM angle<90), then hangs again → 1 rep
    rc = RepCounter()
    send(rc, 'pullup', make_33(BENT_ARM))    # chin up (elbows bent)
    assert rc.count == 0
    send(rc, 'pullup', make_33(STRAIGHT_ARM))  # hang (elbows straight) → 1 rep
    assert rc.count == 1


def test_rest_does_not_count():
    rc = RepCounter()
    send(rc, 'rest', make_33(BENT_ARM))
    send(rc, 'rest', make_33(STRAIGHT_ARM))
    assert rc.count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/utils/test_rep_counter.py -v
```

Expected: `ModuleNotFoundError: No module named 'utils.rep_counter'`

- [ ] **Step 3: Implement `src/utils/rep_counter.py`**

```python
# src/utils/rep_counter.py
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
        self._state = None   # 'up' or 'down'
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
            # pushup/squat start extended (up); pullup/jumping_jack start at rest (down)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/utils/test_rep_counter.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add src/utils/rep_counter.py tests/utils/test_rep_counter.py
git commit -m "feat: add per-exercise rep counter state machine"
```

---

## Task 4: `collect.py` — Data Collection

**Files:**
- Create: `src/collect.py`
- Create: `tests/test_collect.py`

- [ ] **Step 1: Write failing test for `save_windows`**

```python
# tests/test_collect.py
import numpy as np
import pandas as pd
import pytest
from collect import save_windows


def test_save_windows_creates_csv_with_correct_shape(tmp_path):
    windows = [np.ones(2970), np.zeros(2970)]
    save_windows(windows, 'pushup', tmp_path)
    files = list(tmp_path.glob('pushup_*.csv'))
    assert len(files) == 1
    df = pd.read_csv(files[0])
    assert df.shape == (2, 2971)   # 2970 features + 1 label column
    assert list(df.columns[:2]) == ['label', 'feat_0']
    assert (df['label'] == 'pushup').all()


def test_save_windows_values_are_correct(tmp_path):
    windows = [np.ones(2970) * 3.14]
    save_windows(windows, 'squat', tmp_path)
    df = pd.read_csv(list(tmp_path.glob('squat_*.csv'))[0])
    np.testing.assert_allclose(df.iloc[0, 1:].values, 3.14, rtol=1e-5)


def test_save_windows_creates_output_dir_if_missing(tmp_path):
    subdir = tmp_path / 'new' / 'nested'
    windows = [np.ones(2970)]
    save_windows(windows, 'rest', subdir)
    assert subdir.exists()
    assert len(list(subdir.glob('rest_*.csv'))) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_collect.py -v
```

Expected: `ModuleNotFoundError: No module named 'collect'`

- [ ] **Step 3: Implement `src/collect.py`**

```python
# src/collect.py
import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from utils.landmarks import normalize, build_window, POSE_CONNECTIONS

WINDOW_SIZE = 30
STRIDE = 5
TASK_PATH = 'pose_landmarker_lite.task'
EXERCISES = ['pushup', 'pullup', 'squat', 'jumping_jack', 'rest']


def save_windows(windows, label, output_dir):
    """Save list of window arrays to a timestamped CSV. Returns the file path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = output_dir / f'{label}_{timestamp}.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'feat_{i}' for i in range(2970)])
        for w in windows:
            writer.writerow([label] + w.tolist())
    return path


def _draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='Collect exercise training data')
    parser.add_argument('--exercise', required=True, choices=EXERCISES)
    parser.add_argument('--samples', type=int, default=300)
    parser.add_argument('--output-dir', default='data/raw')
    args = parser.parse_args()

    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    windows = []
    recording = False
    frame_count = 0

    print(f"Exercise: {args.exercise} | Target: {args.samples} windows")
    print("SPACE = start/stop recording    Q = quit and save")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            _draw_skeleton(frame, landmarks)

            if recording:
                frame_buffer.append(normalize(landmarks))
                frame_count += 1
                if len(frame_buffer) == WINDOW_SIZE and frame_count % STRIDE == 0:
                    windows.append(build_window(frame_buffer))
                    if len(windows) >= args.samples:
                        recording = False
                        print(f"Target reached: {len(windows)} windows collected.")

        status = 'RECORDING' if recording else 'READY'
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f'{status}  {len(windows)}/{args.samples}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, args.exercise, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            recording = not recording
            if recording:
                frame_buffer.clear()
                frame_count = 0
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if windows:
        path = save_windows(windows, args.exercise, args.output_dir)
        print(f"Saved {len(windows)} windows → {path}")
    else:
        print("No windows recorded. Nothing saved.")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_collect.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/collect.py tests/test_collect.py
git commit -m "feat: add data collection script"
```

---

## Task 5: `train.py` — Training Pipeline

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train.py
import numpy as np
import pandas as pd
import joblib
import pytest
from train import load_data, train


def _write_csv(path, label, n=20, val=1.0):
    data = {'label': [label] * n}
    data.update({f'feat_{i}': [val + np.random.randn() * 0.01] * n for i in range(2970)})
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_data_returns_correct_shape(tmp_path):
    _write_csv(tmp_path / 'pushup_test.csv', 'pushup', n=10)
    X, y = load_data(tmp_path)
    assert X.shape == (10, 2970)
    assert y.shape == (10,)
    assert (y == 'pushup').all()


def test_load_data_merges_multiple_csvs(tmp_path):
    _write_csv(tmp_path / 'pushup_test.csv', 'pushup', n=10)
    _write_csv(tmp_path / 'rest_test.csv', 'rest', n=10)
    X, y = load_data(tmp_path)
    assert X.shape == (20, 2970)
    assert set(y) == {'pushup', 'rest'}


def test_load_data_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="No CSV files"):
        load_data(tmp_path)


def test_train_creates_model_and_scaler_files(tmp_path):
    data_dir = tmp_path / 'raw'
    data_dir.mkdir()
    models_dir = tmp_path / 'models'
    _write_csv(data_dir / 'pushup.csv', 'pushup', n=20, val=1.0)
    _write_csv(data_dir / 'rest.csv',   'rest',   n=20, val=-1.0)
    train(data_dir, models_dir, hidden_layer_sizes=(8,), max_iter=50)
    assert (models_dir / 'model.pkl').exists()
    assert (models_dir / 'scaler.pkl').exists()


def test_trained_model_predicts_separable_classes(tmp_path):
    data_dir = tmp_path / 'raw'
    data_dir.mkdir()
    models_dir = tmp_path / 'models'
    # Clearly separable: pushup=all 1s, rest=all -1s
    _write_csv(data_dir / 'pushup.csv', 'pushup', n=30, val=1.0)
    _write_csv(data_dir / 'rest.csv',   'rest',   n=30, val=-1.0)
    train(data_dir, models_dir, hidden_layer_sizes=(8,), max_iter=100)

    model = joblib.load(models_dir / 'model.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')

    X_pushup = scaler.transform([np.ones(2970)])
    X_rest   = scaler.transform([-np.ones(2970)])
    assert model.predict(X_pushup)[0] == 'pushup'
    assert model.predict(X_rest)[0] == 'rest'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'train'`

- [ ] **Step 3: Implement `src/train.py`**

```python
# src/train.py
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_data(data_dir):
    """Load all CSVs from data_dir. Returns X (n, 2970) and y (n,) as numpy arrays."""
    csvs = list(Path(data_dir).glob('*.csv'))
    if not csvs:
        raise ValueError(f"No CSV files found in {data_dir}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    X = df.drop(columns=['label']).values.astype(np.float32)
    y = df['label'].values
    return X, y


def train(data_dir, models_dir, **mlp_kwargs):
    """
    Train MLP on windows in data_dir, save model + scaler to models_dir.
    mlp_kwargs are forwarded to MLPClassifier (useful for overriding in tests).
    """
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    params = dict(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    params.update(mlp_kwargs)
    model = MLPClassifier(**params)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  Path(models_dir) / 'model.pkl')
    joblib.dump(scaler, Path(models_dir) / 'scaler.pkl')
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description='Train exercise classifier')
    parser.add_argument('--data-dir',   default='data/raw')
    parser.add_argument('--models-dir', default='models')
    args = parser.parse_args()
    train(args.data_dir, args.models_dir)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass** (may take 10–20 seconds for MLP training)

```bash
.venv/bin/pytest tests/test_train.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add training pipeline"
```

---

## Task 6: `app.py` — Real-Time Inference

This file has no unit tests — it requires a trained model, a webcam, and a live display window. Verification is manual (run it and confirm the overlay appears correctly).

**Files:**
- Create: `src/app.py`

- [ ] **Step 1: Implement `src/app.py`**

```python
# src/app.py
import time
from collections import deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from utils.landmarks import normalize, build_window, POSE_CONNECTIONS
from utils.rep_counter import RepCounter

WINDOW_SIZE = 30
MODEL_PATH  = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'
TASK_PATH   = 'pose_landmarker_lite.task'


def _draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)


def _draw_overlay(frame, exercise, confidence, count, fps):
    h, w, _ = frame.shape
    cv2.putText(frame, f'{exercise}  {confidence:.0%}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    label = f'Reps: {count}'
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(frame, label, (w - tw - 10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(frame, f'FPS {fps:.0f}', (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    rep_counter  = RepCounter()
    exercise, confidence = 'rest', 0.0
    prev_time = time.time()

    print('Press Q to quit.')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            _draw_skeleton(frame, landmarks)
            frame_buffer.append(normalize(landmarks))

            if len(frame_buffer) == WINDOW_SIZE:
                window_s = scaler.transform([build_window(frame_buffer)])
                probs    = model.predict_proba(window_s)[0]
                idx      = int(np.argmax(probs))
                exercise    = model.classes_[idx]
                confidence  = float(probs[idx])

            rep_counter.update(exercise, confidence, landmarks)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        _draw_overlay(frame, exercise, confidence, rep_counter.count, fps)
        cv2.imshow('Exercise Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the full test suite to confirm nothing is broken**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all previous tests still pass. No failures.

- [ ] **Step 3: Collect training data (manual — do this for each exercise)**

Run once per exercise label. Stand 6–8 feet from the webcam so your full body is visible. Press SPACE to start, perform the exercise continuously, press SPACE to stop or wait for the target count.

```bash
cd src
../.venv/bin/python collect.py --exercise rest         --samples 300
../.venv/bin/python collect.py --exercise pushup       --samples 300
../.venv/bin/python collect.py --exercise squat        --samples 300
../.venv/bin/python collect.py --exercise jumping_jack --samples 300
../.venv/bin/python collect.py --exercise pullup       --samples 300
```

Each session saves a CSV to `data/raw/`. Verify files exist:

```bash
ls ../data/raw/
```

Expected: one `.csv` per exercise, each ~50–60 KB.

- [ ] **Step 4: Train the model**

```bash
../.venv/bin/python train.py
```

Expected output ends with a confusion matrix and line like:
```
accuracy    0.94   1500
```
Accuracy above 0.85 is a good starting point. If lower, record more data for the underperforming class (check the classification report for which class has low recall).

Verify model files exist:
```bash
ls ../models/
```

Expected: `model.pkl  scaler.pkl`

- [ ] **Step 5: Run the live app and verify the overlay**

```bash
../.venv/bin/python app.py
```

Stand in front of the webcam and verify:
- Skeleton overlay draws on your body
- Exercise label updates when you change exercises
- Rep count increments correctly (do 5 pushups, count should reach 5)
- FPS displays in bottom-left corner
- Press Q to quit

- [ ] **Step 6: Commit**

```bash
cd ..
git add src/app.py
git commit -m "feat: add real-time inference app"
```

---

## Usage Summary (after all tasks complete)

```bash
# 1. Collect data for each exercise (repeat for all 5 labels)
source .venv/bin/activate
cd src
python collect.py --exercise pushup --samples 300

# 2. Train
python train.py

# 3. Run
python app.py
```
