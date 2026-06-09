# Exercise Detection App — Design Spec

**Date:** 2026-06-09
**Status:** Approved

---

## Overview

A real-time exercise detection app that uses webcam pose data to classify which exercise the user is performing and count reps. The initial target exercises are pushup, pullup, squat, and jumping jack. The architecture is designed to extend to weightlifting exercises, soccer juggling, and conditioning drills (ladder drills, etc.) without restructuring.

---

## Architecture

Three standalone phases run in sequence:

```
Phase 1 — collect.py  : Webcam → MediaPipe → normalized landmarks → labeled CSV files
Phase 2 — train.py    : CSVs → sliding windows → MLP classifier → saved model (.pkl)
Phase 3 — app.py      : Webcam → MediaPipe → sliding window → model → exercise + rep count
```

The model file (`.pkl`) is the only artifact that crosses from training into the live app. Data collection and training are offline; only inference runs in real time.

---

## Project Structure

```
ExerciseDetection/
├── data/
│   └── raw/                  # one CSV per recording session
├── models/                   # saved MLP model + scaler (.pkl)
├── src/
│   ├── collect.py            # data collection
│   ├── train.py              # training pipeline
│   ├── app.py                # real-time inference
│   └── utils/
│       ├── landmarks.py      # normalization + feature extraction
│       └── rep_counter.py    # per-exercise state machine
├── pose_landmarker_lite.task
└── requirements.txt
```

---

## Data Model

### Features per frame
MediaPipe outputs 33 landmarks × (x, y, z) = 99 values per frame. Each frame is normalized:
- **Translation:** subtract the hip midpoint (average of left and right hip landmarks)
- **Scale:** divide by torso height (distance from hip midpoint to shoulder midpoint)

This makes features invariant to position in the frame and distance from the camera.

### Sliding window
- **Window size:** 30 consecutive normalized frames (~1 second at 30fps)
- **Flattened input:** 99 × 30 = 2,970 features per sample
- **Inference stride:** 1 frame (slides every frame in real time)
- **Training stride:** 5 frames (reduces redundancy in collected data)

### Class labels
```
pushup | pullup | squat | jumping_jack | rest
```
The `rest` class represents standing still or transitioning between exercises. It prevents the model from being forced to always predict an exercise.

### Data targets
- ~300 labeled windows per class (1,500 total)
- Collected by recording ~2–3 minutes of each exercise per session
- Generalization to other people requires adding their recordings to the training set and retraining

---

## Components

### `utils/landmarks.py`
- `normalize(landmarks, frame_shape)` — returns a 99-element normalized numpy array for one frame
- `build_window(frame_buffer)` — takes a deque of 30 normalized frames, returns a 2,970-element feature vector

### `utils/rep_counter.py`
Per-exercise state machine. Each exercise tracks one key joint metric:

| Exercise | Metric | Down state | Up state |
|---|---|---|---|
| Pushup | Elbow angle | < 90° | > 160° |
| Squat | Knee angle | < 90° | > 160° |
| Pullup | Elbow angle | > 160° (hanging) | < 90° (chin up) |
| Jumping Jack | Wrist-to-hip distance (normalized) | < 0.3 | > 0.6 |

**State machine rules:**
- Angle/distance is smoothed over a 5-frame rolling average to eliminate jitter
- Hysteresis: state only transitions when the value clearly crosses the threshold (not just touches it)
- Rep increments on DOWN → UP → DOWN (or UP → DOWN → UP for pullups)
- Reps only count when classifier confidence for the current exercise exceeds 0.7
- Rep count resets when the predicted exercise changes

### `collect.py`
- CLI args: `--exercise <name>` and `--samples <n>`
- Opens webcam with pose overlay
- User presses SPACE to start/stop recording
- Samples windows at stride 5 during active recording
- Saves to `data/raw/<exercise>_<timestamp>.csv`

### `train.py`
- Loads all CSVs from `data/raw/`
- Builds sliding windows with labels
- Normalizes with `StandardScaler` (fit on training set only, saved alongside model)
- Trains `sklearn.neural_network.MLPClassifier`
- Prints accuracy + confusion matrix on held-out test split (80/20)
- Saves model and scaler to `models/`

### `app.py`
- Loads model and scaler from `models/`
- Opens webcam, runs MediaPipe per frame
- Maintains a 30-frame deque; once full, classifies every frame
- Feeds prediction + landmarks into `RepCounter`
- Draws overlay: pose skeleton, exercise label + confidence (top-left), rep count (top-right), FPS (bottom)

---

## Display Overlay

- **Pose skeleton:** 33 green landmark dots, white connecting lines
- **Top-left:** predicted exercise name + confidence percentage
- **Top-right:** rep count for current exercise
- **Bottom:** FPS counter

---

## Extension Path

The sliding-window MLP is designed to accommodate future exercise categories:

- **Weightlifting:** Add labeled recordings; the window captures rep tempo differences between exercises that share similar static poses (e.g., deadlift vs. squat mid-rep)
- **Soccer juggling:** Temporal rhythm is captured by the window; may require a longer window (60 frames) to see full juggling cadence
- **Conditioning drills (ladder drills, etc.):** Same window approach; the MLP can be swapped for an LSTM at the same interface boundary if accuracy plateaus — `landmarks.py` and `rep_counter.py` are unaffected

---

## Out of Scope (v1)

- Multi-person detection
- Form feedback / injury detection
- Audio cues
- Workout session logging / persistence
- Mobile or web deployment
