# Project Status

## Project Goal

Build a real-time exercise detection app that uses webcam pose data to classify exercises (pushup, pullup, squat, jumping jack) and count reps. Designed to extend to weightlifting, soccer juggling, and conditioning drills.

## Current Architecture

Three-phase ML pipeline:

```
Phase 1 — collect.py  : Webcam → MediaPipe → normalized landmarks → labeled CSV files
Phase 2 — train.py    : CSVs → sliding windows → MLP classifier → saved model (.pkl)
Phase 3 — app.py      : Webcam → MediaPipe → sliding window → model → exercise + rep count
```

**Classifier:** Sliding-window MLP (30 frames × 99 features = 2,970-feature input vector)
**Rep counting:** Per-exercise angle-based state machine with 5-frame smoothing

## Technology Stack

| Component | Library |
|---|---|
| Pose detection | MediaPipe Tasks API (PoseLandmarker, pose_landmarker_lite.task) |
| Computer vision | OpenCV |
| ML classifier | scikit-learn MLPClassifier |
| Feature scaling | scikit-learn StandardScaler |
| Numerical ops | NumPy |
| Data handling | pandas |
| Model persistence | joblib |
| Testing | pytest |
| Language | Python 3.13 |

## Repository Structure

```
ExerciseDetection/
├── data/
│   └── raw/                  # Labeled training CSVs (gitignored)
├── docs/
│   └── superpowers/
│       ├── specs/            # Design documents
│       └── plans/            # Implementation plans
├── models/                   # Trained .pkl files (gitignored)
├── src/
│   ├── utils/
│   │   ├── landmarks.py      # normalize(), build_window(), POSE_CONNECTIONS
│   │   └── rep_counter.py    # RepCounter state machine ✅
│   ├── collect.py            # Data collection script (pending)
│   ├── train.py              # Training pipeline (pending)
│   └── app.py                # Real-time inference (pending)
├── tests/
│   ├── utils/
│   │   ├── test_landmarks.py # 6 passing tests
│   │   └── test_rep_counter.py # 8 passing tests
│   ├── test_collect.py       # (pending)
│   └── test_train.py         # (pending)
├── conftest.py               # Adds src/ to sys.path for tests
├── test.py                   # Visual test for landmarks.py (webcam, skeleton overlay)
├── test_rep_counter_live.py  # Live rep counter test (webcam, no model needed)
├── pose_landmarker_lite.task # MediaPipe model file
├── requirements.txt
├── PROJECT_STATUS.md
└── HANDOFF.md
```

## Completed Tasks

| Task | Description | Commit |
|---|---|---|
| Initial setup | Project init, pinned dependencies | b4d5873 |
| Design spec | Architecture, data model, component design | 0253b33 |
| Implementation plan | 6-task TDD plan | 026e32d |
| Task 1: Scaffold | Directory structure, conftest.py, pytest | 87da256 |
| Task 2: landmarks.py | normalize(), build_window(), POSE_CONNECTIONS, 6 tests | defbd38 |
| Task 3: rep_counter.py | RepCounter state machine, 8 tests | c450c24 |
| Task 3 tuning | Visibility-weighted side selection, threshold tuning, jumping jack fix | 5ae8cdd |

## Current Work

**Task 4: collect.py** — webcam data collection script. Ready to begin — all Task 3 exercises verified.

## Rep Counter State Machine (src/utils/rep_counter.py)

| Exercise | Camera orientation | Metric | Down threshold | Up threshold |
|---|---|---|---|---|
| Pushup | Side profile | Elbow angle (more visible side) | < 110° | > 130° |
| Squat | Side profile | Knee angle (more visible side) | < 90° | > 160° |
| Pullup | Front-facing | shoulder_y − wrist_y | < 0.08 (hanging) | > 0.15 (pulled up) |
| Jumping jack | Facing forward | min(wrist spread, ankle spread) / shoulder width | < 0.7 | > 1.3 |

**Key design decisions:**
- Uses the more visible side's joint (MediaPipe visibility score) instead of averaging — critical for side-profile exercises where one side is partially occluded
- Pullup uses shoulder-rise (`avg_shoulder_y − avg_wrist_y`) instead of elbow angle — works front-facing, no clean side profile needed. Rise *increases* when pulled up (MediaPipe wrist landmark drops when elbows bend). Thresholds calibrated empirically.
- Jumping jack metric is normalized by shoulder width (scale-invariant regardless of camera distance)
- Jumping jack requires BOTH wrist AND ankle spread to open (min of both ratios) — prevents arm-only motion from counting

## Live Test Scripts

- **`test.py`** — opens webcam, draws skeleton overlay, shows normalized hip/shoulder values to verify landmarks.py
- **`test_rep_counter_live.py`** — opens webcam, press 1-4 to select exercise, shows live joint angles, visibility scores, state, and rep count. No trained model needed.

## Verified Behavior (live tested)

| Exercise | Status |
|---|---|
| Squat | ✅ Working — side profile, tracks correctly |
| Pushup | ✅ Working — side profile, natural range of motion |
| Jumping jack | ✅ Working — facing forward, requires both arms and legs |
| Pullup | ✅ Verified — front-facing camera, shoulder-rise metric, 8 reps confirmed |

## Dataset Status

No training data collected yet. Data collection requires:
1. Complete `collect.py` (Task 4)
2. Record ~300 windows per exercise (5 classes × 300 = 1,500 total)
3. Stand 6–8 feet from webcam, full body visible

## Model Status

No model trained yet. Requires dataset first.

## Known Issues

- cSpell IDE warnings on package names (numpy, linalg, pullup, etc.) — cosmetic only, not real errors
- basedpyright warns "utils.landmarks could not be resolved" in test files — resolved at runtime via conftest.py sys.path injection; not a real error

## Next Steps

1. **Task 4: `collect.py`** — data collection script (ready to begin)
2. Task 5: `train.py` — training pipeline
3. Task 6: `app.py` — real-time inference
4. Collect training data (manual, ~15–20 min per exercise class)
5. Train model and evaluate accuracy
6. Run live app and verify rep counting

## Future Enhancements

- Weightlifting exercises (bench press, deadlift, bicep curl, overhead press)
- Soccer juggling detection
- Conditioning drills (ladder drills, agility exercises)
- Multi-person detection
- Form feedback / injury risk detection
- Audio rep cues
- Workout session logging
- LSTM upgrade if MLP accuracy plateaus on complex exercises

## Performance Metrics

Not yet available. Target: >85% classification accuracy on held-out test set.
