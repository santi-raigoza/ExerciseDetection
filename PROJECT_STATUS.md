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
| Video downloading | yt-dlp |
| Testing | pytest |
| Language | Python 3.13 |

## Repository Structure

```
ExerciseDetection/
├── data/
│   ├── raw/                  # Labeled training CSVs (gitignored)
│   └── videos/               # Source videos used for data collection (gitignored)
├── docs/
│   └── superpowers/
│       ├── specs/            # Design documents
│       └── plans/            # Implementation plans
├── models/                   # Trained .pkl files (gitignored)
├── src/
│   ├── utils/
│   │   ├── landmarks.py      # normalize(), build_window(), POSE_CONNECTIONS ✅
│   │   └── rep_counter.py    # RepCounter state machine ✅
│   ├── collect.py            # Data collection script ✅
│   ├── train.py              # Training pipeline (pending)
│   └── app.py                # Real-time inference (pending)
├── tests/
│   ├── utils/
│   │   ├── test_landmarks.py # 6 passing tests
│   │   └── test_rep_counter.py # 8 passing tests
│   ├── test_collect.py       # 3 passing tests
│   └── test_train.py         # (pending)
├── conftest.py               # Adds src/ to sys.path for tests
├── test.py                   # Visual test for landmarks.py (webcam, skeleton overlay)
├── test_rep_counter_live.py  # Live rep counter test (webcam, no model needed)
├── pose_landmarker_lite.task # MediaPipe model file (gitignored)
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
| Task 3 pullup rewrite | Shoulder-rise metric, live test scripts committed | 6b27a08 |
| Task 4: collect.py | Data collection script, 3 tests, --video/--start/--end flags | 1aec2fd |
| Data collection | All 5 exercise classes collected (~300 windows each) | — |

## Current Work

**Task 5: train.py** — ready to begin. All training data collected.

## collect.py CLI Reference

```bash
# Webcam (spacebar to start/stop, Q to quit)
python src/collect.py --exercise pushup

# Video file (auto-records entire video)
python src/collect.py --exercise pushup --video data/videos/pushup.mp4

# Video with start/end timestamps (seconds)
python src/collect.py --exercise pullup --video data/videos/pullup.mp4 --start 60 --end 90

# Limit number of windows
python src/collect.py --exercise squat --samples 150
```

## Rep Counter State Machine (src/utils/rep_counter.py)

| Exercise | Camera orientation | Metric | Down threshold | Up threshold |
|---|---|---|---|---|
| Pushup | Side profile | Elbow angle (more visible side) | < 110° | > 130° |
| Squat | Side profile | Knee angle (more visible side) | < 90° | > 160° |
| Pullup | Front or back facing | shoulder_y − wrist_y | < 0.08 (hanging) | > 0.15 (pulled up) |
| Jumping jack | Facing forward | min(wrist spread, ankle spread) / shoulder width | < 0.7 | > 1.3 |

## Dataset Status

All 5 classes collected. Stored in `data/raw/` as timestamped CSVs.

| Exercise | Windows | Sources |
|---|---|---|
| Pushup | ~299 | Webcam (both sides) + 3 YouTube videos |
| Squat | ~300 | Webcam + 4 YouTube videos |
| Jumping Jack | ~300 | Webcam + 3 YouTube videos |
| Pullup | ~300 | 2 YouTube videos (front/back facing) |
| Rest | ~300 | Webcam (multiple sessions) + 2 YouTube videos |

## Model Status

Not yet trained. Ready to run Task 5.

## Known Issues

- cSpell IDE warnings on package names (numpy, linalg, pullup, etc.) — cosmetic only, not real errors
- basedpyright warns "utils.landmarks could not be resolved" in test files — resolved at runtime via conftest.py sys.path injection; not a real error
- yt-dlp warns about missing JavaScript runtime and ffmpeg — harmless, videos still download correctly

## Next Steps

1. **Task 5: `train.py`** — implement training pipeline + tests (ready to begin)
2. **Task 6: `app.py`** — real-time inference app
3. Run `python src/train.py` and verify >85% accuracy
4. Run `python src/app.py` and verify live overlay + rep counting

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
