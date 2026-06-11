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
│   └── report.txt            # Latest training classification report ✅
├── src/
│   ├── utils/
│   │   ├── landmarks.py      # normalize(), build_window(), POSE_CONNECTIONS ✅
│   │   └── rep_counter.py    # RepCounter state machine ✅
│   ├── collect.py            # Data collection script ✅
│   ├── train.py              # Training pipeline ✅
│   └── app.py                # Real-time inference app ✅
├── tests/
│   ├── utils/
│   │   ├── test_landmarks.py   # 6 passing tests
│   │   └── test_rep_counter.py # 8 passing tests
│   ├── test_collect.py         # 3 passing tests
│   └── test_train.py           # 5 passing tests
├── conftest.py               # Adds src/ to sys.path for tests
├── README.md                 # Setup, usage, architecture overview
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
| Task 3 tuning | Visibility-weighted side selection, threshold tuning | 5ae8cdd |
| Task 3 pullup rewrite | Shoulder-rise metric, live test scripts | 6b27a08 |
| Task 4: collect.py | Data collection script, 3 tests, --video/--start/--end flags | 1aec2fd |
| Data collection | All 5 exercise classes collected (~300 windows each) | — |
| Task 5: train.py | Training pipeline, 5 tests, report.txt output | b771061 |
| Task 6: app.py | Real-time inference app, live tested | 2a3f711 |
| Squat threshold tuning | Loosened down/up thresholds for shallower squats | 03b2b75 |
| README | Full project documentation | a3cb9af |

## Current Work

**Hyperparameter tuning** — the only remaining gap for the course assignment.

The model was trained once with defaults (`hidden_layer_sizes=(256, 128)`, `max_iter=500`) and achieved 100% test accuracy. The course requires systematic tuning across multiple configurations with documented results.

### What needs to be built

A `src/tune.py` script that trains across a grid of configurations and outputs a comparison table. Suggested parameters to sweep:

| Hyperparameter | Values to try |
|---|---|
| `hidden_layer_sizes` | `(64,)`, `(128,)`, `(256, 128)`, `(512, 256, 128)` |
| `activation` | `relu`, `tanh`, `logistic` |
| `alpha` | `0.0001`, `0.001`, `0.01` |
| `learning_rate_init` | `0.001`, `0.01` |
| `solver` | `adam`, `sgd` |

Results go in the written report (due June 17).

## Model Performance

| Metric | Value |
|---|---|
| Test accuracy | 100% |
| Test set size | 300 windows (20% holdout) |
| Training data | ~300 windows per class × 5 classes |
| Architecture | MLP (256, 128), relu, adam, max_iter=500 |

Note: 100% accuracy reflects controlled collection conditions (one person, consistent angles). Real-world accuracy may vary. The interesting tuning question is whether a much smaller network achieves the same result.

## Rep Counter Thresholds

| Exercise | Metric | Down threshold | Up threshold |
|---|---|---|---|
| Pushup | Elbow angle (more visible side) | < 110° | > 130° |
| Squat | Knee angle (more visible side) | < 110° | > 140° |
| Pullup | shoulder_y − wrist_y | < 0.08 | > 0.15 |
| Jumping jack | min(wrist spread, ankle spread) / shoulder width | < 0.7 | > 1.3 |

## Dataset Status

All 5 classes collected. Stored in `data/raw/` as timestamped CSVs (gitignored).

| Exercise | Windows | Sources |
|---|---|---|
| Pushup | ~299 | Webcam (both sides) + 3 YouTube videos |
| Squat | ~300 | Webcam + 4 YouTube videos |
| Jumping Jack | ~300 | Webcam + 3 YouTube videos |
| Pullup | ~300 | 2 YouTube videos (front/back facing) |
| Rest | ~300 | Webcam (multiple sessions) + 2 YouTube videos |

## Known Issues

- cSpell IDE warnings on MediaPipe/OpenCV identifiers — cosmetic only
- 100% test accuracy likely reflects low data diversity; live performance is the real benchmark
- Train/test split is row-based, not clip-based — overlapping windows may inflate accuracy

## Future Enhancements

- Weightlifting exercises (bench press, deadlift, bicep curl, overhead press)
- Soccer juggling detection
- Conditioning drills
- Multi-person detection
- Form feedback / injury risk detection
- Audio rep cues
- Workout session logging
- LSTM upgrade if MLP accuracy plateaus
- Clip-based train/test split for more honest evaluation
