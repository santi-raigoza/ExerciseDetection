# Handoff

**Last updated:** 2026-06-09

## Current Objective

Implement the 6-task exercise detection plan using subagent-driven development (TDD, one task at a time with spec + quality review after each).

## Work Completed This Session

| Task | Status | Commit |
|---|---|---|
| Task 1: Project scaffold | ✅ Complete | 87da256 |
| Task 2: landmarks.py | ✅ Complete | defbd38 |
| Task 3: rep_counter.py | 🔄 Next | — |
| Task 4: collect.py | ⏳ Pending | — |
| Task 5: train.py | ⏳ Pending | — |
| Task 6: app.py | ⏳ Pending | — |

## Files Created This Session

- `src/__init__.py`
- `src/utils/__init__.py`
- `src/utils/landmarks.py`
- `tests/__init__.py`
- `tests/utils/__init__.py`
- `tests/utils/test_landmarks.py`
- `conftest.py`
- `data/raw/.gitkeep`
- `models/.gitkeep`
- `docs/superpowers/specs/2026-06-09-exercise-detection-design.md`
- `docs/superpowers/plans/2026-06-09-exercise-detection.md`
- `PROJECT_STATUS.md`
- `HANDOFF.md`

## Files Modified This Session

- `.gitignore` — added patterns for `data/raw/*.csv`, `models/*.pkl`
- `requirements.txt` — added `pytest==8.4.0`

## Current Status

Tasks 1 and 2 complete and reviewed. `src/utils/landmarks.py` provides `normalize()` and `build_window()` — the feature extraction pipeline shared by collect.py, train.py, and app.py.

Awaiting user approval to begin Task 3.

## Outstanding Tasks

- Task 3: `src/utils/rep_counter.py` + `tests/utils/test_rep_counter.py`
- Task 4: `src/collect.py` + `tests/test_collect.py`
- Task 5: `src/train.py` + `tests/test_train.py`
- Task 6: `src/app.py` + full test suite verification
- Manual: collect training data (5 exercises × 300 windows each)
- Manual: run `python src/train.py` and verify accuracy
- Manual: run `python src/app.py` and verify live overlay

## Design Decisions

- **Sliding-window MLP over LSTM** — chosen because it captures temporal context (needed for juggling/ladder drills) without LSTM complexity; can be upgraded later at same interface boundary
- **`normalize()` uses hip midpoint + torso height** — makes features invariant to position in frame and distance from camera
- **`rest` class included** — prevents model from always predicting an exercise when none is being performed
- **Jumping jack uses wrist horizontal spread** (`|left_wrist_x - right_wrist_x|`) — more reliable in normalized [0,1] coords than wrist-to-hip distance
- **Jumping jack initial state = 'down'** — prevents spurious rep count when starting with arms at sides
- **No Co-Authored-By in commits** — user preference

## Blockers

None currently.

## Next Immediate Action

User must approve before starting Task 3: `rep_counter.py` — the per-exercise state machine that tracks joint angles and counts reps for pushup, squat, pullup, and jumping jack.
