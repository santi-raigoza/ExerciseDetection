"""
Live test for rep_counter.py — uses real webcam landmarks with a hardcoded exercise label.
No trained model needed.

Controls:
  1 = pushup
  2 = squat
  3 = pullup
  4 = jumping jack
  R = reset rep count
  Q = quit
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from utils.landmarks import normalize, POSE_CONNECTIONS
from utils.rep_counter import RepCounter

TASK_PATH = "pose_landmarker_lite.task"

EXERCISES = {
    ord('1'): 'pushup',
    ord('2'): 'squat',
    ord('3'): 'pullup',
    ord('4'): 'jumping_jack',
}

THRESHOLDS = {
    'pushup':       'angle < 110 = down  |  angle > 130 = up (count)',
    'squat':        'angle < 90 = down  |  angle > 160 = up (count)',
    'pullup':       'rise > 0.15 = up (pulled)  |  rise < 0.08 = hanging (count)',
    'jumping_jack': 'spread > 1.3 = open  |  spread < 0.7 = closed (count)',
}

METRIC_LABELS = {
    'pushup':       'Elbow angle (best side)',
    'squat':        'Knee angle (best side)',
    'pullup':       'Shoulder rise (shoulder_y - wrist_y)',
    'jumping_jack': 'Combined spread (min wrist+ankle / shoulder)',
}


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)


def draw_hud(frame, exercise, metric_val, smoothed, state, count, left_angle=0.0, right_angle=0.0, left_vis=0.0, right_vis=0.0):
    h, w, _ = frame.shape

    cv2.rectangle(frame, (0, 0), (w, 135), (0, 0, 0), -1)

    cv2.putText(frame, f"Exercise: {exercise}  (1=pushup 2=squat 3=pullup 4=jumping_jack  R=reset)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    metric_label = METRIC_LABELS.get(exercise, 'metric')
    cv2.putText(frame, f"{metric_label}:  raw={metric_val:.1f}   smoothed={smoothed:.1f}",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f"Thresholds:  {THRESHOLDS.get(exercise, '')}",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if exercise in ('pushup', 'squat'):
        used_l = left_vis >= right_vis
        lc = (0, 255, 0) if used_l else (120, 120, 120)
        rc_color = (120, 120, 120) if used_l else (0, 255, 0)
        cv2.putText(frame, f"L={left_angle:.0f}° (vis={left_vis:.2f})",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2)
        cv2.putText(frame, f"R={right_angle:.0f}° (vis={right_vis:.2f})",
                    (230, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, rc_color, 2)
        cv2.putText(frame, "(bright = side being used)",
                    (450, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    elif exercise == 'pullup':
        shoulder_y, wrist_y = left_angle, right_angle
        cv2.putText(frame, f"shoulder_y={shoulder_y:.3f}  wrist_y={wrist_y:.3f}  rise={shoulder_y - wrist_y:.3f}",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, "up when rise > 0.15  |  hanging when rise < 0.08",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    elif exercise == 'jumping_jack':
        cv2.putText(frame, f"Wrist ratio: {left_angle:.2f}  Ankle ratio: {right_angle:.2f}  min={min(left_angle, right_angle):.2f}",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, "open when min > 1.3  |  closed when min < 0.7",
                    (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    state_color = (0, 200, 255) if state == 'down' else (0, 255, 100)
    cv2.putText(frame, f"State: {state or '—'}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    label = f"Reps: {count}"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.putText(frame, label, (w - tw - 16, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)


def live_metric(rc, exercise, landmarks):
    return rc._metric(exercise, landmarks)


def live_side_angles(exercise, landmarks):
    """Returns (left_angle, right_angle, left_vis, right_vis) for debug display."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    vis = np.array([getattr(lm, 'visibility', 0.5) for lm in landmarks], dtype=np.float32)

    from utils.rep_counter import (
        LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
        LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, _angle,
    )
    if exercise == 'pushup':
        left  = _angle(coords[LEFT_SHOULDER],  coords[LEFT_ELBOW],  coords[LEFT_WRIST])
        right = _angle(coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW], coords[RIGHT_WRIST])
        lv = (vis[LEFT_SHOULDER]  + vis[LEFT_ELBOW]  + vis[LEFT_WRIST])  / 3
        rv = (vis[RIGHT_SHOULDER] + vis[RIGHT_ELBOW] + vis[RIGHT_WRIST]) / 3
        return left, right, lv, rv
    if exercise == 'pullup':
        avg_shoulder_y = (coords[LEFT_SHOULDER][1] + coords[RIGHT_SHOULDER][1]) / 2
        avg_wrist_y = (coords[LEFT_WRIST][1] + coords[RIGHT_WRIST][1]) / 2
        return avg_shoulder_y, avg_wrist_y, 0.0, 0.0
    if exercise == 'squat':
        left  = _angle(coords[LEFT_HIP],  coords[LEFT_KNEE],  coords[LEFT_ANKLE])
        right = _angle(coords[RIGHT_HIP], coords[RIGHT_KNEE], coords[RIGHT_ANKLE])
        lv = (vis[LEFT_HIP]  + vis[LEFT_KNEE]  + vis[LEFT_ANKLE])  / 3
        rv = (vis[RIGHT_HIP] + vis[RIGHT_KNEE] + vis[RIGHT_ANKLE]) / 3
        return left, right, lv, rv
    if exercise == 'jumping_jack':
        shoulder_width = abs(coords[LEFT_SHOULDER][0] - coords[RIGHT_SHOULDER][0]) + 1e-6
        wrist = abs(coords[LEFT_WRIST][0] - coords[RIGHT_WRIST][0]) / shoulder_width
        ankle = abs(coords[LEFT_ANKLE][0] - coords[RIGHT_ANKLE][0]) / shoulder_width
        return wrist, ankle, 0.0, 0.0  # reuse left/right slots for wrist/ankle ratios
    return 0.0, 0.0, 0.0, 0.0


def main():
    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    rc = RepCounter()
    exercise = 'pushup'
    raw_metric = 0.0
    smoothed = 0.0

    print("Live rep counter test. Press 1-4 to switch exercise, R to reset, Q to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        left_angle = right_angle = left_vis = right_vis = 0.0

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame, landmarks)

            raw_metric = live_metric(rc, exercise, landmarks)
            rc.update(exercise, 0.9, landmarks)
            if rc._history:
                smoothed = float(np.mean(rc._history))
            left_angle, right_angle, left_vis, right_vis = live_side_angles(exercise, landmarks)
        else:
            cv2.putText(frame, "No pose — step back so full body is visible",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        draw_hud(frame, exercise, raw_metric, smoothed, rc._state, rc.count,
                 left_angle, right_angle, left_vis, right_vis)
        cv2.imshow("Rep Counter Live Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rc = RepCounter()
            raw_metric = 0.0
            smoothed = 0.0
        elif key in EXERCISES:
            exercise = EXERCISES[key]
            rc = RepCounter()
            raw_metric = 0.0
            smoothed = 0.0
            print(f"Switched to: {exercise}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
