"""
Quick visual test for landmarks.py — opens webcam, draws skeleton overlay,
and shows normalized hip/shoulder values to verify Task 2 is working.

Hip midpoint should stay near (0.0, 0.0) regardless of where you stand.
Shoulder midpoint y should stay near -1.0 regardless of distance from camera.

Run: .venv/bin/python test.py
Press Q to quit.
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

TASK_PATH = "pose_landmarker_lite.task"


def draw_skeleton(frame, landmarks):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    for a, b in POSE_CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)


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

    print("Webcam open. Press Q to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            draw_skeleton(frame, landmarks)

            normed = normalize(landmarks).reshape(33, 3)
            hip_mid = (normed[23] + normed[24]) / 2.0
            shoulder_mid = (normed[11] + normed[12]) / 2.0

            cv2.putText(frame, f"Hip mid (norm):      ({hip_mid[0]:.2f}, {hip_mid[1]:.2f})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Shoulder mid (norm): ({shoulder_mid[0]:.2f}, {shoulder_mid[1]:.2f})",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Hip mid should be ~(0.00, 0.00)", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.putText(frame, "Shoulder y should be ~-1.00", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        else:
            cv2.putText(frame, "No pose detected — step back so full body is visible",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("landmarks.py visual test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
