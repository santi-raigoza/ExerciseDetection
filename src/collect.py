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
    parser.add_argument('--video', default=None,
                        help='Path to a video file. Omit to use the webcam.')
    args = parser.parse_args()

    options = PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=TASK_PATH),
        running_mode=RunningMode.IMAGE,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open {'video file: ' + args.video if args.video else 'webcam'}.")
        return

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    windows = []
    # Video files collect automatically; webcam requires SPACE to start/stop.
    recording = bool(args.video)
    frame_count = 0

    print(f"Exercise: {args.exercise} | Target: {args.samples} windows")
    if args.video:
        print(f"Source: {args.video}  (auto-recording)")
    else:
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
                        print(f"Target reached: {len(windows)} windows collected.")
                        break

        status = 'RECORDING' if recording else 'READY'
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(frame, f'{status}  {len(windows)}/{args.samples}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, args.exercise, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not args.video:
            recording = not recording
            if recording:
                frame_buffer.clear()
                frame_count = 0

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
