"""
calibrate.py
------------
Run this BEFORE your first real session to find YOUR personal
EAR baseline. Everyone's face geometry is slightly different.

Usage:
    python calibrate.py

Follow the on-screen prompts. Takes about 30 seconds.
Outputs recommended threshold values to copy into detector.py
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import time
from detector import eye_aspect_ratio, mouth_aspect_ratio, RIGHT_EYE, LEFT_EYE, MOUTH


def calibrate():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    phases = [
        {"label": "EYES OPEN - look at the camera naturally", "duration": 5, "key": "open"},
        {"label": "EYES CLOSED - hold them closed",           "duration": 3, "key": "closed"},
        {"label": "YAWNING - open your mouth wide",           "duration": 3, "key": "yawn"},
        {"label": "NORMAL - relax your face",                 "duration": 3, "key": "normal"},
    ]

    collected = {p["key"]: [] for p in phases}
    phase_idx = 0
    phase_start = None
    countdown_start = 3

    print("\n=== Calibration Mode ===")
    print("Follow the on-screen instructions.")
    print("Press SPACE to start each phase, Q to quit.\n")

    waiting_for_space = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        current_phase = phases[phase_idx] if phase_idx < len(phases) else None

        # Draw instruction
        if current_phase:
            label = current_phase["label"]
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        if waiting_for_space and current_phase:
            cv2.putText(frame, "Press SPACE to begin this phase",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        elif phase_start and current_phase:
            elapsed = time.time() - phase_start
            remaining = current_phase["duration"] - elapsed
            cv2.putText(frame, f"Recording... {remaining:.1f}s",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Collect metrics
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
                ear = (right_ear + left_ear) / 2.0
                mar = mouth_aspect_ratio(lm, MOUTH, w, h)
                collected[current_phase["key"]].append((ear, mar))

                # Live readout
                cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if remaining <= 0:
                phase_idx += 1
                phase_start = None
                waiting_for_space = True
                if phase_idx >= len(phases):
                    break

        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and waiting_for_space:
            phase_start = time.time()
            waiting_for_space = False

    cap.release()
    cv2.destroyAllWindows()

    # Compute recommendations
    print("\n========= CALIBRATION RESULTS =========")

    if collected["open"]:
        open_ears = [e for e, m in collected["open"]]
        mean_open = np.mean(open_ears)
        std_open  = np.std(open_ears)
        print(f"Eyes open  EAR: {mean_open:.3f} ± {std_open:.3f}")
    else:
        mean_open = 0.28

    if collected["closed"]:
        closed_ears = [e for e, m in collected["closed"]]
        mean_closed = np.mean(closed_ears)
        print(f"Eyes closed EAR: {mean_closed:.3f}")
    else:
        mean_closed = 0.18

    if collected["yawn"]:
        yawn_mars = [m for e, m in collected["yawn"]]
        mean_yawn = np.mean(yawn_mars)
        print(f"Yawn MAR:   {mean_yawn:.3f}")
    else:
        mean_yawn = 0.7

    if collected["normal"]:
        normal_mars = [m for e, m in collected["normal"]]
        mean_normal_mar = np.mean(normal_mars)
        print(f"Normal MAR: {mean_normal_mar:.3f}")
    else:
        mean_normal_mar = 0.35

    # Recommended thresholds
    recommended_ear = round((mean_open + mean_closed) / 2, 3)
    recommended_mar = round((mean_yawn + mean_normal_mar) / 2, 3)

    print(f"\n>>> Recommended EAR_THRESHOLD : {recommended_ear}")
    print(f">>> Recommended MAR_THRESHOLD : {recommended_mar}")
    print("\nUpdate these values in detector.py if they differ from defaults.")
    print("=======================================\n")


if __name__ == "__main__":
    calibrate()
