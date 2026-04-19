"""
detector.py
-----------
Core drowsiness detection engine.
Handles facial landmark extraction, EAR/MAR computation,
head pose estimation, and temporal alert logic.
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from collections import deque


# ---------------------------------------------------------------------------
# MediaPipe landmark indices (478-point face mesh)

# Eye landmarks (MediaPipe specific indices)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # p1..p6 for EAR
LEFT_EYE  = [362, 385, 387, 263, 373, 380]  # p1..p6 for EAR

# Mouth landmarks
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]  # outer lip points for MAR

# Head pose reference points (3D model + 2D landmark indices)
HEAD_POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]  # nose, eyes, mouth corners, chin

# 3D model reference points (generic face model in mm)
MODEL_POINTS_3D = np.array([
    (0.0,    0.0,    0.0),    # nose tip (landmark 1)
    (-30.0, -125.0, -30.0),   # right eye corner (33)
    (30.0,  -125.0, -30.0),   # left eye corner (263)
    (-25.0,  170.0, -50.0),   # right mouth corner (61)
    (25.0,   170.0, -50.0),   # left mouth corner (291)
    (0.0,    250.0, -50.0),   # chin (199)
], dtype=np.float64)


# ---------------------------------------------------------------------------
# EAR and MAR helpers

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute Eye Aspect Ratio (EAR).
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    A lower EAR indicates a more closed eye.
    """
    pts = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices
    ], dtype=np.float64)

    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])

    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    """
    Compute Mouth Aspect Ratio (MAR).
    Higher MAR indicates an open mouth (yawning).
    """
    pts = np.array([
        (landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices
    ], dtype=np.float64)

    # Vertical distances
    A = dist.euclidean(pts[2], pts[6])
    B = dist.euclidean(pts[3], pts[7])
    # Horizontal distance
    C = dist.euclidean(pts[0], pts[1])

    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


# ---------------------------------------------------------------------------
# Head pose estimation

def get_head_pose(landmarks, w, h):
    """
    Estimate head pitch and yaw using solvePnP.
    Returns (pitch, yaw) in degrees. Negative pitch = head down.
    """
    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in HEAD_POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1        ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat([rotation_mat, np.zeros((3, 1))])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = float(euler_angles[0])
    yaw   = float(euler_angles[1])
    return pitch, yaw


# ---------------------------------------------------------------------------
# Drowsiness detector class

class DrowsinessDetector:
    """
    Full drowsiness detection pipeline.
    Tracks EAR, MAR, head pose across frames and fires alerts
    using temporal logic.
    """

    # Thresholds
    EAR_THRESHOLD       = 0.22   # below this = eye closing
    MAR_THRESHOLD       = 0.65   # above this = yawning
    PITCH_THRESHOLD     = 20.0   # degrees down = head nodding
    EAR_CONSEC_FRAMES   = 20     # ~0.67s at 30fps before eye alert
    MAR_CONSEC_FRAMES   = 15     # frames of yawning before alert
    PITCH_CONSEC_FRAMES = 25     # frames of head down before alert

    # Temporal smoothing window (mirrors DoG smoothing from Project 1)
    SMOOTH_WINDOW = 5

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Frame counters
        self.eye_counter   = 0
        self.yawn_counter  = 0
        self.pitch_counter = 0

        # Alert states
        self.eye_alert   = False
        self.yawn_alert  = False
        self.pitch_alert = False

        # Smoothing buffers (deque acts as sliding window)
        self.ear_buffer   = deque(maxlen=self.SMOOTH_WINDOW)
        self.mar_buffer   = deque(maxlen=self.SMOOTH_WINDOW)
        self.pitch_buffer = deque(maxlen=self.SMOOTH_WINDOW)

        # Data log for analysis
        self.log = []

    def process_frame(self, frame, frame_idx=0, timestamp=0.0):
        """
        Process a single frame. Returns annotated frame + status dict.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        status = {
            "frame": frame_idx,
            "timestamp": round(timestamp, 3),
            "ear": None,
            "mar": None,
            "pitch": None,
            "yaw": None,
            "eye_alert": False,
            "yawn_alert": False,
            "pitch_alert": False,
            "face_detected": False,
        }

        if not results.multi_face_landmarks:
            self._draw_no_face(frame)
            return frame, status

        lm = results.multi_face_landmarks[0].landmark
        status["face_detected"] = True

        # Compute raw metrics
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear       = (right_ear + left_ear) / 2.0
        mar       = mouth_aspect_ratio(lm, MOUTH, w, h)
        pitch, yaw = get_head_pose(lm, w, h)

        # Temporal smoothing (sliding window mean)
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        self.pitch_buffer.append(pitch)

        smooth_ear   = float(np.mean(self.ear_buffer))
        smooth_mar   = float(np.mean(self.mar_buffer))
        smooth_pitch = float(np.mean(self.pitch_buffer))

        # Temporal alert logic
        # EAR - eye closure
        if smooth_ear < self.EAR_THRESHOLD:
            self.eye_counter += 1
            if self.eye_counter >= self.EAR_CONSEC_FRAMES:
                self.eye_alert = True
        else:
            self.eye_counter = 0
            self.eye_alert   = False

        # MAR - yawning
        if smooth_mar > self.MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= self.MAR_CONSEC_FRAMES:
                self.yawn_alert = True
        else:
            self.yawn_counter = 0
            self.yawn_alert   = False

        # Pitch - head nodding down
        if smooth_pitch < -self.PITCH_THRESHOLD:
            self.pitch_counter += 1
            if self.pitch_counter >= self.PITCH_CONSEC_FRAMES:
                self.pitch_alert = True
        else:
            self.pitch_counter = 0
            self.pitch_alert   = False

        # Update status
        status.update({
            "ear":         round(smooth_ear,   3),
            "mar":         round(smooth_mar,   3),
            "pitch":       round(smooth_pitch, 2),
            "yaw":         round(yaw,          2),
            "eye_alert":   self.eye_alert,
            "yawn_alert":  self.yawn_alert,
            "pitch_alert": self.pitch_alert,
        })

        # Log for later analysis
        self.log.append(status.copy())

        # --- Draw overlays ---
        self._draw_landmarks(frame, lm, w, h)
        self._draw_hud(frame, status)

        return frame, status

    # -----------------------------------------------------------------------
    # Drawing helpers

    def _draw_landmarks(self, frame, lm, w, h):
        """Draw eye and mouth landmarks only (clean look)."""
        for idx in RIGHT_EYE:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)   # red = right eye

        for idx in LEFT_EYE:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1) # cyan = left eye

        for idx in MOUTH:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)   # green = mouth

    def _draw_hud(self, frame, status):
        """Draw HUD metrics and alert banners."""
        h, w = frame.shape[:2]

        # Alert border - red frame flash when drowsy
        any_alert = status["eye_alert"] or status["yawn_alert"] or status["pitch_alert"]
        if any_alert:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        # Metric panel background
        cv2.rectangle(frame, (0, 0), (260, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (260, 110), (50, 50, 50), 1)

        ear   = status["ear"]   if status["ear"]   is not None else 0.0
        mar   = status["mar"]   if status["mar"]   is not None else 0.0
        pitch = status["pitch"] if status["pitch"] is not None else 0.0

        ear_color   = (0, 0, 255) if status["eye_alert"]   else (0, 255, 0)
        mar_color   = (0, 0, 255) if status["yawn_alert"]  else (0, 255, 0)
        pitch_color = (0, 0, 255) if status["pitch_alert"] else (0, 255, 0)

        cv2.putText(frame, f"EAR:   {ear:.3f}",   (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_color,   2)
        cv2.putText(frame, f"MAR:   {mar:.3f}",   (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_color,   2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, 88),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, pitch_color, 2)

        # Alert banners
        y_offset = 140
        if status["eye_alert"]:
            cv2.putText(frame, "DROWSINESS ALERT: EYES CLOSING",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            y_offset += 35

        if status["yawn_alert"]:
            cv2.putText(frame, "YAWN DETECTED",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
            y_offset += 35

        if status["pitch_alert"]:
            cv2.putText(frame, "HEAD NODDING DETECTED",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 100, 255), 2)

    def _draw_no_face(self, frame):
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    def save_log(self, path="session_log.csv"):
        """Save session data to CSV for analysis."""
        import pandas as pd
        if self.log:
            pd.DataFrame(self.log).to_csv(path, index=False)
            print(f"[LOG] Session saved to {path}")
