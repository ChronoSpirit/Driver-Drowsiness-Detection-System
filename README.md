# Real-time Driver Drowsiness Detection

A multi-signal computer vision pipeline that detects drivers drowsiness in real time using facial landmarks, temporal modeling, and head pose estimation.

### Author: Liam Campbell - EECE-5639 - Computer Vision - Northeastern University

## Overview

Driver drowsiness is a leading cause of road accidents. This system monitors three physiological signals simultaneously through a standard webcam:

- **Eye Aspect Ratio (EAR)** — detects prolonged eye closure
- **Mouth Aspect Ratio (MAR)** — detects yawning
- **Head pitch estimation** — detects head nodding via `cv2.solvePnP`

Alerts fire only after a signal persists across multiple consecutive frames — a temporal logic approach that mirrors derivative-of-Gaussian smoothing in classical motion detection, dramatically reducing false positives.

---

## System Architecture

```
Webcam feed
    │
    ▼
MediaPipe Face Mesh (468 landmarks)
    │
    ├── Eye landmarks → EAR computation
    ├── Mouth landmarks → MAR computation
    └── 6-point landmarks → solvePnP → pitch/yaw angles
            │
            ▼
    Temporal smoothing (sliding window)
            │
            ▼
    Frame counter logic (N consecutive frames)
            │
            ▼
    Alert system (visual overlay + audio beep)
            │
            ▼
    CSV data logger → Post-session analysis plots
```

---

## Drowsiness Signals

| Signal | Threshold | Consecutive Frames | Interpretation |
|--------|-----------|-------------------|----------------|
| EAR | < 0.22 | 20 frames (~0.67s) | Eyes closing |
| MAR | > 0.65 | 15 frames (~0.5s) | Yawning |
| Head pitch | < -20° | 25 frames (~0.83s) | Head nodding down |

Thresholds were selected empirically and can be adjusted in `detector.py`.

---

## Temporal Logic Design

A key design decision is that a single low EAR frame does not trigger an alert — the signal must persist. This is intentional:

- **False negative cost** (missing real drowsiness) >> **False positive cost** (unnecessary alert)
- Consecutive-frame counters act as a low-pass temporal filter, analogous to derivative-of-Gaussian smoothing in classical motion detection pipelines

---

### Eye Aspect Ratio (EAR)

EAR is computed using the formula from [Soukupová & Čech (2016)](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf):

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

Where `p1`–`p6` are the six eye landmark coordinates. EAR ≈ 0.28 for an open eye; drops toward 0 when closed.

### Temporal Logic Design

A single low-EAR frame does **not** trigger an alert — the signal must persist. This is intentional:

- A natural blink lasts 3–5 frames (~0.1–0.17s at 30fps) → no alert
- Drowsy eye closure lasts 20+ frames (~0.67s) → alert fires
- This mirrors the derivative-of-Gaussian temporal smoothing from classical motion detection — suppressing transient noise while preserving genuine events

---

### Head Pose Estimation

Six stable facial landmarks (nose tip, eye corners, mouth corners, chin) are matched against a generic 3D face model using `cv2.solvePnP`. The rotation vector is decomposed via `cv2.Rodrigues` into Euler angles, extracting **pitch** (forward head nod) and **yaw** (side turn).

---

## Project Structure

```
drowsiness_detection/
├── main.py          # Entry point — live webcam detection loop
├── detector.py      # Core engine (EAR, MAR, head pose, temporal logic, logging)
├── calibrate.py     # Personal threshold calibration (run before first session)
├── analyze.py       # Post-session analysis and plot generation
├── requirements.txt # Pinned dependencies
└── README.md
```

---


## Results

### Preliminary Observations

- **EAR baseline** for alert individuals: ~0.25–0.32
- **EAR at closure**: drops below 0.10 within 3–4 frames
- **MAR at yawn**: spikes to 0.7–0.9 (vs. resting ~0.35)
- **No spurious alerts** observed in initial testing with the 5-frame smoothing window
- **FPS on Surface Pro 8**: ~28–30fps sustained (MediaPipe CPU inference)

---

## How to use

```bash
# 1. Clone the repository
git clone https://github.com/ChronoSpirit/drowsiness_detection.git
cd drowsiness_detection

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
# Run live detection (default camera)
python main.py
```

Press `Q` to quit · Press `S` to save a snapshot of the current frame.

### Full Options

```bash
# Specify a different camera index (try if webcam doesn't appear)
python main.py --camera 1

# Save session data to CSV for analysis
python main.py --save-log

# Disable audio alert
python main.py --no-alert

# All options
python main.py --camera 0 --save-log --width 640 --height 480
```

### Recommended Workflow

```bash
# Step 1 — Calibrate to your face (run once, ~30 seconds)
python calibrate.py

# Step 2 — Run live detection with logging
python main.py --save-log

# Step 3 — Generate analysis plots from session
python analyze.py

# Plots saved to ./plots/
#   ear_timeline.png   — EAR over time with alert regions
#   mar_timeline.png   — MAR over time with yawn events
#   head_pitch.png     — Pitch trajectory with nod threshold
#   dashboard.png      — Combined 3-panel dashboard (use in reports)
```

---

## How It Works

### Pipeline Architecture

```
Webcam frame (640×480 @ 30fps)
        │
        ▼
MediaPipe Face Mesh — 478 facial landmarks
        │
        ├──► Eye landmarks (6 points) ──► EAR computation
        │
        ├──► Mouth landmarks (8 points) ──► MAR computation
        │
        └──► 6 anchor points ──► cv2.solvePnP ──► pitch / yaw angles
                │
                ▼
        Sliding window smoothing (n=5 frames)
                │
                ▼
        Consecutive-frame counter
        (EAR < 0.22 for 20+ frames → eye alert)
        (MAR > 0.65 for 15+ frames → yawn alert)
        (pitch < -20° for 25+ frames → nod alert)
                │
                ▼
        Visual overlay + audio beep (60-frame cooldown)
                │
                ▼
        CSV data logger → post-session analysis
```

---

## Future Work

* Implement full body detection
* Fix nodding
* Integrate driving gameplay to detect more variables

---

## References

1. Soukupová, T. & Čech, J. (2016). *Real-Time Eye Blink Detection using Facial Landmarks*. CVWW. [PDF](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
2. Kazemi, V. & Sullivan, J. (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees*. CVPR.
3. MediaPipe Face Mesh — https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html
4. OpenCV Documentation — https://docs.opencv.org/4.x/

---