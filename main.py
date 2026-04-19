"""
main.py
-------
Entry point for the drowsiness detection system.
Run this file to start real-time detection from your webcam.

Usage:
    python main.py
    python main.py --camera 0       # specify camera index
    python main.py --no-alert       # disable audio alert
    python main.py --save-log       # save session CSV on exit
"""

import cv2
import time
import argparse
import sys
from detector import DrowsinessDetector

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARN] pygame not found. Audio alerts disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection")
    parser.add_argument("--camera",    type=int,  default=0,     help="Camera index (default: 0)")
    parser.add_argument("--no-alert",  action="store_true",       help="Disable audio alert")
    parser.add_argument("--save-log",  action="store_true",       help="Save session log to CSV on exit")
    parser.add_argument("--width",     type=int,  default=640,   help="Frame width")
    parser.add_argument("--height",    type=int,  default=480,   help="Frame height")
    return parser.parse_args()


def generate_alert_sound():
    """Generate a simple beep using pygame synthesizer."""
    if not PYGAME_AVAILABLE:
        return None
    import numpy as np
    sample_rate = 44100
    duration    = 0.4
    frequency   = 880
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    stereo = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(stereo)
    return sound


def main():
    args = parse_args()

    print("\n=== Drowsiness Detection System ===")
    print(f"Camera index : {args.camera}")
    print(f"Resolution   : {args.width}x{args.height}")
    print(f"Audio alerts : {'disabled' if args.no_alert else 'enabled'}")
    print("Press 'q' to quit | Press 's' to save snapshot")
    print("===================================\n")

    # Init webcam
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera}.")
        print("Try: python main.py --camera 1")
        sys.exit(1)

    # Init detector and alert sound
    detector = DrowsinessDetector()
    alert_sound = generate_alert_sound() if not args.no_alert else None

    # Timing
    start_time  = time.time()
    frame_idx   = 0
    alert_cooldown = 0   # prevent alert spam

    print("INFO: Starting detection loop. Initializing camera...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame. Check camera connection.")
            break

        frame_idx += 1
        timestamp = time.time() - start_time

        # Flip horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)

        # Run detection
        frame, status = detector.process_frame(frame, frame_idx, timestamp)

        # FPS counter
        fps = frame_idx / max(timestamp, 0.001)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Session timer
        elapsed = int(timestamp)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (frame.shape[1] - 75, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Audio alert (with cooldown so it doesn't loop constantly)
        any_alert = status["eye_alert"] or status["yawn_alert"] or status["pitch_alert"]
        if any_alert and alert_sound and alert_cooldown <= 0:
            alert_sound.play()
            alert_cooldown = 60  # ~2 seconds at 30fps before next beep
        if alert_cooldown > 0:
            alert_cooldown -= 1

        # Show frame
        cv2.imshow("Drowsiness Detection - Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_path = f"snapshot_{frame_idx}.png"
            cv2.imwrite(snapshot_path, frame)
            print(f"[SNAP] Saved {snapshot_path}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Session ended. {frame_idx} frames processed.")
    print(f"[INFO] Duration: {int(time.time() - start_time)}s")

    if args.save_log:
        detector.save_log("session_log.csv")
        print("[INFO] Run: python analyze.py  to generate plots")

    if PYGAME_AVAILABLE:
        pygame.quit()


if __name__ == "__main__":
    main()
