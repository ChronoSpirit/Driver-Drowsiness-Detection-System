"""
analyze.py
----------
Post-session analysis. Reads session_log.csv and generates:
  - EAR / MAR timeline plot with alert markers
  - Head pitch over time
  - Alert event summary
  - Precision / recall estimation

Usage:
    python analyze.py
    python analyze.py --log my_session.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze drowsiness session log")
    parser.add_argument("--log", default="session_log.csv", help="Path to session CSV")
    parser.add_argument("--out", default="plots",           help="Output folder for plots")
    return parser.parse_args()


def load_log(path):
    if not os.path.exists(path):
        print(f"[ERROR] Log file not found: {path}")
        print("Run: python main.py --save-log  to generate a session log first.")
        exit(1)
    df = pd.read_csv(path)
    df = df[df["face_detected"] == True].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df


def plot_ear_timeline(df, out_dir):
    """EAR over time with alert markers - key figure for report."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df["timestamp"], df["ear"], color="#007BFF", linewidth=1.2, label="EAR (smoothed)")
    ax.axhline(y=0.22, color="red", linestyle="--", linewidth=1, label="EAR threshold (0.22)")

    # Shade alert regions
    alert_mask = df["eye_alert"].astype(bool)
    for _, group in df[alert_mask].groupby((~alert_mask).cumsum()):
        ax.axvspan(group["timestamp"].iloc[0], group["timestamp"].iloc[-1],
                   alpha=0.25, color="red", label="_nolegend_")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Eye Aspect Ratio", fontsize=11)
    ax.set_title("EAR Over Time with Drowsiness Alert Regions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.5)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "ear_timeline.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_mar_timeline(df, out_dir):
    """MAR over time with yawn markers."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df["timestamp"], df["mar"], color="#7B68EE", linewidth=1.2, label="MAR (smoothed)")
    ax.axhline(y=0.65, color="orange", linestyle="--", linewidth=1, label="MAR threshold (0.65)")

    yawn_mask = df["yawn_alert"].astype(bool)
    for _, group in df[yawn_mask].groupby((~yawn_mask).cumsum()):
        ax.axvspan(group["timestamp"].iloc[0], group["timestamp"].iloc[-1],
                   alpha=0.25, color="orange", label="_nolegend_")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Mouth Aspect Ratio", fontsize=11)
    ax.set_title("MAR Over Time with Yawn Alert Regions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "mar_timeline.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_head_pitch(df, out_dir):
    """Head pitch over time."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df["timestamp"], df["pitch"], color="#00FF6A", linewidth=1.2, label="Head pitch (degrees)")
    ax.axhline(y=-20, color="red", linestyle="--", linewidth=1, label="Nod threshold (-20°)")
    ax.axhline(y=0,   color="gray", linestyle=":",  linewidth=0.8)

    pitch_mask = df["pitch_alert"].astype(bool)
    for _, group in df[pitch_mask].groupby((~pitch_mask).cumsum()):
        ax.axvspan(group["timestamp"].iloc[0], group["timestamp"].iloc[-1],
                   alpha=0.2, color="red", label="_nolegend_")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Pitch angle (degrees)", fontsize=11)
    ax.set_title("Head Pitch Over Time (Negative = Head Down)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(out_dir, "head_pitch.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_combined_dashboard(df, out_dir):
    """3-panel dashboard - use this in your report."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Drowsiness Detection - Session Analysis Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)

    # EAR
    axes[0].plot(df["timestamp"], df["ear"], color="#007BFF", linewidth=1.2)
    axes[0].axhline(y=0.22, color="red", linestyle="--", linewidth=1, label="Threshold")
    axes[0].fill_between(df["timestamp"], df["ear"], 0.22,
                         where=df["ear"] < 0.22, alpha=0.2, color="red")
    axes[0].set_ylabel("EAR", fontsize=10)
    axes[0].set_ylim(0, 0.5)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Eye Aspect Ratio", fontsize=11)

    # MAR
    axes[1].plot(df["timestamp"], df["mar"], color="#664CFC", linewidth=1.2)
    axes[1].axhline(y=0.65, color="orange", linestyle="--", linewidth=1, label="Threshold")
    axes[1].fill_between(df["timestamp"], df["mar"], 0.65,
                         where=df["mar"] > 0.65, alpha=0.2, color="orange")
    axes[1].set_ylabel("MAR", fontsize=10)
    axes[1].set_ylim(0, 1.2)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("Mouth Aspect Ratio (Yawn Detection)", fontsize=11)

    # Pitch
    axes[2].plot(df["timestamp"], df["pitch"], color="#00FF6A", linewidth=1.2)
    axes[2].axhline(y=-20, color="red", linestyle="--", linewidth=1, label="Nod threshold")
    axes[2].axhline(y=0,   color="gray", linestyle=":", linewidth=0.8)
    axes[2].fill_between(df["timestamp"], df["pitch"], -20,
                         where=df["pitch"] < -20, alpha=0.2, color="red")
    axes[2].set_ylabel("Pitch (°)", fontsize=10)
    axes[2].set_xlabel("Time (seconds)", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Head Pitch (Nodding Detection)", fontsize=11)

    path = os.path.join(out_dir, "dashboard.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {path}")


def print_summary(df):
    """Print session summary statistics."""
    total_frames = len(df)
    duration     = df["timestamp"].max()

    eye_alert_frames   = df["eye_alert"].sum()
    yawn_alert_frames  = df["yawn_alert"].sum()
    pitch_alert_frames = df["pitch_alert"].sum()

    print("\n========== SESSION SUMMARY ==========")
    print(f"Duration           : {duration:.1f}s")
    print(f"Frames analyzed    : {total_frames}")
    print(f"Avg EAR            : {df['ear'].mean():.3f}")
    print(f"Avg MAR            : {df['mar'].mean():.3f}")
    print(f"Avg head pitch     : {df['pitch'].mean():.1f}°")
    print(f"Eye alert frames   : {eye_alert_frames} ({100*eye_alert_frames/total_frames:.1f}%)")
    print(f"Yawn alert frames  : {yawn_alert_frames} ({100*yawn_alert_frames/total_frames:.1f}%)")
    print(f"Pitch alert frames : {pitch_alert_frames} ({100*pitch_alert_frames/total_frames:.1f}%)")
    print("=====================================\n")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[INFO] Loading log: {args.log}")
    df = load_log(args.log)

    print_summary(df)
    plot_ear_timeline(df, args.out)
    plot_mar_timeline(df, args.out)
    plot_head_pitch(df, args.out)
    plot_combined_dashboard(df, args.out)

    print(f"\n[INFO] All plots saved to ./{args.out}/")
    print("[INFO] Use dashboard.png in your project report.")


if __name__ == "__main__":
    main()
