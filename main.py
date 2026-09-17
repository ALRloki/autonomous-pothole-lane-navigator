"""
Main Real-time Execution Pipeline for Autonomous Navigator.
Processes live camera streams or pre-recorded road footage with
low-latency ADAS heads-up display rendering and telemetry logging.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

from navigator.detector import PotholeDetector
from navigator.lane_tracker import LanePrecisionTracker
from navigator.size_estimator import PotholeSizeEstimator
from navigator.fusion_engine import AutonomousFusionEngine
from navigator.telemetry_hud import TelemetryHUD


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Enhanced Autonomous Navigator for Real-Time Pothole Sensing and Lane Precision"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sample_data/sample_road.mp4",
        help="Path to video file or webcam device index ('0')"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom YOLOv11 weights (.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold for pothole detection (0.0 - 1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="annotated_output.mp4",
        help="Path to save annotated output video (or empty string to skip)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode without opening GUI window"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input source
    source_input = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source_input, str) and not os.path.exists(source_input):
        print(f"[-] Error: Source file not found: {source_input}")
        print("[!] Tip: Run 'python generate_sample_data.py' first to create test media.")
        sys.exit(1)

    print("==================================================================")
    print("  AI-Enhanced Autonomous Navigator for Road Monitoring & ADAS   ")
    print("==================================================================")
    print(f"[*] Input Source:      {args.source}")
    print(f"[*] Detection Model:   YOLOv11 ({args.weights or 'Standard/Adaptive CV Engine'})")
    print(f"[*] Confidence Thresh: {args.conf}")
    print("------------------------------------------------------------------")

    # Initialize Perception Modules
    detector = PotholeDetector(model_path=args.weights, confidence_threshold=args.conf)
    lane_tracker = LanePrecisionTracker()
    size_estimator = PotholeSizeEstimator()
    fusion_engine = AutonomousFusionEngine()
    hud = TelemetryHUD()

    cap = cv2.VideoCapture(source_input)
    if not cap.isOpened():
        print(f"[-] Failed to open video source: {source_input}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Video Writer setup if output requested
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, input_fps, (width, height))
        print(f"[*] Recording annotated output to: {args.output}")

    fps_history = []
    frame_idx = 0
    paused = False

    print("\n[+] Stream active. Controls:")
    print("    [Q] Quit | [Space] Pause/Resume | [S] Save Snapshot\n")

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[*] Reached end of video stream.")
                    break

                frame_idx += 1
                t_start = time.perf_counter()

                # 1. Detect Potholes (YOLOv11)
                potholes = detector.detect(frame)

                # 2. Track Lane Boundaries & Compute Precision (Canny + Hough)
                lane_info = lane_tracker.process_frame(frame)

                # 3. Estimate Physical Pothole Dimensions (Perspective scaling)
                enhanced_potholes = size_estimator.batch_estimate(
                    potholes, frame.shape, lane_width_px_bottom=width * 0.55
                )

                # 4. Synthesize Trajectory & Avoidance Directives
                directive = fusion_engine.compute_navigation_directive(
                    lane_info, enhanced_potholes, frame.shape
                )

                # Measure latency & calculate FPS
                latency_ms = (time.perf_counter() - t_start) * 1000.0
                curr_fps = 1000.0 / max(1.0, latency_ms)
                fps_history.append(curr_fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                smooth_fps = np.mean(fps_history)

                # 5. Render Automotive Heads-Up Display
                annotated = hud.render(
                    frame, lane_info, enhanced_potholes, directive,
                    fps=smooth_fps, latency_ms=latency_ms
                )

                if writer:
                    writer.write(annotated)

            if not args.no_display:
                cv2.imshow("AI Autonomous Navigator - ADAS Real-Time Feed", annotated)
                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key == ord('q'):
                    print("[*] User requested exit.")
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"[*] {'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    snap_name = f"snapshot_frame_{frame_idx:04d}.png"
                    cv2.imwrite(snap_name, annotated)
                    print(f"[+] Saved snapshot to: {snap_name}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        avg_overall_fps = np.mean(fps_history) if fps_history else 0.0
        print("\n==================================================================")
        print(f"[+] Processing Completed: {frame_idx} frames processed.")
        print(f"[+] Average Processing Speed: {avg_overall_fps:.1f} FPS")
        if args.output:
            print(f"[+] Annotated Video Saved: {args.output}")
        print("==================================================================")


if __name__ == "__main__":
    main()
