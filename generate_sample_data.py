"""
Synthetic Road Video and Dataset Generator for Autonomous Navigator Testing.
Generates realistic perspective road video and annotated image frames
containing asphalt textures, lane markings, and varying pothole hazards.
"""

import os
import cv2
import json
import numpy as np


def create_realistic_road_frame(
    frame_idx: int,
    total_frames: int,
    width: int = 1280,
    height: int = 720,
    pothole_specs: list = None
) -> tuple:
    """
    Synthesizes a single high-definition road frame with perspective lane markings
    and realistic asphalt depressions.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky & Horizon Gradient
    horizon_y = int(height * 0.44)
    for y in range(horizon_y):
        ratio = y / float(horizon_y)
        # Blue to light haze gradient
        r = int(135 + (200 - 135) * ratio)
        g = int(180 + (220 - 180) * ratio)
        b = int(230 + (245 - 230) * ratio)
        img[y, :] = (b, g, r)

    # 2. Road Asphalt Texture
    road_h = height - horizon_y
    for y in range(horizon_y, height):
        ratio = (y - horizon_y) / float(road_h)
        base_gray = int(55 + 25 * ratio)
        noise = np.random.randint(-6, 7, size=(width,), dtype=np.int16)
        row_vals = np.clip(base_gray + noise, 35, 95).astype(np.uint8)
        img[y, :, 0] = row_vals
        img[y, :, 1] = row_vals
        img[y, :, 2] = row_vals

    # Subtle lateral road weave simulation
    curve_offset = int(np.sin(frame_idx * 0.03) * 20)

    # Road Vanishing Point
    vp_x = int(width * 0.50) + curve_offset
    vp_y = horizon_y

    # Road Shoulders / Curbs
    left_bottom_x = int(width * 0.12)
    right_bottom_x = int(width * 0.88)

    # Left Solid White Lane
    left_lane_bot = int(width * 0.22)
    left_lane_top = vp_x - 18

    # Right Solid/Dashed White Lane
    right_lane_bot = int(width * 0.78)
    right_lane_top = vp_x + 18

    # Center Dashed Yellow Divider (for realistic highway context)
    center_lane_bot = int(width * 0.50)
    center_lane_top = vp_x

    # Draw Road Boundary Lines
    cv2.line(img, (left_bottom_x, height), (vp_x - 45, vp_y), (40, 45, 50), 3)
    cv2.line(img, (right_bottom_x, height), (vp_x + 45, vp_y), (40, 45, 50), 3)

    # Draw Left Lane Line (Solid White)
    cv2.line(img, (left_lane_bot, height), (left_lane_top, vp_y), (235, 240, 245), 9, cv2.LINE_AA)

    # Draw Right Lane Line (Solid White)
    cv2.line(img, (right_lane_bot, height), (right_lane_top, vp_y), (235, 240, 245), 9, cv2.LINE_AA)

    # Draw Center Dashed Yellow Line
    dash_speed = int((frame_idx * 16) % 120)
    for d in range(0, road_h + 120, 90):
        y1 = height - d + dash_speed
        y2 = y1 - 45
        if y1 > horizon_y:
            y1_c = min(height, y1)
            y2_c = max(horizon_y + 1, y2)
            if y1_c > y2_c:
                t1 = (y1_c - vp_y) / float(height - vp_y)
                t2 = (y2_c - vp_y) / float(height - vp_y)
                x1 = int(vp_x + (center_lane_bot - vp_x) * t1)
                x2 = int(vp_x + (center_lane_bot - vp_x) * t2)
                cv2.line(img, (x1, y1_c), (x2, y2_c), (30, 210, 250), 5, cv2.LINE_AA)

    # 3. Add Dynamic Potholes with perspective progression
    active_boxes = []
    if pothole_specs:
        for spec in pothole_specs:
            start_f = spec["start_frame"]
            end_f = spec["end_frame"]
            if start_f <= frame_idx <= end_f:
                progress = (frame_idx - start_f) / float(end_f - start_f)
                # Pothole moves from horizon towards bumper as car drives forward
                py = int(horizon_y + 40 + progress * (height - horizon_y - 60))
                
                # Lateral lane position (relative to left/right lane)
                lane_pct = spec["lane_fraction"]  # 0.25 (left wheel), 0.75 (right wheel)
                t_depth = (py - vp_y) / float(height - vp_y)
                l_x = int(vp_x + (left_lane_bot - vp_x) * t_depth)
                r_x = int(vp_x + (right_lane_bot - vp_x) * t_depth)
                px = int(l_x + (r_x - l_x) * lane_pct)

                # Scaling based on depth
                base_w = spec["base_width"]
                base_h = spec["base_height"]
                scale = 0.25 + 1.25 * (t_depth ** 1.4)
                w_curr = int(base_w * scale)
                h_curr = int(base_h * scale)

                # Draw realistic dark asphalt depression
                draw_pothole_texture(img, px, py, w_curr, h_curr, spec.get("depth_factor", 1.0))

                x1 = max(0, px - w_curr // 2)
                y1 = max(0, py - h_curr // 2)
                x2 = min(width, px + w_curr // 2)
                y2 = min(height, py + h_curr // 2)
                active_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": "pothole",
                    "severity": spec.get("severity", "MODERATE")
                })

    return img, active_boxes


def draw_pothole_texture(img: np.ndarray, cx: int, cy: int, w: int, h: int, depth_factor: float = 1.0):
    """Renders a naturalistic irregular pothole cavity on asphalt."""
    ih, iw = img.shape[:2]
    # Irregular polygon for natural jagged cavity edge
    num_pts = 16
    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    rng = np.random.RandomState(cx + cy)  # Deterministic jitter for this pothole instance
    radii_x = (w / 2.0) * (1.0 + 0.22 * rng.randn(num_pts))
    radii_y = (h / 2.0) * (1.0 + 0.22 * rng.randn(num_pts))

    pts = []
    for a, rx, ry in zip(angles, radii_x, radii_y):
        pt_x = int(np.clip(cx + rx * np.cos(a), 0, iw - 1))
        pt_y = int(np.clip(cy + ry * np.sin(a), 0, ih - 1))
        pts.append([pt_x, pt_y])
    pts = np.array(pts, dtype=np.int32)

    # 1. Dark interior crater
    dark_gray = max(12, int(28 / depth_factor))
    cv2.fillPoly(img, [pts], (dark_gray, dark_gray, dark_gray))

    # 2. Inner depth shadow gradient
    inner_pts = (pts * 0.65 + np.array([cx, cy]) * 0.35).astype(np.int32)
    inner_dark = max(8, int(18 / depth_factor))
    cv2.fillPoly(img, [inner_pts], (inner_dark, inner_dark, inner_dark))

    # 3. High-contrast weathered eroded rim
    cv2.polylines(img, [pts], True, (35, 38, 42), 2, cv2.LINE_AA)
    # Bright sun-glint edge on lower rim
    cv2.polylines(img, [pts[num_pts // 4 : 3 * num_pts // 4]], False, (95, 100, 105), 1, cv2.LINE_AA)


def generate_dataset(output_dir: str = "sample_data", video_duration_sec: int = 12, fps: int = 30):
    """Generates test video and annotated benchmark image set."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    video_path = os.path.join(output_dir, "sample_road.mp4")
    total_frames = video_duration_sec * fps
    width, height = 1280, 720

    # Define 3 sequential pothole encounters
    pothole_specs = [
        # Pothole 1: Right wheel path, frames 20-110, moderate
        {
            "start_frame": 25,
            "end_frame": 115,
            "lane_fraction": 0.72,
            "base_width": 75,
            "base_height": 45,
            "depth_factor": 1.2,
            "severity": "MODERATE"
        },
        # Pothole 2: Left wheel path, frames 130-220, critical
        {
            "start_frame": 135,
            "end_frame": 225,
            "lane_fraction": 0.28,
            "base_width": 115,
            "base_height": 65,
            "depth_factor": 1.7,
            "severity": "CRITICAL"
        },
        # Pothole 3: Center straddle, frames 245-335, minor
        {
            "start_frame": 250,
            "end_frame": 340,
            "lane_fraction": 0.50,
            "base_width": 55,
            "base_height": 35,
            "depth_factor": 1.0,
            "severity": "MINOR"
        }
    ]

    print(f"[+] Generating synthetic road test video: {video_path} ({total_frames} frames)...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    ground_truth_db = []

    for f in range(total_frames):
        frame, bboxes = create_realistic_road_frame(f, total_frames, width, height, pothole_specs)
        writer.write(frame)

        # Save sample benchmark frames every 15 frames
        if f % 15 == 0:
            img_filename = f"road_frame_{f:04d}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, frame)

            # Write YOLO format label (class x_center y_center width height normalized)
            label_filename = f"road_frame_{f:04d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, "w") as lf:
                for b in bboxes:
                    bx1, by1, bx2, by2 = b["bbox"]
                    bw = (bx2 - bx1) / float(width)
                    bh = (by2 - by1) / float(height)
                    bcx = (bx1 + bx2) / 2.0 / float(width)
                    bcy = (by1 + by2) / 2.0 / float(height)
                    lf.write(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}\n")

            ground_truth_db.append({
                "frame": f,
                "image_file": img_filename,
                "boxes": bboxes
            })

    writer.release()
    print(f"[+] Video successfully written to: {video_path}")

    # Write ground truth JSON index
    gt_json_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_json_path, "w") as jf:
        json.dump(ground_truth_db, jf, indent=2)
    print(f"[+] Ground truth benchmark dataset saved to: {gt_json_path}")


if __name__ == "__main__":
    generate_dataset()
