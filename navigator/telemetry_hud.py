"""
Real-time Heads-Up Display (HUD) Rendering Module using OpenCV.
Renders driving corridor overlay, pothole bounding boxes, dimensions,
telemetry stats, steering horizon, and driver-assistance warnings.
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional


class TelemetryHUD:
    """
    Renders high-definition automotive ADAS telemetry overlays on video frames.
    """

    def __init__(self):
        # Color palette (BGR format)
        self.COLOR_GREEN = (46, 204, 113)
        self.COLOR_AMBER = (0, 165, 255)
        self.COLOR_RED = (60, 60, 235)
        self.COLOR_CYAN = (240, 200, 0)
        self.COLOR_DARK_BG = (20, 20, 25)
        self.COLOR_WHITE = (255, 255, 255)

    def render(
        self,
        frame: np.ndarray,
        lane_info: Dict[str, Any],
        potholes: List[Dict[str, Any]],
        directive: Dict[str, Any],
        fps: float = 30.0,
        latency_ms: float = 33.0
    ) -> np.ndarray:
        """
        Overlay complete ADAS HUD telemetry onto the input frame.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Render Lane Corridor and Boundaries
        canvas = self._render_lane_corridor(canvas, lane_info, directive)

        # 2. Render Pothole Detections and Sizing Badges
        canvas = self._render_potholes(canvas, potholes)

        # 3. Render Top Telemetry Bar
        canvas = self._render_top_banner(canvas, directive, fps, latency_ms, len(potholes))

        # 4. Render Bottom Steering & Deviation Instrument
        canvas = self._render_steering_dial(canvas, lane_info, directive)

        return canvas

    def _render_lane_corridor(
        self,
        frame: np.ndarray,
        lane_info: Dict[str, Any],
        directive: Dict[str, Any]
    ) -> np.ndarray:
        """Render translucent lane polygon and lane lines."""
        polygon = lane_info.get("lane_polygon")
        left_line = lane_info.get("left_line")
        right_line = lane_info.get("right_line")

        if polygon is not None:
            alert = directive.get("alert_level", "SAFE")
            if alert == "SAFE":
                corridor_color = self.COLOR_GREEN
            elif alert in ("CAUTION", "WARNING"):
                corridor_color = self.COLOR_AMBER
            else:
                corridor_color = self.COLOR_RED

            # Translucent filled polygon
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], corridor_color)
            cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

        # Draw crisp lane boundary lines
        if left_line is not None:
            cv2.line(frame, left_line[0], left_line[1], self.COLOR_CYAN, 3, cv2.LINE_AA)
        if right_line is not None:
            cv2.line(frame, right_line[0], right_line[1], self.COLOR_CYAN, 3, cv2.LINE_AA)

        return frame

    def _render_potholes(
        self,
        frame: np.ndarray,
        potholes: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Render bounding boxes, dimension labels, and severity tags."""
        for p in potholes:
            x1, y1, x2, y2 = p["bbox"]
            conf = p.get("confidence", 0.0)
            w_cm = p.get("width_cm", 0.0)
            l_cm = p.get("length_cm", 0.0)
            area = p.get("area_cm2", 0.0)
            sev = p.get("severity", "MINOR")
            dist = p.get("distance_m", 10.0)
            color = p.get("hazard_color_bgr", (0, 200, 255))

            # Bounding box with corner accents
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            self._draw_corner_brackets(frame, x1, y1, x2, y2, color, length=12)

            # Center target reticle
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)

            # Floating label badge
            label_text = f"POTHOLE {int(conf * 100)}% | {w_cm:.0f}x{l_cm:.0f}cm | {area:.0f}cm2 | {dist:.1f}m [{sev}]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.42
            thickness = 1
            (lw, lh), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            badge_y1 = max(10, y1 - lh - 12)
            badge_y2 = y1 - 2
            badge_x1 = max(5, x1)
            badge_x2 = min(frame.shape[1] - 5, badge_x1 + lw + 12)

            # Translucent background for text badge
            sub_roi = frame[badge_y1:badge_y2, badge_x1:badge_x2]
            if sub_roi.shape[0] > 0 and sub_roi.shape[1] > 0:
                dark_rect = np.full_like(sub_roi, 15)
                cv2.addWeighted(dark_rect, 0.75, sub_roi, 0.25, 0, sub_roi)
                frame[badge_y1:badge_y2, badge_x1:badge_x2] = sub_roi

            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), color, 1)
            cv2.putText(
                frame,
                label_text,
                (badge_x1 + 6, badge_y2 - 5),
                font,
                font_scale,
                self.COLOR_WHITE,
                thickness,
                cv2.LINE_AA
            )

        return frame

    def _render_top_banner(
        self,
        frame: np.ndarray,
        directive: Dict[str, Any],
        fps: float,
        latency_ms: float,
        pothole_count: int
    ) -> np.ndarray:
        """Render futuristic top ADAS telemetry bar."""
        w = frame.shape[1]
        banner_h = 55

        # Background banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, banner_h), (w, banner_h), (60, 65, 80), 1)

        # Title & Subtitle
        cv2.putText(frame, "AUTONOMOUS NAVIGATOR ADAS", (18, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "YOLOv11 Deep Learning + Canny/Hough Fusion", (18, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 190), 1, cv2.LINE_AA)

        # Telemetry stats (FPS, Latency, Hazards)
        stats_x = int(w * 0.45)
        stat_text = f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Potholes: {pothole_count}"
        cv2.putText(frame, stat_text, (stats_x, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)

        # Active Maneuver Badge
        maneuver = directive.get("maneuver", "MAINTAIN_CENTER")
        alert = directive.get("alert_level", "SAFE")
        if alert == "SAFE":
            badge_color = self.COLOR_GREEN
        elif alert in ("CAUTION", "WARNING"):
            badge_color = self.COLOR_AMBER
        else:
            badge_color = self.COLOR_RED

        badge_str = f"[{maneuver.replace('_', ' ')}]"
        (bw, bh), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        bx = w - bw - 25
        cv2.rectangle(frame, (bx - 8, 14), (w - 15, 44), badge_color, 2)
        cv2.putText(frame, badge_str, (bx, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, badge_color, 1, cv2.LINE_AA)

        return frame

    def _render_steering_dial(
        self,
        frame: np.ndarray,
        lane_info: Dict[str, Any],
        directive: Dict[str, Any]
    ) -> np.ndarray:
        """Render bottom steering horizon and lane precision telemetry."""
        h, w = frame.shape[:2]
        panel_h = 60
        y_start = h - panel_h

        # Semi-transparent bottom panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), (12, 14, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, y_start), (w, y_start), (60, 65, 80), 1)

        # Lane departure status & Lateral offset
        offset_cm = lane_info.get("lateral_offset_cm", 0.0)
        status = lane_info.get("offset_status", "CENTERED")
        steer_deg = directive.get("recommended_steer_deg", 0.0)
        hazard_summary = directive.get("hazard_summary", "")

        # Offset & Steer angle text
        offset_sign = "+" if offset_cm > 0 else ""
        telemetry_txt = f"Lateral Offset: {offset_sign}{offset_cm:.1f} cm | Rec Steer: {steer_deg:+.1f} deg | Status: {status}"
        cv2.putText(frame, telemetry_txt, (20, y_start + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 230, 245), 1, cv2.LINE_AA)

        # Hazard summary message
        cv2.putText(frame, f"Guidance: {hazard_summary}", (20, y_start + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 190, 210), 1, cv2.LINE_AA)

        # Steering Horizon Dial in Center
        cx = int(w * 0.82)
        cy = y_start + 30
        dial_w = 90
        cv2.line(frame, (cx - dial_w, cy), (cx + dial_w, cy), (100, 110, 125), 1)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (200, 200, 200), 2)

        # Indicator needle based on offset / steering
        indicator_shift = int(np.clip(-offset_cm * 1.5, -dial_w, dial_w))
        ix = cx + indicator_shift
        cv2.circle(frame, (ix, cy), 6, self.COLOR_CYAN, -1)
        cv2.drawMarker(frame, (ix, cy), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 6, 1)

        return frame

    def _draw_corner_brackets(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: tuple,
        length: int = 10
    ) -> None:
        """Draw aesthetic corner brackets around bounding box."""
        # Top-left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, 3)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, 3)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, 3)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, 3)
