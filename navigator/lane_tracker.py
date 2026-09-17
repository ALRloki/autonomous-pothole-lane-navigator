"""
Lane Precision Tracking Module utilizing Canny Edge Detection,
Dynamic Polygonal ROI Masking, and Probabilistic Hough Line Transform.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from collections import deque


class LanePrecisionTracker:
    """
    Real-time Lane Detection and Vehicle Deviation Precision System.
    Computes exact road boundaries, lane center, and vehicle lateral offset.
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        gaussian_kernel: int = 5,
        hough_threshold: int = 25,
        min_line_length: int = 35,
        max_line_gap: int = 40,
        lane_width_meters: float = 3.7,
        history_len: int = 8
    ):
        """
        Initialize the Lane Precision Tracker.

        Args:
            canny_low: Canny edge detector lower threshold.
            canny_high: Canny edge detector upper threshold.
            gaussian_kernel: Kernel size for noise smoothing.
            hough_threshold: Hough accumulator threshold.
            min_line_length: Minimum line segment length for HoughLinesP.
            max_line_gap: Maximum allowed gap between points on the same line.
            lane_width_meters: Standard highway lane width in meters (3.7m typical).
            history_len: Frame count for exponential temporal smoothing.
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gaussian_kernel = gaussian_kernel
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.lane_width_meters = lane_width_meters

        # Temporal smoothing buffers for stable line projection
        self.left_fit_history = deque(maxlen=history_len)
        self.right_fit_history = deque(maxlen=history_len)

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Execute full lane detection pipeline on a single frame.

        Args:
            frame: Input BGR frame.

        Returns:
            Dictionary containing:
            - 'left_line': ((x1, y1), (x2, y2))
            - 'right_line': ((x1, y1), (x2, y2))
            - 'lane_center_x': float
            - 'vehicle_center_x': float
            - 'lateral_offset_cm': float (positive = right of center, negative = left)
            - 'offset_status': str ('CENTERED', 'DEVIATING_LEFT', 'DEVIATING_RIGHT', 'LANE_DEPARTURE')
            - 'lane_polygon': np.ndarray of polygon coordinates for corridor fill
            - 'canny_edges': np.ndarray (debug edge map)
            - 'roi_edges': np.ndarray (masked edge map)
            - 'is_tracking_valid': bool
        """
        h, w = frame.shape[:2]
        vehicle_center_x = w / 2.0

        # Step 1: Color Isolation for Yellow and White Lane Markings
        color_mask = self._isolate_lane_colors(frame)

        # Step 2: Grayscale and Gaussian Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        combined = cv2.bitwise_and(gray, gray, mask=color_mask)
        blur = cv2.GaussianBlur(combined, (self.gaussian_kernel, self.gaussian_kernel), 0)

        # Step 3: Canny Edge Detection
        canny_edges = cv2.Canny(blur, self.canny_low, self.canny_high)

        # Step 4: Region of Interest (Trapezoid Mask)
        roi_edges = self._apply_roi_mask(canny_edges)

        # Step 5: Probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        # Step 6: Slope segregation and line fitting
        left_line, right_line, is_valid = self._fit_lane_lines(lines, h, w)

        # Step 7: Telemetry & Lane Precision Calculation
        lateral_offset_cm = 0.0
        offset_status = "CENTERED"
        lane_center_x = vehicle_center_x
        lane_polygon = None

        if is_valid and left_line is not None and right_line is not None:
            (lx1, ly1), (lx2, ly2) = left_line
            (rx1, ry1), (rx2, ry2) = right_line

            # Calculate bottom coordinates (at bumper level y = h)
            left_bottom_x = lx1
            right_bottom_x = rx1
            lane_width_pixels = max(10, right_bottom_x - left_bottom_x)

            # Calculated center of the lane at vehicle base
            lane_center_x = (left_bottom_x + right_bottom_x) / 2.0

            # Pixel to metric scale (centimeters per pixel at bumper depth)
            cm_per_pixel = (self.lane_width_meters * 100.0) / float(lane_width_pixels)

            # Lateral offset: positive means vehicle is shifted to right of lane center
            lateral_offset_px = vehicle_center_x - lane_center_x
            lateral_offset_cm = round(lateral_offset_px * cm_per_pixel, 1)

            # Determine Lane Departure / Precision Status
            abs_offset = abs(lateral_offset_cm)
            if abs_offset <= 15.0:
                offset_status = "CENTERED"
            elif 15.0 < abs_offset <= 35.0:
                offset_status = "DEVIATING_RIGHT" if lateral_offset_cm > 0 else "DEVIATING_LEFT"
            else:
                offset_status = "LANE_DEPARTURE"

            # Create driving corridor polygon for HUD rendering
            lane_polygon = np.array([
                [lx1, ly1],
                [lx2, ly2],
                [rx2, ry2],
                [rx1, ry1]
            ], dtype=np.int32)

        return {
            "left_line": left_line,
            "right_line": right_line,
            "lane_center_x": round(lane_center_x, 1),
            "vehicle_center_x": round(vehicle_center_x, 1),
            "lateral_offset_cm": lateral_offset_cm,
            "offset_status": offset_status,
            "lane_polygon": lane_polygon,
            "canny_edges": canny_edges,
            "roi_edges": roi_edges,
            "is_tracking_valid": is_valid
        }

    def _isolate_lane_colors(self, frame: np.ndarray) -> np.ndarray:
        """Isolate white and yellow pixels common to road lane markings."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White lane markings
        lower_white = np.array([0, 0, 180], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Yellow lane markings
        lower_yellow = np.array([15, 80, 100], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        return cv2.bitwise_or(white_mask, yellow_mask)

    def _apply_roi_mask(self, edges: np.ndarray) -> np.ndarray:
        """Apply dynamic trapezoidal perspective ROI mask on road surface."""
        h, w = edges.shape[:2]
        mask = np.zeros_like(edges)

        # Dynamic trapezoid coordinates matching vehicle forward perspective
        polygon = np.array([[
            (int(w * 0.08), int(h * 0.95)),  # Bottom-left
            (int(w * 0.42), int(h * 0.58)),  # Apex-left
            (int(w * 0.58), int(h * 0.58)),  # Apex-right
            (int(w * 0.92), int(h * 0.95))   # Bottom-right
        ]], dtype=np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _fit_lane_lines(
        self,
        lines: Optional[np.ndarray],
        height: int,
        width: int
    ) -> Tuple[Optional[Tuple[Tuple[int, int], Tuple[int, int]]], Optional[Tuple[Tuple[int, int], Tuple[int, int]]], bool]:
        """
        Group Hough lines by slope, filter outliers, and apply temporal smoothing.
        """
        if lines is None or len(lines) == 0:
            return self._use_cached_fit(height, width)

        left_points_x = []
        left_points_y = []
        right_points_x = []
        right_points_y = []

        mid_x = width / 2.0

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / float(x2 - x1)
                
                # Filter out nearly horizontal noise lines
                if abs(slope) < 0.3 or abs(slope) > 3.0:
                    continue

                # Negative slope -> Left lane (in image coordinates where y increases downwards)
                if slope < -0.3 and x1 < (mid_x + 60) and x2 < (mid_x + 60):
                    left_points_x.extend([x1, x2])
                    left_points_y.extend([y1, y2])
                # Positive slope -> Right lane
                elif slope > 0.3 and x1 > (mid_x - 60) and x2 > (mid_x - 60):
                    right_points_x.extend([x1, x2])
                    right_points_y.extend([y1, y2])

        y_bottom = height
        y_top = int(height * 0.60)

        # Fit left lane
        left_fit = None
        if len(left_points_y) >= 4:
            poly = np.polyfit(left_points_y, left_points_x, 1)
            self.left_fit_history.append(poly)
            left_fit = np.mean(self.left_fit_history, axis=0)
        elif len(self.left_fit_history) > 0:
            left_fit = np.mean(self.left_fit_history, axis=0)

        # Fit right lane
        right_fit = None
        if len(right_points_y) >= 4:
            poly = np.polyfit(right_points_y, right_points_x, 1)
            self.right_fit_history.append(poly)
            right_fit = np.mean(self.right_fit_history, axis=0)
        elif len(self.right_fit_history) > 0:
            right_fit = np.mean(self.right_fit_history, axis=0)

        left_line = None
        right_line = None

        if left_fit is not None:
            lx_bottom = int(np.polyval(left_fit, y_bottom))
            lx_top = int(np.polyval(left_fit, y_top))
            left_line = ((lx_bottom, y_bottom), (lx_top, y_top))

        if right_fit is not None:
            rx_bottom = int(np.polyval(right_fit, y_bottom))
            rx_top = int(np.polyval(right_fit, y_top))
            right_line = ((rx_bottom, y_bottom), (rx_top, y_top))

        is_valid = (left_line is not None and right_line is not None)
        return left_line, right_line, is_valid

    def _use_cached_fit(
        self, height: int, width: int
    ) -> Tuple[Optional[Tuple[Tuple[int, int], Tuple[int, int]]], Optional[Tuple[Tuple[int, int], Tuple[int, int]]], bool]:
        """Fallback to smoothed history when current frame has momentary occlusion."""
        if len(self.left_fit_history) > 0 and len(self.right_fit_history) > 0:
            y_bottom = height
            y_top = int(height * 0.60)
            left_fit = np.mean(self.left_fit_history, axis=0)
            right_fit = np.mean(self.right_fit_history, axis=0)

            left_line = ((int(np.polyval(left_fit, y_bottom)), y_bottom),
                         (int(np.polyval(left_fit, y_top)), y_top))
            right_line = ((int(np.polyval(right_fit, y_bottom)), y_bottom),
                          (int(np.polyval(right_fit, y_top)), y_top))
            return left_line, right_line, True

        return None, None, False
