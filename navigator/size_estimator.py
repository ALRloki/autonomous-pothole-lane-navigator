"""
Pothole Dimension and Severity Estimation Module.
Calculates real-world metric dimensions (cm), surface area (cm^2),
and safety risk classification using perspective-corrected scaling.
"""

import numpy as np
from typing import Dict, Any, List


class PotholeSizeEstimator:
    """
    Translates image pixel bounding boxes into physical metric dimensions (cm),
    surface area (cm^2), and categorizes pothole severity.
    """

    def __init__(
        self,
        lane_width_meters: float = 3.7,
        camera_height_meters: float = 1.3,
        camera_pitch_deg: float = 10.0
    ):
        """
        Initialize the size estimator with road perspective parameters.

        Args:
            lane_width_meters: Known reference lane width in meters.
            camera_height_meters: Camera mount height above ground.
            camera_pitch_deg: Downward pitch angle of vehicle camera.
        """
        self.lane_width_meters = lane_width_meters
        self.camera_height_meters = camera_height_meters
        self.camera_pitch_deg = camera_pitch_deg

    def estimate_dimensions(
        self,
        detection: Dict[str, Any],
        frame_shape: tuple,
        lane_width_px_bottom: float = 550.0
    ) -> Dict[str, Any]:
        """
        Estimate physical dimensions and severity for a single pothole detection.

        Args:
            detection: Dictionary with 'bbox': [x1, y1, x2, y2], 'confidence', etc.
            frame_shape: (height, width) of the input video frame.
            lane_width_px_bottom: Estimated lane pixel width at the bottom of the frame.

        Returns:
            Dictionary with added fields:
            - 'width_cm': float
            - 'length_cm': float
            - 'area_cm2': float
            - 'severity': str ('MINOR', 'MODERATE', 'CRITICAL')
            - 'hazard_color_bgr': (B, G, R)
            - 'estimated_distance_m': float
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = detection["bbox"]
        pw_px = max(1, x2 - x1)
        ph_px = max(1, y2 - y1)
        cy = (y1 + y2) / 2.0

        # Perspective ground scaling:
        # Potholes closer to the horizon (smaller y) cover more physical meters per pixel.
        y_horizon = h * 0.48
        clamped_y = max(y_horizon + 10.0, min(float(h), cy))
        
        # Relative depth ratio (0.0 at horizon, 1.0 at vehicle bumper)
        depth_ratio = (clamped_y - y_horizon) / (h - y_horizon)
        
        # Pixels per centimeter at bottom bumper level
        base_px_per_cm = (lane_width_px_bottom / (self.lane_width_meters * 100.0))
        
        # Perspective scaling factor with non-linear foreshortening
        px_per_cm_at_y = max(0.08, base_px_per_cm * (depth_ratio ** 1.35))
        
        # Metric dimensions in centimeters
        width_cm = round(pw_px / px_per_cm_at_y, 1)
        
        # Longitudinal length accounts for forward foreshortening perspective compression
        foreshortening_factor = max(1.0, 1.8 / (depth_ratio + 0.1))
        length_cm = round((ph_px * foreshortening_factor) / px_per_cm_at_y, 1)
        
        # Realistic bounds clipping
        width_cm = max(10.0, min(250.0, width_cm))
        length_cm = max(8.0, min(220.0, length_cm))

        # Surface area approximated by an ellipse (pi * (w/2) * (l/2))
        area_cm2 = round((np.pi * (width_cm / 2.0) * (length_cm / 2.0)), 1)

        # Longitudinal distance to vehicle bumper in meters
        # Inverted depth: bumper is 1.5m away, horizon is 35m away
        distance_m = round(1.5 + (1.0 - depth_ratio) * 32.0, 1)

        # Severity Classification
        if area_cm2 < 200.0:
            severity = "MINOR"
            hazard_color = (0, 220, 255)   # Yellow/Cyan (Advisory)
        elif 200.0 <= area_cm2 < 550.0:
            severity = "MODERATE"
            hazard_color = (0, 140, 255)   # Amber/Orange (Caution)
        else:
            severity = "CRITICAL"
            hazard_color = (0, 0, 255)     # Red (Urgent Danger)

        enhanced_det = dict(detection)
        enhanced_det.update({
            "width_cm": width_cm,
            "length_cm": length_cm,
            "area_cm2": area_cm2,
            "severity": severity,
            "hazard_color_bgr": hazard_color,
            "distance_m": distance_m
        })

        return enhanced_det

    def batch_estimate(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: tuple,
        lane_width_px_bottom: float = 550.0
    ) -> List[Dict[str, Any]]:
        """Batch process all detected potholes in a frame."""
        return [
            self.estimate_dimensions(d, frame_shape, lane_width_px_bottom)
            for d in detections
        ]
