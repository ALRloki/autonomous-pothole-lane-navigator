"""
Comprehensive Unit and Integration Test Suite for Autonomous Navigator.
Validates lane precision tracking, size estimation, fusion avoidance logic,
and telemetry heads-up display rendering.
"""

import os
import unittest
import numpy as np
import cv2

from navigator.detector import PotholeDetector
from navigator.lane_tracker import LanePrecisionTracker
from navigator.size_estimator import PotholeSizeEstimator
from navigator.fusion_engine import AutonomousFusionEngine
from navigator.telemetry_hud import TelemetryHUD
from generate_sample_data import create_realistic_road_frame


class TestAutonomousNavigator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.width = 1280
        cls.height = 720
        # Generate a sample frame with known lane lines and a pothole
        pothole_spec = [{
            "start_frame": 1,
            "end_frame": 10,
            "lane_fraction": 0.72,
            "base_width": 80,
            "base_height": 50,
            "depth_factor": 1.3,
            "severity": "MODERATE"
        }]
        cls.test_frame, cls.gt_boxes = create_realistic_road_frame(
            frame_idx=5, total_frames=10, width=cls.width, height=cls.height,
            pothole_specs=pothole_spec
        )

    def test_lane_precision_tracker(self):
        """Verify Canny edge detection, Hough Line Transform, and offset computation."""
        tracker = LanePrecisionTracker()
        result = tracker.process_frame(self.test_frame)

        self.assertIn("left_line", result)
        self.assertIn("right_line", result)
        self.assertIn("lateral_offset_cm", result)
        self.assertIn("offset_status", result)
        self.assertIn("canny_edges", result)
        self.assertIn("roi_edges", result)

        # Canny edge output must match input dimensions
        self.assertEqual(result["canny_edges"].shape, (self.height, self.width))
        self.assertEqual(result["roi_edges"].shape, (self.height, self.width))

        # Vehicle offset must be within reasonable roadway bounds (< 80 cm)
        offset = result["lateral_offset_cm"]
        self.assertLess(abs(offset), 80.0)

    def test_pothole_size_estimator(self):
        """Verify metric dimension scaling, surface area, and severity categorization."""
        estimator = PotholeSizeEstimator(lane_width_meters=3.7)
        dummy_det = {
            "bbox": [650, 480, 750, 540],
            "confidence": 0.92,
            "class_name": "pothole"
        }
        res = estimator.estimate_dimensions(dummy_det, (self.height, self.width))

        self.assertGreater(res["width_cm"], 10.0)
        self.assertGreater(res["length_cm"], 8.0)
        self.assertGreater(res["area_cm2"], 50.0)
        self.assertIn(res["severity"], ["MINOR", "MODERATE", "CRITICAL"])
        self.assertGreater(res["distance_m"], 0.0)

    def test_fusion_engine_avoidance_logic(self):
        """Verify autonomous trajectory decision-making for various obstacle scenarios."""
        fusion = AutonomousFusionEngine()
        lane_info = {
            "vehicle_center_x": 640.0,
            "lane_center_x": 640.0,
            "lateral_offset_cm": 0.0,
            "offset_status": "CENTERED",
            "left_line": ((280, 720), (620, 430)),
            "right_line": ((1000, 720), (660, 430))
        }

        # Scenario A: Path is totally clear
        directive_clear = fusion.compute_navigation_directive(lane_info, [], (self.height, self.width))
        self.assertEqual(directive_clear["maneuver"], "MAINTAIN_CENTER")
        self.assertEqual(directive_clear["alert_level"], "SAFE")

        # Scenario B: Pothole threatening right wheel
        pothole_right = [{
            "bbox": [750, 520, 830, 570],
            "confidence": 0.90,
            "distance_m": 12.0,
            "severity": "MODERATE"
        }]
        directive_right = fusion.compute_navigation_directive(lane_info, pothole_right, (self.height, self.width))
        self.assertEqual(directive_right["maneuver"], "AVOID_LEFT")
        self.assertLess(directive_right["recommended_steer_deg"], 0.0)  # Steer left is negative

        # Scenario C: Pothole threatening left wheel
        pothole_left = [{
            "bbox": [450, 520, 530, 570],
            "confidence": 0.90,
            "distance_m": 12.0,
            "severity": "MODERATE"
        }]
        directive_left = fusion.compute_navigation_directive(lane_info, pothole_left, (self.height, self.width))
        self.assertEqual(directive_left["maneuver"], "AVOID_RIGHT")
        self.assertGreater(directive_left["recommended_steer_deg"], 0.0)  # Steer right is positive

        # Scenario D: Severe dual hazard spanning lane
        potholes_both = [pothole_left[0], pothole_right[0]]
        directive_both = fusion.compute_navigation_directive(lane_info, potholes_both, (self.height, self.width))
        self.assertEqual(directive_both["maneuver"], "EMERGENCY_BRAKE")
        self.assertEqual(directive_both["alert_level"], "CRITICAL")

    def test_telemetry_hud_rendering(self):
        """Verify HUD overlay generation produces valid BGR frames."""
        hud = TelemetryHUD()
        tracker = LanePrecisionTracker()
        fusion = AutonomousFusionEngine()

        lane_info = tracker.process_frame(self.test_frame)
        directive = fusion.compute_navigation_directive(lane_info, [], self.test_frame.shape)

        annotated = hud.render(
            self.test_frame, lane_info, [], directive, fps=45.0, latency_ms=22.2
        )

        self.assertEqual(annotated.shape, self.test_frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
