"""
Autonomous Fusion Engine for Trajectory Planning and Obstacle Avoidance.
Combines lane corridor geometry with detected pothole hazards to provide
safe steering adjustments and speed advisory commands.
"""

from typing import Dict, Any, List, Tuple


class AutonomousFusionEngine:
    """
    Synthesizes perception inputs from LanePrecisionTracker and PotholeDetector.
    Computes tire collision risk and provides autonomous navigation directives.
    """

    def __init__(self, vehicle_wheelbase_m: float = 2.7, vehicle_width_m: float = 1.9):
        self.wheelbase = vehicle_wheelbase_m
        self.vehicle_width = vehicle_width_m

    def compute_navigation_directive(
        self,
        lane_info: Dict[str, Any],
        potholes: List[Dict[str, Any]],
        frame_shape: tuple
    ) -> Dict[str, Any]:
        """
        Evaluate trajectory and generate steering/speed recommendations.

        Args:
            lane_info: Output dictionary from LanePrecisionTracker.
            potholes: Output list of enhanced pothole detections.
            frame_shape: (height, width) of the video frame.

        Returns:
            Dictionary containing:
            - 'maneuver': str ('MAINTAIN_CENTER', 'AVOID_LEFT', 'AVOID_RIGHT', 'EMERGENCY_BRAKE')
            - 'alert_level': str ('SAFE', 'CAUTION', 'WARNING', 'CRITICAL')
            - 'recommended_steer_deg': float (negative = steer left, positive = steer right)
            - 'hazard_summary': str
            - 'threatened_wheel': str ('NONE', 'LEFT_WHEEL', 'RIGHT_WHEEL', 'BOTH')
            - 'threat_count': int
        """
        h, w = frame_shape[:2]
        vehicle_center_x = lane_info.get("vehicle_center_x", w / 2.0)
        lane_center_x = lane_info.get("lane_center_x", vehicle_center_x)
        lateral_offset_cm = lane_info.get("lateral_offset_cm", 0.0)

        # Baseline steering angle required to nullify lane departure (P-controller)
        # 1 cm offset translates to ~0.08 degrees corrective steering
        baseline_steer_deg = round(-0.08 * lateral_offset_cm, 1)

        # Determine approximate pixel boundaries of the ego-lane
        left_line = lane_info.get("left_line")
        right_line = lane_info.get("right_line")

        left_bound_x = left_line[0][0] if left_line else int(w * 0.15)
        right_bound_x = right_line[0][0] if right_line else int(w * 0.85)
        lane_width_px = max(100, right_bound_x - left_bound_x)

        # Wheel paths roughly at 25% and 75% across the vehicle lane envelope
        left_wheel_min = vehicle_center_x - (lane_width_px * 0.35)
        left_wheel_max = vehicle_center_x - (lane_width_px * 0.08)

        right_wheel_min = vehicle_center_x + (lane_width_px * 0.08)
        right_wheel_max = vehicle_center_x + (lane_width_px * 0.35)

        threatened_wheel = "NONE"
        highest_severity = "NONE"
        threat_count = 0
        closest_distance = 999.0
        pothole_on_left = False
        pothole_on_right = False
        pothole_in_center = False

        for p in potholes:
            px1, py1, px2, py2 = p["bbox"]
            dist_m = p.get("distance_m", 15.0)
            sev = p.get("severity", "MINOR")

            # Only consider potholes in forward collision zone (within 28 meters)
            if dist_m < 28.0:
                threat_count += 1
                closest_distance = min(closest_distance, dist_m)

                # Check lateral overlap with wheel paths
                overlaps_left = not (px2 < left_wheel_min or px1 > left_wheel_max)
                overlaps_right = not (px2 < right_wheel_min or px1 > right_wheel_max)
                overlaps_center = not (px2 < left_wheel_max or px1 > right_wheel_min)

                if overlaps_left:
                    pothole_on_left = True
                if overlaps_right:
                    pothole_on_right = True
                if overlaps_center:
                    pothole_in_center = True

                if sev == "CRITICAL" or highest_severity != "CRITICAL":
                    if sev == "MODERATE" and highest_severity != "CRITICAL":
                        highest_severity = "MODERATE"
                    elif sev == "CRITICAL":
                        highest_severity = "CRITICAL"
                    elif highest_severity == "NONE":
                        highest_severity = "MINOR"

        # Determine threatened wheel and avoidance maneuver
        recommended_steer_deg = baseline_steer_deg
        maneuver = "MAINTAIN_CENTER"
        alert_level = "SAFE"
        hazard_summary = "Path is clear. Maintain centered trajectory."

        if pothole_on_left and pothole_on_right:
            threatened_wheel = "BOTH"
            maneuver = "EMERGENCY_BRAKE"
            alert_level = "CRITICAL"
            hazard_summary = f"Multiple potholes spanning lane at {closest_distance}m! Decelerate immediately."
            recommended_steer_deg = 0.0

        elif pothole_on_left:
            threatened_wheel = "LEFT_WHEEL"
            if highest_severity == "CRITICAL" and closest_distance < 6.0:
                maneuver = "EMERGENCY_BRAKE"
                alert_level = "CRITICAL"
                hazard_summary = f"Critical pothole left wheel at {closest_distance}m! Brake & steer right."
                recommended_steer_deg = min(12.0, baseline_steer_deg + 6.0)
            else:
                maneuver = "AVOID_RIGHT"
                alert_level = "WARNING" if highest_severity == "CRITICAL" else "CAUTION"
                hazard_summary = f"Pothole left wheel at {closest_distance}m. Steer right within lane."
                recommended_steer_deg = min(10.0, baseline_steer_deg + 4.5)

        elif pothole_on_right:
            threatened_wheel = "RIGHT_WHEEL"
            if highest_severity == "CRITICAL" and closest_distance < 6.0:
                maneuver = "EMERGENCY_BRAKE"
                alert_level = "CRITICAL"
                hazard_summary = f"Critical pothole right wheel at {closest_distance}m! Brake & steer left."
                recommended_steer_deg = max(-12.0, baseline_steer_deg - 6.0)
            else:
                maneuver = "AVOID_LEFT"
                alert_level = "WARNING" if highest_severity == "CRITICAL" else "CAUTION"
                hazard_summary = f"Pothole right wheel at {closest_distance}m. Steer left within lane."
                recommended_steer_deg = max(-10.0, baseline_steer_deg - 4.5)

        elif pothole_in_center and highest_severity == "CRITICAL":
            threatened_wheel = "CENTER"
            maneuver = "AVOID_LEFT"
            alert_level = "CAUTION"
            hazard_summary = f"High-depth cavity in center at {closest_distance}m. Straddle left."
            recommended_steer_deg = max(-6.0, baseline_steer_deg - 3.0)

        elif lane_info.get("offset_status") == "LANE_DEPARTURE":
            alert_level = "WARNING"
            hazard_summary = f"Lane Departure Alert ({lateral_offset_cm}cm offset). Steer to recover center."

        return {
            "maneuver": maneuver,
            "alert_level": alert_level,
            "recommended_steer_deg": round(recommended_steer_deg, 1),
            "hazard_summary": hazard_summary,
            "threatened_wheel": threatened_wheel,
            "threat_count": threat_count,
            "closest_distance_m": closest_distance if threat_count > 0 else 0.0
        }
