"""
AI-Enhanced Autonomous Navigator for Real-Time Pothole Sensing and Lane Precision
Modular Computer Vision and ADAS Architecture
"""

from .detector import PotholeDetector
from .lane_tracker import LanePrecisionTracker
from .size_estimator import PotholeSizeEstimator
from .fusion_engine import AutonomousFusionEngine
from .telemetry_hud import TelemetryHUD

__version__ = "1.0.0"
__all__ = [
    "PotholeDetector",
    "LanePrecisionTracker",
    "PotholeSizeEstimator",
    "AutonomousFusionEngine",
    "TelemetryHUD"
]
