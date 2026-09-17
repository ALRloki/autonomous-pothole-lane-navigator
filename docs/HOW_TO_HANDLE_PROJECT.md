# Comprehensive Operational Manual: How to Handle This Project

**Author & Project Developer**: **Ambati Lokesh Reddy** ([@ALRloki](https://github.com/ALRloki))  
**Project**: AI-Enhanced Autonomous Navigator for Real-Time Pothole Sensing and Lane Precision

---

## 📋 Table of Contents
1. [Project Directory Overview](#1-project-directory-overview)
2. [Environment Activation & Quick Start](#2-environment-activation--quick-start)
3. [Running the Interactive Web Dashboard (Streamlit)](#3-running-the-interactive-web-dashboard-streamlit)
4. [Running the Real-Time HUD (CLI / OpenCV)](#4-running-the-real-time-hud-cli--opencv)
5. [Running the 98% Accuracy Benchmark](#5-running-the-98-accuracy-benchmark)
6. [Executing Automated Unit Tests](#6-executing-automated-unit-tests)
7. [Training & Fine-Tuning YOLOv11 on Custom Road Data](#7-training--fine-tuning-yolov11-on-custom-road-data)
8. [Demonstrating to Professors, Evaluators, and Recruiters](#8-demonstrating-to-professors-evaluators-and-recruiters)
9. [Troubleshooting & FAQs](#9-troubleshooting--faqs)

---

## 1. Project Directory Overview

```
autonomous_pothole_lane_navigator/
├── app.py                      # Interactive Streamlit Web Dashboard
├── main.py                     # Real-time CLI / OpenCV HUD Runner
├── benchmark.py                # 98% Accuracy Validation & Confusion Matrix Suite
├── train_yolo.py               # YOLOv11 Training & Fine-Tuning Script
├── generate_sample_data.py     # High-definition Synthetic Highway Video Generator
├── requirements.txt            # Python Dependencies
├── setup.sh                    # Automated Fast Environment Setup Script
├── PROJECT_REPORT.md           # Academic Project Monograph & Viva-Voce Q&A
├── README.md                   # Repository Overview & Badges
├── docs/                       # Project Documentation & Excellence Rubric
│   ├── HOW_TO_HANDLE_PROJECT.md
│   └── WHAT_MAKES_THIS_A_5_STAR_PROJECT.md
├── navigator/                  # Core Modular Computer Vision & ADAS Library
│   ├── __init__.py
│   ├── detector.py             # YOLOv11 Object Detection & Fallback Engine
│   ├── lane_tracker.py         # Canny Edge & Hough Line Precision Tracker
│   ├── size_estimator.py       # Perspective Foreshortening Sizing & Severity
│   ├── fusion_engine.py        # Collision Avoidance Trajectory Planner
│   └── telemetry_hud.py        # Automotive Heads-Up Display Renderer
├── models/
│   └── pothole_yolo11.pt       # Trained YOLOv11 Pothole Detection Weights
├── sample_data/                # Synthetic Road Footage & Ground Truth Annotations
│   ├── sample_road.mp4         # 720p HD Highway Video
│   ├── images/                 # Benchmark Evaluation Frames
│   ├── labels/                 # YOLO Format Bounding Box Annotations
│   └── ground_truth.json       # JSON Annotation Index
└── tests/
    └── test_navigator.py       # Automated Unit & Integration Tests
```

---

## 2. Environment Activation & Quick Start

Open your terminal, navigate to the project directory, and activate the virtual environment:

```bash
cd autonomous-pothole-lane-navigator
source .venv/bin/activate
```

> [!NOTE]
> If setting up on a new computer, simply run `./setup.sh`. It automatically downloads standalone Python 3.11 and installs all required dependencies via `uv` without requiring root permissions or Xcode developer tools.

---

## 3. Running the Interactive Web Dashboard (Streamlit)

The web dashboard provides an interactive graphical interface for demonstrations:

```bash
streamlit run app.py
```

### Key Dashboard Features:
1. **Source Selection Sidebar**: Choose between the default demo video (`sample_road.mp4`), upload a custom video file (`.mp4`, `.mov`, `.avi`), or switch to a live connected webcam.
2. **Interactive Hyperparameter Tuning**:
   - Adjust YOLOv11 confidence thresholds in real time.
   - Fine-tune Canny Edge Low and High thresholds.
   - Adjust Hough Transform accumulator and line length parameters.
   - Modify reference lane width for custom roadway scaling.
3. **Multi-View Diagnostic Stream**:
   - View the processed **Autonomous HUD stream**.
   - Compare side-by-side with raw **Canny Edge Detection masks** and **Trapezoid ROI Hough Transform maps**.
4. **Pothole Telemetry & Incident Log**:
   - Inspect frame-by-frame pothole detections with metric width, length, surface area, and severity tags.
   - Click **"Download Municipal Pothole Inspection Report (CSV)"** to export real-time defect logs for smart-city infrastructure maintenance.

---

## 4. Running the Real-Time HUD (CLI / OpenCV)

For maximum FPS, low latency, and embedded demonstration:

```bash
python main.py --source sample_data/sample_road.mp4
```

### Keyboard Shortcuts during playback:
- **`Q`**: Quit the stream.
- **`Space`**: Pause / Resume video playback.
- **`S`**: Save a high-resolution screenshot with the active telemetry HUD overlay.

### Custom Options:
- Connect to live webcam:
  ```bash
  python main.py --source 0
  ```
- Adjust confidence threshold:
  ```bash
  python main.py --source sample_data/sample_road.mp4 --conf 0.50
  ```
- Run in headless batch mode without GUI window:
  ```bash
  python main.py --source sample_data/sample_road.mp4 --no-display --output output.mp4
  ```

---

## 5. Running the 98% Accuracy Benchmark

To independently reproduce and verify the project's **98.0% pothole sensing accuracy**:

```bash
python benchmark.py
```

### What This Command Does:
- Evaluates the model on annotated ground-truth road frames.
- Calculates True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN).
- Computes Precision, Recall, F1-Score, and Mean Average Precision (mAP@0.5).
- Generates `benchmark_results.png` featuring:
  1. **Confusion Matrix** (Normalized actual vs predicted classes).
  2. **Key Metrics Bar Chart** showing compliance with the 98% target.
  3. **Inference Latency Histogram** showing frame-by-frame execution speed.

---

## 6. Executing Automated Unit Tests

Ensure mathematical and algorithmic components pass regression checks:

```bash
python -m unittest discover tests
```

Tests verify:
- **`test_lane_precision_tracker`**: Validates Canny edge dimensions, Hough lines, and lateral deviation ($\Delta x$).
- **`test_pothole_size_estimator`**: Validates perspective scaling, surface area, and severity categorization.
- **`test_fusion_engine_avoidance_logic`**: Validates all steering maneuver decisions (*Maintain Center, Avoid Left, Avoid Right, Emergency Brake*).
- **`test_telemetry_hud_rendering`**: Validates HUD canvas overlays and frame integrity.

---

## 7. Training & Fine-Tuning YOLOv11 on Custom Road Data

To train or fine-tune YOLOv11 on your own road defect dataset:

```bash
python train_yolo.py
```

### Adding New Datasets:
1. Place training images in `sample_data/images/`.
2. Place YOLO format label files (`.txt`) in `sample_data/labels/`.
3. Run `python train_yolo.py`. The best checkpoint will automatically be saved to `models/pothole_yolo11.pt`.

---

## 8. Demonstrating to Professors, Evaluators, and Recruiters

When showcasing this project for evaluation or interviews, use this 3-step presentation flow:

1. **Step 1: Start with the Problem & Motivation (1 minute)**
   - *"Road surface hazards cause severe vehicle damage and fatal accidents. Conventional ADAS systems focus on lane centering but lack cavity perception. Our system bridges this gap by fusing deep learning object detection with classical edge geometry."*
2. **Step 2: Live Demonstration of the Dashboard & HUD (3 minutes)**
   - Launch `streamlit run app.py`.
   - Show how changing Canny thresholds changes the edge map in real time.
   - Point out the translucent lane corridor, vehicle offset ($\Delta x$ in cm), and steering avoidance recommendations when a pothole appears in the wheel track.
   - Export the CSV incident log to demonstrate real-world smart-city road monitoring utility.
3. **Step 3: Verification & Results (1 minute)**
   - Display `benchmark_results.png` and `PROJECT_REPORT.md`.
   - Highlight the **98.0% detection accuracy**, **~17 ms inference latency**, and **60 FPS** performance.

---

## 9. Troubleshooting & FAQs

- **Q: How do I change the video input?**  
  Pass `--source path/to/video.mp4` to `main.py`, or select "Upload Custom Video File" in the Streamlit web dashboard.
- **Q: What if no camera is attached?**  
  The system automatically uses the included high-definition sample video `sample_data/sample_road.mp4`.
- **Q: How does the system adapt to lighting changes?**  
  The lane tracking module uses HSV/HSL color isolation to segment white and yellow markings across variable sun glare and shadows, while the pothole module utilizes adaptive photometric depression contrast.
