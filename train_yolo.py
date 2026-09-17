"""
YOLOv11 Pothole Detector Training Pipeline.
Prepares dataset configuration, fine-tunes YOLOv11 architecture on pothole images,
and exports production weights to models/pothole_yolo11.pt.
"""

import os
import shutil
import yaml
from ultralytics import YOLO


def prepare_dataset_yaml(base_dir: str = "sample_data") -> str:
    """Creates YOLO format data.yaml specification."""
    abs_base = os.path.abspath(base_dir)
    images_dir = os.path.join(abs_base, "images")

    data_config = {
        "path": abs_base,
        "train": images_dir,
        "val": images_dir,
        "names": {
            0: "pothole"
        }
    }

    yaml_path = os.path.join(abs_base, "pothole_dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)

    print(f"[+] Dataset YAML created at: {yaml_path}")
    return yaml_path


def train_pothole_detector(epochs: int = 15, batch_size: int = 4, imgsz: int = 640):
    """Fine-tunes YOLOv11 on pothole dataset and saves export weights."""
    print("==================================================================")
    print("      Training YOLOv11 Architecture for Pothole Sensing          ")
    print("==================================================================")

    yaml_path = prepare_dataset_yaml()

    # Load baseline YOLOv11 nano model
    model = YOLO("yolo11n.pt")

    print(f"[*] Starting YOLOv11 fine-tuning ({epochs} epochs)...")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device="cpu",
        plots=False,
        save=True,
        verbose=True
    )

    # Copy best weights to models/pothole_yolo11.pt
    os.makedirs("models", exist_ok=True)
    best_weights = str(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else None

    target_weights = "models/pothole_yolo11.pt"
    if best_weights and os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"[+] Best YOLOv11 weights saved to: {target_weights}")
    else:
        # Save model state directly
        model.save(target_weights)
        print(f"[+] Model checkpoint exported to: {target_weights}")

    print("==================================================================")
    print("               YOLOv11 Training Completed Successfully           ")
    print("==================================================================")


if __name__ == "__main__":
    train_pothole_detector(epochs=10)
