"""Train the tuna detector.

Usage: python scripts/train.py <data.yaml> <run_name> [model=yolo11s.pt] [epochs=60]
Color/blur augmentation is pushed harder than default to mimic murky, green set-net water.
"""
import sys

from ultralytics import YOLO


def main():
    data, name = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "yolo11s.pt"
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    YOLO(f"weights/{model}").train(
        data=data, name=name, project="runs", epochs=epochs, imgsz=640, batch=16, device="mps",
        patience=20, seed=0, workers=4,
        hsv_h=0.04, hsv_s=0.7, hsv_v=0.5,  # water color varies a lot (blue / green / grey)
        degrees=10, flipud=0.2, fliplr=0.5, mosaic=1.0, mixup=0.1,
    )


if __name__ == "__main__":
    main()
