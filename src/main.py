# train_yolo_detector.py
from pathlib import Path

from ultralytics import YOLO

# --- paths (use absolute paths on Windows to avoid CWD issues) ---
REPO = Path(r"E:/GitHub/chart-info-detector")  # ← change if needed
DATA_YAML = REPO / "datasets/tradingview/data.yaml"  # your data.yaml
WEIGHTS = "yolo12n.pt"  # or a local .pt path


def main() -> None:
    # Create/choose a save dir (Ultralytics will make subfolders)
    save_dir = REPO / "runs" / "detect"

    model = YOLO(WEIGHTS)

    # Train
    model.train(
        data=str(DATA_YAML),  # absolute path safest on Windows
        epochs=60,
        imgsz=1280,  # UI text is small; 1280–1536 helps
        batch=16,
        seed=42,
        device=0,  # GPU index; use "cpu" if no GPU
        workers=0,  # Windows: avoid multi-worker DataLoader quirks
        optimizer="auto",
        cos_lr=True,  # cosine LR schedule
        amp=True,  # mixed precision (faster on GPU)
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=True,
    )

    # Evaluate on the test split explicitly
    best = model  # model is updated to best after train()
    best.val(data=str(DATA_YAML), split="test", imgsz=1280, batch=16, workers=0)

    # Export for inference (ONNX)
    best.export(format="onnx", opset=17, dynamic=True)


if __name__ == "__main__":
    main()
