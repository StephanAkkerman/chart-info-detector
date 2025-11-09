import os
from pathlib import Path

from ultralytics import YOLO

REPO = Path(os.getcwd())
DATA_YAML = REPO / "datasets/tradingview/data.yml"
WEIGHTS = "yolo12n.pt"  # or absolute path to .pt
IMAGE_SIZE = 1536  # 1280–1536 good for small UI text
# 1) Hard sanity checks BEFORE calling Ultralytics


def main(resume: bool = False) -> None:
    # Create/choose a save dir (Ultralytics will make subfolders)
    run_name = WEIGHTS.replace(".pt", "").replace("yolo", "")

    if resume:
        print("Resuming training from last.pt")
        model = YOLO(REPO / "Ultralytics" / run_name / "weights" / "last.pt")
    else:
        model = YOLO(WEIGHTS)

    # Train
    model.train(
        data=DATA_YAML,  # absolute path safest on Windows
        epochs=80,
        imgsz=IMAGE_SIZE,
        batch=0.9,  # Use 90% of GPU VRAM
        seed=42,
        device=0,  # GPU index; use "cpu" if no GPU
        workers=0,  # Windows: avoid multi-worker DataLoader quirks
        optimizer="auto",
        cos_lr=True,  # cosine LR schedule
        amp=True,  # mixed precision (faster on GPU)
        project="Ultralytics",
        name=run_name,
        exist_ok=True,
        verbose=False,  # less output
        resume=resume,  # resume from last train() if possible
    )
    best = model  # model is updated to best after train()

    # Export for inference (ONNX)
    best.export(format="onnx", opset=17, dynamic=True)

    # Evaluate on the test split explicitly
    best.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMAGE_SIZE,
        batch=4,
        workers=0,
        rect=False,
        plots=True,
        save_json=True,
    )

    # Upload to huggingface


if __name__ == "__main__":
    main()
