import json
import os
from datetime import datetime
from pathlib import Path

from huggingface_hub import create_repo, upload_file

from ultralytics import YOLO

REPO = Path(os.getcwd())
DATA_YAML = (REPO / "datasets" / "tradingview" / "data.yml").resolve().as_posix()
PROJECT = "Ultralytics"
HF_REPO_ID = "StephanAkkerman/chart-info-detector"
MLOPS_STATE = REPO / "mlops_state.json"

## Model settings
YOLO_MODEL = "yolo12n"
IMAGE_SIZE = 1792  # bigger means more VRAM
EPOCHS = 80


def make_run_name(imgsz: int, epochs: int) -> str:
    """Create a descriptive, mostly unique run name."""
    model_short = YOLO_MODEL.replace("yolo", "")  # "12n" from "yolo12n.pt"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")  # 20250110-221530
    return f"{model_short}_img{imgsz}_e{epochs}_{ts}"


# -----------------
# MLOps helpers
# -----------------


def load_state() -> dict:
    if MLOPS_STATE.exists():
        return json.loads(MLOPS_STATE.read_text(encoding="utf-8"))
    # default: no best yet
    return {"best_test_map50": 0.0, "best_run_name": None}


def save_state(state: dict) -> None:
    MLOPS_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_run_dir(run_name: str) -> Path:
    return REPO / PROJECT / run_name


def auto_upload_to_hf(run_name: str, test_map50: float) -> None:
    """
    Upload best.pt + best.onnx + results.csv for this run to Hugging Face.
    Assumes you already did `huggingface-cli login` or set HF_TOKEN.
    """
    run_dir = get_run_dir(run_name)
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    best_onnx = next(weights_dir.glob("*.onnx"), None)
    results_csv = run_dir / "results.csv"

    if not best_pt.exists():
        print("[HF] best.pt not found, skipping upload.")
        return

    create_repo(HF_REPO_ID, repo_type="model", private=False, exist_ok=True)

    print(f"[HF] Uploading best.pt (mAP50={test_map50:.4f}) to {HF_REPO_ID} ...")
    upload_file(
        path_or_fileobj=best_pt,
        path_in_repo="weights/best.pt",
        repo_id=HF_REPO_ID,
        repo_type="model",
    )

    if best_onnx is not None:
        upload_file(
            path_or_fileobj=best_onnx,
            path_in_repo="weights/best.onnx",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )

    if results_csv.exists():
        upload_file(
            path_or_fileobj=results_csv,
            path_in_repo="results.csv",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )

    print(f"[HF] Upload complete: https://huggingface.co/{HF_REPO_ID}")


# -----------------
# Training
# -----------------


def main(resume: bool = False, run_name: str | None = None) -> None:
    # if no run_name passed, create one for fresh training
    if not resume:
        run_name = run_name or make_run_name(IMAGE_SIZE, EPOCHS)
    else:
        # when resuming, you *must* pass the existing run_name explicitly
        if run_name is None:
            raise ValueError(
                "resume=True requires an explicit run_name to resume from."
            )
        print(f"Resuming training from run '{run_name}'")

    run_dir = REPO / PROJECT / run_name

    if resume:
        print("[Train] Resuming training from last.pt")
        ckpt = run_dir / "weights" / "last.pt"
        model = YOLO(ckpt.as_posix())
    else:
        model = YOLO(f"{YOLO_MODEL}.pt")

    # Train
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=0.9,
        seed=42,
        device=0,
        workers=0,
        optimizer="auto",
        cos_lr=True,
        amp=True,
        project=PROJECT,
        name=run_name,
        exist_ok=True,
        verbose=False,
        resume=resume,
        plots=True,
        save_json=True,
    )

    # Export ONNX (this will go into the run's weights dir)
    model.export(format="onnx", opset=17, dynamic=True)

    # Evaluate on test split
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMAGE_SIZE,
        batch=4,
        workers=0,
        rect=False,
        plots=True,
        save_json=True,
    )

    # Ultralytics metrics object usually has box.map50
    test_map50 = float(getattr(metrics.box, "map50", 0.0))
    print(f"[Eval] Test mAP50: {test_map50:.4f}")

    # -----------------
    # MLOps "is this better?"
    # -----------------
    state = load_state()
    best_so_far = float(state.get("best_test_map50", 0.0))

    if test_map50 > best_so_far:
        print(f"[MLOps] New best model! {test_map50:.4f} > {best_so_far:.4f}")
        state["best_test_map50"] = test_map50
        state["best_run_name"] = run_name
        save_state(state)

        # Auto-upload to HF
        auto_upload_to_hf(run_name=run_name, test_map50=test_map50)
    else:
        print(f"[MLOps] Not better than best ({best_so_far:.4f}). Skipping upload.")


if __name__ == "__main__":
    main()
