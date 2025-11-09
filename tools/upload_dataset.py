from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Image

ROOT = Path("datasets/tradingview")
IMG_DIR = ROOT / "images"
LBL_DIR = ROOT / "labels"
REPO_ID = "StephanAkkerman/chart-info-yolo"


def load_split(split: str) -> Dataset:
    """
    Build a Dataset for one split from YOLO-style images/labels.
    - images/<split>/*.png
    - labels/<split>/*.txt  (same stem; may be missing or empty)
    """
    img_dir = IMG_DIR / split
    lbl_dir = LBL_DIR / split

    records: List[Dict[str, Any]] = []

    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        boxes: List[List[float]] = []
        labels: List[int] = []

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:5]]  # [x_c, y_c, w, h]
                labels.append(cid)
                boxes.append(coords)

        records.append(
            {
                "image": str(img_path.resolve()),
                "split": split,
                "bboxes": boxes,  # list[list[float]]
                "labels": labels,  # list[int]
            }
        )

    # Let datasets infer types, then cast image to Image()
    ds = Dataset.from_list(records)
    ds = ds.cast_column("image", Image())
    return ds


def main() -> None:
    splits = {}
    for split in ("train", "val", "test"):
        if (IMG_DIR / split).exists():
            print(f"Building split: {split}")
            splits[split] = load_split(split)

    dsd = DatasetDict(splits)
    print(dsd)

    dsd.push_to_hub(REPO_ID)
    print(f"Pushed to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
