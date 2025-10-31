import io
import random
from pathlib import Path

from PIL import Image

from datasets import load_dataset

SEED = 42
NUM = 150
DATASET = "StephanAkkerman/stock-charts"
SPLIT = "train"
OUTDIR = Path("datasets/tradingview/images/raw")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Sample 150 images with label==0 ('charts') deterministically and save as PNGs.

    Notes
    -----
    - Uses a fixed RNG seed for reproducibility.
    - Expects a column 'label' and an 'image' column with bytes.
    """
    ds = load_dataset(DATASET, split=SPLIT)
    ds_charts = ds.filter(lambda ex: ex.get("label") == 0)

    # deterministic shuffle & take first NUM
    idx = list(range(len(ds_charts)))
    random.Random(SEED).shuffle(idx)
    idx = idx[:NUM]

    for i, j in enumerate(idx):
        ex = ds_charts[int(j)]
        # HF image column can be PIL Image, bytes, or path-like
        img = ex.get("image")
        if isinstance(img, Image.Image):
            pil = img.convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            # datasets.Image can also give a path/array; fallback:
            pil = Image.open(img).convert("RGB")  # if it's a path

        pil.save(OUTDIR / f"chart_{i:04d}.png")

    print(f"Saved {NUM} images to {OUTDIR}")


if __name__ == "__main__":
    main()
