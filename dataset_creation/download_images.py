import io
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image

from datasets import load_dataset

SEED = 42
DATASET = "StephanAkkerman/stock-charts"
HF_SPLIT = "train"  # the split on HF to sample from
BASE_DIR = Path("local_datasets/tradingview/images")

# Set exact counts per split (deterministic)
SPLIT_COUNTS = {
    "train": 150,
    "val": 40,
    "test": 40,
}

LABEL_VALUE_FOR_CHART = 0  # keep only label==0


def _ensure_dirs() -> None:
    for s in SPLIT_COUNTS:
        (BASE_DIR / s).mkdir(parents=True, exist_ok=True)


def _save_pil(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path)


def _iter_filtered(ds) -> Iterable[tuple[int, Image.Image]]:
    """Yield (original_index, PIL.Image) for items with label==0."""
    for i, ex in enumerate(ds):
        if ex.get("label") != LABEL_VALUE_FOR_CHART:
            continue
        im = ex.get("image")
        if isinstance(im, Image.Image):
            yield i, im
        elif isinstance(im, dict) and "bytes" in im:
            yield i, Image.open(io.BytesIO(im["bytes"]))
        else:
            # datasets.Image can be a path/array as well
            yield i, Image.open(im)


def main() -> None:
    """
    Download images with label==0 from HF and split deterministically into
    train/val/test. Also writes a manifest with the original HF indices.

    Notes
    -----
    - Deterministic via SEED.
    - If the dataset has fewer items than requested, it will raise.
    """
    _ensure_dirs()
    ds = load_dataset(DATASET, split=HF_SPLIT)
    pool = list(_iter_filtered(ds))

    need_total = sum(SPLIT_COUNTS.values())
    if len(pool) < need_total:
        raise RuntimeError(
            f"Not enough label=={LABEL_VALUE_FOR_CHART} images: "
            f"needed {need_total}, found {len(pool)}"
        )

    # Deterministic shuffle of original indices
    rng = random.Random(SEED)
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)

    # Slice per split
    cursor = 0
    manifest = {"dataset": DATASET, "hf_split": HF_SPLIT, "seed": SEED, "splits": {}}

    for split_name, count in SPLIT_COUNTS.items():
        selected = idxs[cursor : cursor + count]
        cursor += count

        split_entries = []
        for k, sel in enumerate(selected):
            orig_idx, pil = pool[sel]
            fname = f"chart_{k:04d}.png"
            out_path = BASE_DIR / split_name / fname
            _save_pil(pil, out_path)

            split_entries.append(
                {
                    "filename": str(out_path.resolve()),
                    "orig_hf_index": orig_idx,
                    "label": LABEL_VALUE_FOR_CHART,
                }
            )

        manifest["splits"][split_name] = split_entries

    # Write manifest (so you can recreate the exact split)
    man_path = BASE_DIR.parent / "split_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Wrote images to {BASE_DIR} and manifest to {man_path}")


if __name__ == "__main__":
    main()
