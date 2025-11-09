"""Check YOLO dataset integrity and report statistics."""

import re
from pathlib import Path

CLASSES = {0: "symbol_title", 1: "last_price_pill"}
ROOT = Path("datasets/tradingview")  # adjust if needed
IMG_EXTS = (".png", ".jpg", ".jpeg")
ID_PREFIX_RE = re.compile(r"^([^-]+)-(.*)$")  # matches "<id>-<basename>"


def image_stems(split: str) -> set[str]:
    imgd = ROOT / f"images/{split}"
    stems = set()
    for p in imgd.iterdir():
        if p.suffix.lower() in IMG_EXTS:
            stems.add(p.stem)
    return stems


def label_map_by_basestem(split: str) -> dict[str, Path]:
    """Map canonical base stem -> label path, stripping any '<id>-' prefix."""
    lbld = ROOT / f"labels/{split}"
    mapping: dict[str, Path] = {}
    for p in lbld.glob("*.txt"):
        stem = p.stem
        m = ID_PREFIX_RE.match(stem)
        base_stem = m.group(2) if m else stem  # strip id- if present
        # If duplicates exist for same base, prefer the newest mtime
        prev = mapping.get(base_stem)
        if prev is None or p.stat().st_mtime_ns > prev.stat().st_mtime_ns:
            mapping[base_stem] = p
    return mapping


def validate_label_file(p: Path) -> tuple[int, int, list[tuple[int, str]]]:
    """Return (num_boxes, num_bad_ids, bad_boxes_list)."""
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    num_boxes = 0
    bad_ids = 0
    bad_boxes: list[tuple[int, str]] = []
    for i, ln in enumerate(lines, 1):
        parts = ln.split()
        try:
            cid = int(parts[0])
            if cid not in CLASSES:
                bad_ids += 1
            x, y, w, h = map(float, parts[1:5])
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_boxes.append((i, f"{x},{y},{w},{h}"))
            num_boxes += 1
        except Exception:
            bad_boxes.append((i, "parse-error"))
    return num_boxes, bad_ids, bad_boxes


def check_split(split: str) -> None:
    imgs = image_stems(split)
    lbl_map = label_map_by_basestem(split)

    missing_label_files = []
    empty_label_files = []
    bad_class_ids = []
    bad_box_values = []
    per_class_counts = {0: 0, 1: 0}
    images_with_any_box = 0

    for stem in sorted(imgs):
        lblp = lbl_map.get(stem)
        if not lblp:
            missing_label_files.append(stem)
            continue

        lines = [ln.strip() for ln in lblp.read_text().splitlines() if ln.strip()]
        if not lines:
            empty_label_files.append(stem)
            continue

        # validate and count
        has_box = False
        for i, ln in enumerate(lines, 1):
            parts = ln.split()
            try:
                cid = int(parts[0])
                if cid in CLASSES:
                    per_class_counts[cid] += 1
                else:
                    bad_class_ids.append((lblp.name, i, cid))
                x, y, w, h = map(float, parts[1:5])
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    bad_box_values.append((lblp.name, i, (x, y, w, h)))
                has_box = True
            except Exception:
                bad_box_values.append((lblp.name, i, "parse-error"))
        if has_box:
            images_with_any_box += 1

    total_imgs = len(imgs)
    print(f"\n== {split.upper()} ==")
    print(f"Images: {total_imgs}")
    print(f"Labeled images (≥1 box): {images_with_any_box}")
    print(f"Missing label files: {len(missing_label_files)}")
    print(f"Empty label files: {len(empty_label_files)}")
    print(
        "Per-class box counts: "
        + ", ".join(f"{CLASSES[k]}={v}" for k, v in per_class_counts.items())
    )

    # Debug a few samples if something looks off
    if missing_label_files[:5]:
        print("Examples missing labels (first 5):", missing_label_files[:5])
    if bad_class_ids[:5]:
        print("Bad class IDs (first 5):", bad_class_ids[:5])
    if bad_box_values[:5]:
        print("Bad/invalid bbox lines (first 5):", bad_box_values[:5])

    if per_class_counts[1] == 0:
        print("Warning: no 'last_price_pill' boxes found in this split.")
    if per_class_counts[0] == 0:
        print("Warning: no 'symbol_title' boxes found in this split.")


def main() -> None:
    for s in ["train", "val", "test"]:
        check_split(s)
    print("\nDataset check completed.")


if __name__ == "__main__":
    main()
