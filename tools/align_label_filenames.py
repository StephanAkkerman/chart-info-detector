# tools/align_label_filenames.py
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("datasets/tradingview")
IMG_EXTS = (".png", ".jpg", ".jpeg")
ID_PREFIX = re.compile(r"^([^-]+)-(.*)$")  # "<id>-<stem>"


def find_image_for_stem(split: str, stem: str) -> bool:
    imgd = ROOT / f"images/{split}"
    return any((imgd / f"{stem}{ext}").exists() for ext in IMG_EXTS)


def align_split(split: str) -> None:
    lbld = ROOT / f"labels/{split}"
    if not lbld.exists():
        return
    grouped: dict[str, list[Path]] = {}
    for p in lbld.glob("*.txt"):
        m = ID_PREFIX.match(p.stem)
        base = m.group(2) if m else p.stem
        grouped.setdefault(base, []).append(p)

    kept = renamed = deleted = missing_img = 0
    for base, paths in grouped.items():
        if not find_image_for_stem(split, base):
            missing_img += len(paths)
            continue
        paths.sort(key=lambda x: x.stat().st_mtime_ns)  # newest last
        keep = paths[-1]
        for old in paths[:-1]:
            old.unlink()
            deleted += 1
        kept += 1
        canonical = lbld / f"{base}.txt"
        if keep != canonical:
            if canonical.exists():
                canonical.unlink()
            keep.rename(canonical)
            renamed += 1
    print(
        f"{split}: kept {kept}, renamed {renamed}, deleted {deleted}, labels without image {missing_img}"
    )


def main():
    for s in ("train", "val", "test"):
        align_split(s)


if __name__ == "__main__":
    main()
