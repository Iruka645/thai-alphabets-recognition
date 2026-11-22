"""Collect every writer's images into one file: data/processed/dataset.npz.

Put each person's data in data/raw/ as either a .zip or a folder (one per person). Inside, the
images must be named like 01_ko_kai.png ... 44_ho_nokhuk.png (sub-folders are fine). The zip or
folder name becomes that writer's ID, which is what keeps one person's letters out of both
training and testing at the same time.

    python -m thai_ocr.prepare
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import NUM_CLASSES, THAI, label_from_filename
from .preprocess import to_array

IMAGE_RE = re.compile(r"(^|/)\d{2}_[a-z_]+\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)


def _iter_images(source: Path):
    """Yield (filename, PIL image) for every letter image in a zip or folder."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if IMAGE_RE.search(name) and "__MACOSX" not in name:
                    yield name, Image.open(io.BytesIO(zf.read(name)))
    else:
        for p in sorted(source.rglob("*")):
            if p.is_file() and IMAGE_RE.search(p.name):
                yield p.name, Image.open(p)


def _writer_id(source: Path) -> str:
    return source.stem if source.suffix.lower() == ".zip" else source.name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw", type=Path)
    ap.add_argument("--out", default="data/processed/dataset.npz", type=Path)
    args = ap.parse_args()

    sources = sorted(p for p in args.raw.iterdir()
                     if not p.name.startswith(".") and (p.is_dir() or p.suffix.lower() == ".zip"))
    if not sources:
        raise SystemExit(f"No zips or folders found in {args.raw}/")

    X, y, w, writers = [], [], [], []
    problems = 0
    for source in sources:
        wid = _writer_id(source)
        seen = {}
        for name, img in _iter_images(source):
            try:
                label = label_from_filename(name)
                arr = to_array(img)
            except ValueError as e:
                print(f"  [skip] {wid}: {e}")
                problems += 1
                continue
            if label in seen:
                print(f"  [skip] {wid}: duplicate letter {THAI[label]} ({name} and {seen[label]})")
                problems += 1
                continue
            seen[label] = name
            X.append(arr)
            y.append(label)
            w.append(len(writers))
        missing = [THAI[i] for i in range(NUM_CLASSES) if i not in seen]
        status = "ok" if not missing else f"missing {' '.join(missing)}"
        print(f"{wid}: {len(seen)}/{NUM_CLASSES} letters ({status})")
        if seen:
            writers.append(wid)
        problems += bool(missing)

    X = np.stack(X)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=np.array(y, dtype=np.int64),
                        writer=np.array(w, dtype=np.int64), writers=np.array(writers))
    print(f"\nSaved {len(X)} images from {len(writers)} writers to {args.out}"
          + (f"  ({problems} warnings above)" if problems else ""))


if __name__ == "__main__":
    main()
