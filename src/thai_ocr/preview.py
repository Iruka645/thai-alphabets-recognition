"""Save a picture of what augmentation does, to check it still looks like real handwriting.

    python -m thai_ocr.preview            # -> runs/augment_preview.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import augment
from .train import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.npz")
    ap.add_argument("--out", default="runs/augment_preview.png")
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--letters", type=int, default=12, help="rows")
    ap.add_argument("--copies", type=int, default=10, help="augmented copies per row")
    args = ap.parse_args()

    X, *_ = load(args.data)
    torch.manual_seed(0)
    idx = torch.randperm(len(X))[: args.letters]
    rows = []
    for i in idx:
        x = X[i: i + 1]
        batch = torch.cat([x, augment(x.repeat(args.copies, 1, 1, 1), args.aug)])
        rows.append(torch.cat(list(batch[:, 0]), dim=1))
    grid = (1 - torch.cat(rows, dim=0)).numpy() * 255  # back to dark ink on white
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(grid.astype(np.uint8))
    img.resize((img.width * 3, img.height * 3), Image.NEAREST).save(args.out)
    print(f"Saved {args.out}  (first column = original, rest = augmented)")


if __name__ == "__main__":
    main()
