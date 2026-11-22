"""Read one or more images and say which consonant each one is.

    python -m thai_ocr.predict runs/final-XXXX/model.pt my_letter.png other.jpg

Images can be any size; they are cleaned to the training format first (use --no-clean for
images that are already 32x32 dark-on-white like the dataset).
"""

import argparse

import numpy as np
import torch
from PIL import Image

from .model import LetterCNN
from .preprocess import clean_image, normalize, to_array


def load_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = LetterCNN(width=ckpt["width"], dropout=ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["thai"], ckpt["names"]


@torch.no_grad()
def classify(model, img: Image.Image, clean=True, top=3):
    if clean:
        img = clean_image(img)
    x = torch.from_numpy(normalize(to_array(img).astype(np.float32)))[None, None]
    p = torch.softmax(model(x.to(next(model.parameters()).device)), 1)[0].cpu()
    conf, idx = p.topk(top)
    return list(zip(idx.tolist(), conf.tolist()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    model, thai, names = load_model(args.model)
    for path in args.images:
        results = classify(model, Image.open(path), clean=not args.no_clean, top=args.top)
        guesses = "   ".join(f"{thai[i]} ({names[i]}) {c:.0%}" for i, c in results)
        print(f"{path}: {guesses}")


if __name__ == "__main__":
    main()
