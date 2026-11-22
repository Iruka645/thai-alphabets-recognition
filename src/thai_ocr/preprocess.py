"""Turn any image of a single handwritten letter into the 32x32 format the model was trained on.

Training images are already cleaned (dark ink on white, letter centred, 32x32), so they only
need `to_tensor`. Images from elsewhere (phone photos, drawings) go through `clean_image` first.
"""

import numpy as np
from PIL import Image, ImageOps

SIZE = 32


def clean_image(img: Image.Image, size: int = SIZE, margin: int = 2) -> Image.Image:
    """Grayscale, make ink dark on white, crop to the ink, pad to a square, resize to size x size."""
    g = ImageOps.grayscale(img)
    a = np.asarray(g, dtype=np.float32)
    # If the background is dark (e.g. white chalk / white-on-black drawing), flip it.
    if np.median(a) < 128:
        a = 255 - a
    # Ink = pixels clearly darker than the background.
    thresh = min(200.0, (np.median(a) + a.min()) / 2)
    ys, xs = np.where(a < thresh)
    if len(xs) == 0:
        raise ValueError("No ink found in image")
    a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = a.shape
    side = max(h, w)
    sq = np.full((side, side), 255.0, dtype=np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    inner = size - 2 * margin
    out = Image.fromarray(sq.astype(np.uint8)).resize((inner, inner), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    canvas.paste(out, (margin, margin))
    return canvas


def to_array(img: Image.Image) -> np.ndarray:
    """PIL 32x32 grayscale -> uint8 array (32, 32), dark ink on white."""
    g = ImageOps.grayscale(img)
    if g.size != (SIZE, SIZE):
        raise ValueError(f"Expected {SIZE}x{SIZE}, got {g.size[0]}x{g.size[1]}")
    return np.asarray(g, dtype=np.uint8)


def normalize(x):
    """uint8 images (white background) -> float in [0, 1] with ink = 1, background = 0.

    Works on numpy arrays or torch tensors. Ink-as-1 means padding during augmentation is 0 = background.
    """
    return 1.0 - x / 255.0
