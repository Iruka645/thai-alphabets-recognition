"""A small CNN for 32x32 letters, plus batch augmentation that runs on the GPU."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import NUM_CLASSES


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class LetterCNN(nn.Module):
    """32x32 -> 16 -> 8 -> 4, then a small classifier. About 0.8M parameters."""

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 32, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(_block(1, width), _block(width, width * 2), _block(width * 2, width * 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def augment(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Make each image look like it came from a slightly different writer.

    x: (B, 1, 32, 32) with ink = 1, background = 0. Never mirrors (a mirrored letter can be a
    different letter) and keeps the whole letter in frame most of the time, so tails such as the
    ones under ญ ฎ ฏ ฐ survive.
    """
    if strength <= 0:
        return x
    B, _, H, W = x.shape
    dev = x.device
    s = strength

    def u(lo, hi):
        return torch.empty(B, device=dev).uniform_(lo, hi)

    # Random rotation, per-axis scale, shear and shift.
    ang = u(-12, 12) * s * math.pi / 180
    sx = 1 + u(-0.15, 0.10) * s
    sy = 1 + u(-0.15, 0.10) * s
    sh = u(-0.2, 0.2) * s
    tx, ty = u(-0.1, 0.1) * s, u(-0.1, 0.1) * s
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([cos / sx, (-sin + sh) / sx, tx], 1),
        torch.stack([sin / sy, cos / sy, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)

    # Elastic wobble: a smooth random displacement, like a shaky or different hand.
    noise = torch.randn(B, 2, 4, 4, device=dev) * 0.03 * s
    grid = grid + F.interpolate(noise, size=(H, W), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    # Pen thickness: a third get thicker strokes, a third thinner, a third unchanged.
    r = torch.rand(B, 1, 1, 1, device=dev)
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    mix = torch.rand(B, 1, 1, 1, device=dev) * 0.5 * s  # partial, so it is not too extreme at 32px
    x = torch.where(r < 1 / 3, x + mix * (thick - x), torch.where(r < 2 / 3, x + mix * (thin - x), x))

    # Contrast / ink darkness and a little noise.
    x = x * (1 + u(-0.3, 0.1).view(B, 1, 1, 1) * s)
    x = x + torch.randn_like(x) * 0.03 * s
    return x.clamp(0, 1)
