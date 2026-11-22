"""Turn predictions into something readable: overall score, per-letter score, worst mix-ups, confusion chart."""

from pathlib import Path

import numpy as np

from .labels import NAMES, NUM_CLASSES, THAI


def confusion(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def text_report(y_true, y_pred, writer=None, writers=None, top_k=15) -> str:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion(y_true, y_pred)
    lines = [f"Overall accuracy on unseen writers: {np.mean(y_true == y_pred):.1%}  ({len(y_true)} images)", ""]

    if writer is not None:
        lines.append("Per writer (each scored by a model that never saw their handwriting):")
        for w in np.unique(writer):
            m = writer == w
            name = writers[w] if writers is not None else str(w)
            lines.append(f"  {name:<40s} {np.mean(y_true[m] == y_pred[m]):6.1%}")
        lines.append("")

    support = cm.sum(1)
    acc = np.divide(np.diag(cm), support, out=np.zeros(NUM_CLASSES), where=support > 0)
    lines.append("Hardest letters:")
    for i in np.argsort(acc)[:10]:
        lines.append(f"  {THAI[i]} {NAMES[i]:<14s} {acc[i]:6.1%}")
    lines.append("")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0]
    lines.append("Most common mix-ups (true -> predicted, count):")
    for t, p in pairs[:top_k]:
        if off[t, p] == 0:
            break
        lines.append(f"  {THAI[t]} -> {THAI[p]}   {off[t, p]}   ({NAMES[t]} -> {NAMES[p]})")
    return "\n".join(lines)


def _thai_font():
    from matplotlib import font_manager
    wanted = ["Leelawadee UI", "Tahoma", "Noto Sans Thai", "Noto Sans Thai Looped", "Garuda", "Loma", "Sarabun"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in wanted if f in have), None)


def save_confusion_png(y_true, y_pred, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion(np.asarray(y_true), np.asarray(y_pred)).astype(float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    font = _thai_font()
    labels = THAI if font else [f"{i + 1:02d}" for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES), labels, fontsize=9, fontname=font)
    ax.set_yticks(range(NUM_CLASSES), labels, fontsize=9, fontname=font)
    ax.set_xlabel("Predicted letter")
    ax.set_ylabel("True letter")
    ax.set_title("Which letter got read as which (rows sum to 1)"
                 + ("" if font else "\nNo Thai font found: numbers are the 01-44 file prefixes"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
