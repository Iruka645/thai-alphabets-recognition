"""Train the letter model.

Two modes:

  python -m thai_ocr.train cv      # "How good is it on a stranger?"  (do this first, and after every change)
  python -m thai_ocr.train final   # Train on everyone and save the model you actually use.

`cv` splits the WRITERS (not the images) into groups. Each round trains on all groups but one and
tests on the people in the left-out group, so every person gets tested by a model that never saw
their handwriting. With ~20 people this is far more reliable than one fixed 3-person test set.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .labels import NAMES, THAI
from .model import LetterCNN, augment
from .preprocess import normalize
from .report import save_confusion_png, text_report


def load(path):
    d = np.load(path, allow_pickle=False)
    X = torch.from_numpy(normalize(d["X"].astype(np.float32))).unsqueeze(1)
    return X, torch.from_numpy(d["y"]), d["writer"], list(d["writers"])


def fit(X, y, args, device, log_prefix=""):
    torch.manual_seed(args.seed)
    model = LetterCNN(width=args.width, dropout=args.dropout).to(device)
    X, y = X.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(X) / args.batch_size)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X), device=device)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(X), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = augment(X[idx], args.aug), y[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yb, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(idx)
        if (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs:
            print(f"{log_prefix}epoch {epoch + 1:4d}/{args.epochs}  loss {loss_sum / total:.3f}  "
                  f"train acc (augmented) {correct / total:.1%}")
    return model


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    return torch.cat([model(X[i:i + 512].to(device)).argmax(1).cpu() for i in range(0, len(X), 512)]).numpy()


def writer_folds(n_writers, k, seed):
    order = np.random.default_rng(seed).permutation(n_writers)
    return [order[i::k] for i in range(k)]


def run_cv(X, y, writer, writers, args, device, out: Path):
    k = min(args.folds, len(writers))
    if k < 2:
        raise SystemExit("Cross-validation needs at least 2 writers.")
    preds = np.full(len(y), -1)
    for f, test_w in enumerate(writer_folds(len(writers), k, args.seed)):
        test = np.isin(writer, test_w)
        names = ", ".join(writers[w] for w in test_w)
        print(f"\n=== Fold {f + 1}/{k}: testing on {names} ===")
        model = fit(X[~test], y[~test], args, device, log_prefix=f"[fold {f + 1}] ")
        preds[test] = predict(model, X[test], device)
        print(f"[fold {f + 1}] accuracy on unseen writers: {np.mean(preds[test] == y[test].numpy()):.1%}")

    y_np = y.numpy()
    report = text_report(y_np, preds, writer, writers)
    print("\n" + report)
    (out / "report.txt").write_text(report + "\n", encoding="utf-8")
    save_confusion_png(y_np, preds, out / "confusion.png")
    np.savez(out / "predictions.npz", y_true=y_np, y_pred=preds, writer=writer)
    print(f"\nSaved report.txt, confusion.png and predictions.npz to {out}/")


def run_final(X, y, writers, args, device, out: Path):
    print(f"Training on all {len(writers)} writers ({len(X)} images)")
    model = fit(X, y, args, device)
    torch.save({"state_dict": model.state_dict(), "width": args.width, "dropout": args.dropout,
                "thai": THAI, "names": NAMES}, out / "model.pt")
    print(f"Saved {out / 'model.pt'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cv", "final"])
    ap.add_argument("--data", default="data/processed/dataset.npz")
    ap.add_argument("--out", default=None, help="output folder (default runs/<mode>-<time>)")
    ap.add_argument("--folds", type=int, default=5, help="cv: number of writer groups")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--width", type=int, default=32, help="channels in the first layer (model size)")
    ap.add_argument("--aug", type=float, default=1.0, help="augmentation strength, 0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out = Path(args.out or f"runs/{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    X, y, writer, writers = load(args.data)
    print(f"{len(X)} images, {len(writers)} writers, device: {device}")
    if args.mode == "cv":
        run_cv(X, y, writer, writers, args, device, out)
    else:
        run_final(X, y, writers, args, device, out)


if __name__ == "__main__":
    main()
