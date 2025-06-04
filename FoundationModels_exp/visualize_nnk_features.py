#!/usr/bin/env python3
"""
visualize_nnk_features.py — class‑aware histograms for NNK output
================================================================

Files expected (per split)  ✦ features  ✦ labels  ✦ correctness mask
------------------------------------------------------------------
NNK_features_<DATASET>_<MODEL>_<split>.npy      (float  [N,D])
Y_labels_<DATASET>_<MODEL>_<split>.npy          (int    [N] )
CORRECT_mask_<DATASET>_<MODEL>_<split>.npy      (int 0/1 [N] )   # test only

For backward compatibility the script falls back to the original filenames
without the <split> suffix, and to label arrays embedded elsewhere — but for
full functionality, save the explicit *Y_labels* files when you dump features.

Outputs
-------
1. **train_byClass** — one figure, 27 dims × *C* classes coloured uniquely.
2. **class<c>** — for every class *c*: 27‑dim grid showing
      • train samples (class *c*)
      • test‑correct samples  (class *c*)
      • test‑incorrect samples (class *c*)
Colours auto‑extend beyond 10 classes via Matplotlib’s *tab20* palette.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.style.use("ggplot")

# ───────────────────────── I/O helpers ───────────────────────────────────────

def _load(split: str, dataset: str, model: str):
    """Return features X, labels y, (optional) correctness mask."""
    feat_f = Path(f"NNK_features_{dataset}_{model}_{split}.npy")
    if not feat_f.exists():
        feat_f = Path(f"NNK_features_{dataset}_{model}.npy")  # legacy
    if not feat_f.exists():
        return None, None, None
    X = np.load(feat_f)

    y_f = Path(f"Y_labels_{dataset}_{model}_{split}.npy")
    if not y_f.exists():
        y = np.zeros(len(X), dtype=int)  # fallback dummy
        print(f"[WARN] {y_f} missing — labels set to 0")
    else:
        y = np.load(y_f).astype(int)

    mask_f = Path(f"CORRECT_mask_{dataset}_{model}_{split}.npy")
    mask = np.load(mask_f).astype(int) if mask_f.exists() else None
    return X, y, mask

# ───────────────────────── colour utils ──────────────────────────────────────

def colour_map(n_cls):
    """Return n_cls distinct colours (cycled from tab20)."""
    base = cm.get_cmap("tab20", 20).colors
    repeats = int(np.ceil(n_cls / 20))
    palette = np.vstack([base]*repeats)[:n_cls]
    return palette

# ───────────────────────── plotting functions ───────────────────────────────

def plot_train_by_class(Xtr, ytr, title, outdir, bins=40, cols_per_row=6):
    D = Xtr.shape[1]; classes = np.unique(ytr); C = len(classes)
    colours = colour_map(C)

    rows = int(np.ceil(D / cols_per_row))
    fig, axes = plt.subplots(rows, cols_per_row,
                             figsize=(cols_per_row*3, rows*2.4), squeeze=False)
    axes = axes.flatten()
    for d in range(D):
        ax = axes[d]; ax.set_title(f"dim {d}", fontsize=7)
        for c, col in zip(classes, colours):
            ax.hist(Xtr[ytr==c, d], bins=bins, histtype="step", color=col,
                    label=f"c{c}" if d==0 else None)
        if d == 0: ax.legend(fontsize=6, ncol=max(1, C//8))
    for j in range(D, len(axes)): axes[j].axis("off")
    fig.suptitle(f"{title} — train hist by class", fontsize=12)
    fig.tight_layout(rect=[0,0.03,1,0.95])
    p = outdir / f"{title}_train_byClass.png"; fig.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] {p}")


def plot_per_class_grids(Xtr, ytr, Xte, yte, mask_te, title, outdir,
                         bins=40, cols_per_row=6):
    D = Xtr.shape[1]
    for c in np.unique(ytr):
        idx_tr = (ytr == c)
        idx_te = (yte == c)
        idx_ok  = idx_te & (mask_te == 1) if mask_te is not None else np.zeros_like(idx_te, bool)
        idx_bad = idx_te & (mask_te == 0) if mask_te is not None else np.zeros_like(idx_te, bool)

        rows = int(np.ceil(D / cols_per_row))
        fig, axes = plt.subplots(rows, cols_per_row,
                                 figsize=(cols_per_row*3, rows*2.4), squeeze=False)
        axes = axes.flatten()
        for d in range(D):
            ax = axes[d]
            ax.hist(Xtr[idx_tr,d],  bins=bins, alpha=0.5, label="train", color="C0")
            if mask_te is not None:
                ax.hist(Xte[idx_ok,d],  bins=bins, alpha=0.5, label="test OK",  color="C2")
                ax.hist(Xte[idx_bad,d], bins=bins, alpha=0.5, label="test bad", color="C3")
            ax.set_title(f"dim {d}", fontsize=6)
            if d==0:
                ax.legend(fontsize=6)
        for j in range(D, len(axes)): axes[j].axis("off")
        fig.suptitle(f"{title} — class {c}", fontsize=12)
        fig.tight_layout(rect=[0,0.03,1,0.95])
        p = outdir / f"{title}_class{c}.png"; fig.savefig(p, dpi=150); plt.close()
        print(f"[PLOT] {p}")

# ───────────────────────── CLI main ─────────────────────────────────────────

def main():
    ag = argparse.ArgumentParser()
    ag.add_argument("--dataset", nargs="+", default=["cifar10"])
    ag.add_argument("--model",   nargs="+", default=["CLIP-ViT"])
    ag.add_argument("--out", default="nnk_vis")
    args = ag.parse_args()
    outdir = Path(args.out); outdir.mkdir(exist_ok=True)

    for d in args.dataset:
        for m in args.model:
            Xtr, ytr, _        = _load("train", d, m)
            Xte, yte, mask_te  = _load("test",  d, m)
            if Xtr is None or Xte is None:
                print(f"[SKIP] missing files for {d}-{m}"); continue
            if ytr is None or yte is None:
                print(f"[SKIP] label arrays missing for {d}-{m}"); continue
            title = f"{d}_{m}"
            plot_train_by_class(Xtr, ytr, title, outdir)
            plot_per_class_grids(Xtr, ytr, Xte, yte, mask_te, title, outdir)

if __name__ == "__main__":
    main()
