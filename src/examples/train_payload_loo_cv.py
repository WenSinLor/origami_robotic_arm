"""
Leave-One-Out Cross-Validation (LOO-CV) for Payload Position Classifier
========================================================================
The only statistically unbiased accuracy estimate for small datasets.

How it works
------------
For N sample files and C classes:
  - Run N folds. In fold i, sample_i is the held-out test file for ALL classes.
  - Train on the remaining N-1 files, test on sample_i.
  - Each sample is tested exactly once, never seen during its own evaluation.
  - Final accuracy = total correct / total predictions  (N × C predictions total).

Why this is the right method
-----------------------------
Unlike the combinatorial sweep (train_payload_cv_sweep.py), LOO-CV produces:
  - One single unbiased accuracy number with a clean interpretation:
    "Given N-1 training samples, what is the expected accuracy on a new one?"
  - A per-sample breakdown showing which samples are consistently hard/easy.
  - No test-set leakage, no selection bias, no averaging over correlated splits.

Outputs
-------
    loo_confusion_matrix.png  — counts + normalised recall
    loo_per_sample.png        — per-fold accuracy bar chart (which sample was hard?)

Usage
-----
    python train_payload_loo_cv.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, classification_report


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state_20g"

COOR_DIRS = ["coor_0", "coor_1", "coor_2", "coor_3"]

# All sample files to pool from (must exist under every active COOR_DIR)
ALL_SAMPLE_FILES = [
    "trajectories_sample_0.h5",
    "trajectories_sample_1.h5",
    "trajectories_sample_2.h5",
    "trajectories_sample_3.h5",
    "trajectories_sample_4.h5",
    "trajectories_sample_5.h5",
    "trajectories_sample_6.h5",
    "trajectories_sample_7.h5",
]

T_START = 0.0           # time window start (seconds)
T_END   = 3.0           # time window end   (seconds)

CLASS_LABELS  = [1, 2, 3, 4]
CLASS_NAMES   = ["coor_0", "coor_1", "coor_2", "coor_3"]
CLASS_TARGETS = np.array([[ 1.,  0.],
                           [ 0.,  1.],
                           [-1.,  0.],
                           [ 0., -1.]], dtype=float)

ACTIVE_CLASSES  = [1, 2, 3, 4]   # subset of CLASS_LABELS to include
EXCLUDE_MARKERS = []              # 0-based marker indices to drop from features

OUTPUT_DIR = BASE_DIR


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_h5(path: Path, baseline_frames: int = 10):
    """Load HDF5 trajectory file → baseline-subtracted displacement (F, N, 2)."""
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]   # (F, N, 2)
        time = f["time_series/time"][:]              # (F,)

    # Forward-fill NaN values per marker per axis
    for n in range(pos.shape[1]):
        for ax in range(2):
            col = pos[:, n, ax]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                col[:] = 0.0
            elif nan_mask.any():
                idx = np.where(~nan_mask)[0]
                col[:idx[0]]  = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i+1] - idx[i] > 1:
                        col[idx[i]+1:idx[i+1]] = col[idx[i]]

    baseline = pos[:min(baseline_frames, pos.shape[0])].mean(axis=0, keepdims=True)
    return pos - baseline, time


def extract_features(disp, time, t_start, t_end, exclude_markers):
    """Slice time window, flatten to (T, N*2), drop excluded marker columns."""
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end,   side="right")) if t_end else len(time)
    X  = disp[i0:i1].reshape(i1 - i0, -1)   # (T, N*2)
    if exclude_markers:
        N    = X.shape[1] // 2
        keep = [c for n in range(N) if n not in exclude_markers
                for c in (2*n, 2*n+1)]
        X    = X[:, keep]
    return X


# ══════════════════════════════════════════════════════════════════════════════
#  OLS CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def nearest_class(p_xy, targets, labels):
    """Return the label whose 2-D target is closest to p_xy."""
    return int(labels[np.argmin(np.linalg.norm(targets - p_xy, axis=1))])


def fit_ols(base, coor_dirs, class_labels, class_targets, train_files):
    """
    Stack training data from all classes × train_files, z-score, fit OLS.
    Returns (Wout, mu, std) or None if no files found.
    """
    Xb, Yb = [], []
    for label, cdir, tgt in zip(class_labels, coor_dirs, class_targets):
        for fname in train_files:
            p = base / cdir / fname
            if not p.exists():
                continue
            disp, time = load_h5(p)
            X = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            Xb.append(X)
            Yb.append(np.tile(tgt, (len(X), 1)))

    if not Xb:
        return None

    X_all = np.vstack(Xb)
    mu    = X_all.mean(0)
    std   = X_all.std(0);  std[std < 1e-8] = 1.0
    Wout, _, _, _ = np.linalg.lstsq(add_bias((X_all - mu) / std),
                                     np.vstack(Yb), rcond=None)
    return Wout, mu, std


def evaluate_fold(base, coor_dirs, class_labels, class_targets,
                  test_file, Wout, mu, std):
    """
    Run inference on a single held-out file across all classes.
    Returns (y_true, y_pred) — one prediction per class.
    """
    y_true, y_pred = [], []
    for label, cdir in zip(class_labels, coor_dirs):
        p = base / cdir / test_file
        if not p.exists():
            return None, None
        disp, time = load_h5(p)
        X       = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
        mean_xy = (add_bias((X - mu) / std) @ Wout).mean(axis=0)
        y_true.append(label)
        y_pred.append(nearest_class(mean_xy, class_targets, class_labels))
    return np.array(y_true), np.array(y_pred)


# ══════════════════════════════════════════════════════════════════════════════
#  LOO-CV LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_loo(base, coor_dirs, class_labels, class_names, class_targets,
            valid_files):
    """
    Leave-one-out cross-validation.

    In each fold i, valid_files[i] is held out for ALL classes simultaneously.
    The model is trained on the remaining N-1 files.

    Returns
    -------
    y_true_all   : ndarray  shape (N × C,)   ground-truth labels (all folds)
    y_pred_all   : ndarray  shape (N × C,)   predicted labels    (all folds)
    fold_accs    : list[float]               per-fold accuracy (C predictions each)
    fold_labels  : list[str]                 held-out filename per fold (short)
    """
    N = len(valid_files)
    y_true_all, y_pred_all = [], []
    fold_accs, fold_labels = [], []

    print(f"  Running {N} LOO folds …")
    for i, held_out in enumerate(valid_files):
        train_files = [f for f in valid_files if f != held_out]

        result = fit_ols(base, coor_dirs, class_labels, class_targets,
                         train_files)
        if result is None:
            print(f"    [WARN] Fold {i+1}: no training data — skipped")
            continue
        Wout, mu, std = result

        y_true, y_pred = evaluate_fold(base, coor_dirs, class_labels,
                                       class_targets, held_out, Wout, mu, std)
        if y_true is None:
            print(f"    [WARN] Fold {i+1}: test file missing for ≥1 class — skipped")
            continue

        acc        = float(np.mean(y_true == y_pred))
        short_name = held_out.replace("trajectories_sample_", "S")
        fold_accs.append(acc)
        fold_labels.append(short_name)
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        # Per-class result for this fold
        detail = "  ".join(
            f"{n}:{'✓' if p == t else '✗'}"
            for t, p, n in zip(y_true, y_pred, class_names)
        )
        print(f"    Fold {i+1:>2}/{N}  held-out={short_name:<6}  "
              f"acc={acc*100:.0f}%  [{detail}]")

    return (np.array(y_true_all), np.array(y_pred_all),
            fold_accs, fold_labels)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES  —  Nature-journal style
# ══════════════════════════════════════════════════════════════════════════════

# Shared rcParams for publication-quality output
NATURE_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size":          7,
    "axes.titlesize":     8,
    "axes.labelsize":     7,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "legend.fontsize":    6.5,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

# Colorblind-safe palette (Wong 2011)
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]


def plot_confusion_matrix(cm_raw, class_names, loo_acc, out_path):
    """
    Side-by-side confusion matrices: raw counts (left) and row-normalised
    recall (right). Title shows the single unbiased LOO-CV accuracy.
    """
    C       = len(class_names)
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_raw / row_sum, 0.0)

    cmap_count  = LinearSegmentedColormap.from_list(
        "nat_blue",  ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    cmap_recall = LinearSegmentedColormap.from_list(
        "nat_green", ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"])

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4),
                                 gridspec_kw={"wspace": 0.65})

        panels = [
            (axes[0], cm_raw.astype(float), cmap_count,  None, ".0f", "Count"),
            (axes[1], cm_norm,              cmap_recall,  1.0, ".2f", "Recall"),
        ]

        for ax, data, cmap, vmax, fmt, cbar_label in panels:
            vm = vmax if vmax else max(data.max(), 1e-6)
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vm,
                           interpolation="nearest", aspect="equal")

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.80)
            cb.set_label(cbar_label, labelpad=4, fontsize=6)
            cb.ax.tick_params(labelsize=5.5, width=0.5, length=2, pad=2)
            cb.outline.set_linewidth(0.4)

            ax.set_xticks(range(C))
            ax.set_xticklabels(class_names, rotation=30, ha="right")
            ax.set_yticks(range(C))
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")

            thresh = vm * 0.55
            for i in range(C):
                for j in range(C):
                    v = data[i, j]
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=5.5, fontweight="bold",
                            color="white" if v > thresh else "#333333")
            for k in range(C + 1):
                ax.axhline(k - 0.5, color="white", lw=0.4)
                ax.axvline(k - 0.5, color="white", lw=0.4)
            ax.set_xlim(-0.5, C - 0.5)
            ax.set_ylim(C - 0.5, -0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

        axes[0].set_title("Count")
        axes[1].set_title("Recall")
        fig.suptitle(
            f"LOO-CV confusion matrix  |  "
            f"overall accuracy = {loo_acc*100:.1f}%  "
            f"({int(round(loo_acc * cm_raw.sum()))} / {int(cm_raw.sum())} correct)",
            fontsize=7.5, fontweight="bold", y=1.03)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_per_sample(fold_accs, fold_labels, class_names, loo_acc, out_path):
    """
    Horizontal bar chart showing per-fold accuracy.
    Each bar = one held-out sample (all C classes tested on it).
    Reference line at overall LOO-CV accuracy and at chance level.
    Bars coloured green/red by whether the fold was above/below chance.

    This plot answers: "which samples are consistently hard to generalise to?"
    """
    N      = len(fold_accs)
    chance = 1.0 / len(class_names)
    colors = [PALETTE[2] if a > chance else PALETTE[5] for a in fold_accs]

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 0.55 * N + 0.8))

        ys = np.arange(N)
        ax.barh(ys, [a * 100 for a in fold_accs], height=0.55,
                color=colors, alpha=0.82, linewidth=0, zorder=2)

        # Accuracy labels on bars
        for y, a in zip(ys, fold_accs):
            ax.text(a * 100 + 0.8, y, f"{a*100:.0f}%",
                    va="center", fontsize=6, color="#333333")

        # Reference lines
        ax.axvline(loo_acc * 100, lw=1.2, ls="-", color="#333333",
                   zorder=3, label=f"LOO-CV acc = {loo_acc*100:.1f}%")
        ax.axvline(chance * 100, lw=0.8, ls="--", color="#AAAAAA",
                   zorder=1, label=f"Chance = {chance*100:.0f}%")

        ax.set_yticks(ys)
        ax.set_yticklabels(fold_labels)
        ax.set_xlabel("Accuracy on held-out sample (%)")
        ax.set_xlim(0, 115)
        ax.set_ylim(-0.5, N - 0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(25))
        ax.set_title("Per-fold accuracy  (held-out sample → test)",
                     fontweight="bold")
        ax.legend(fontsize=6, loc="lower right", frameon=True,
                  framealpha=0.92, edgecolor="#DDDDDD")

        # Colour legend
        import matplotlib.patches as mpatches
        ax.legend(handles=[
            mpatches.Patch(color=PALETTE[2], alpha=0.82, label="Above chance"),
            mpatches.Patch(color=PALETTE[5], alpha=0.82, label="At/below chance"),
            plt.Line2D([0],[0], color="#333333", lw=1.2,
                       label=f"LOO-CV acc = {loo_acc*100:.1f}%"),
            plt.Line2D([0],[0], color="#AAAAAA", lw=0.8, ls="--",
                       label=f"Chance = {chance*100:.0f}%"),
        ], fontsize=6, loc="lower right", frameon=True,
           framealpha=0.92, edgecolor="#DDDDDD")

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    base    = Path(BASE_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Filter to active classes
    idx           = [i for i, l in enumerate(CLASS_LABELS) if l in ACTIVE_CLASSES]
    coor_dirs     = [COOR_DIRS[i]    for i in idx]
    class_labels  = [CLASS_LABELS[i] for i in idx]
    class_names   = [CLASS_NAMES[i]  for i in idx]
    class_targets = CLASS_TARGETS[idx]

    # 2. Keep only files that exist for every active class
    valid_files = [f for f in ALL_SAMPLE_FILES
                   if all((base / d / f).exists() for d in coor_dirs)]
    skipped = set(ALL_SAMPLE_FILES) - set(valid_files)
    if skipped:
        print(f"  [WARN] Missing for ≥1 class, skipped: {sorted(skipped)}")

    if len(valid_files) < 2:
        print(f"\n  [ERROR] Need at least 2 valid files for LOO-CV "
              f"(found {len(valid_files)}).")
        print(f"  Checked: {base / coor_dirs[0] / ALL_SAMPLE_FILES[0]}")
        return

    print(f"\n{'═'*60}")
    print(f"  LOO-CV — Payload Classifier")
    print(f"{'═'*60}")
    print(f"  Classes     : {class_names}")
    print(f"  Valid files : {len(valid_files)}  →  {len(valid_files)} folds")
    print(f"  Train size  : {len(valid_files) - 1} files/class per fold")
    print(f"  Test size   : 1 file/class per fold  ({len(class_names)} predictions/fold)")
    print(f"  Time window : {T_START}–{T_END} s\n")

    # 3. Run LOO-CV
    y_true, y_pred, fold_accs, fold_labels = run_loo(
        base, coor_dirs, class_labels, class_names, class_targets, valid_files)

    # 4. Compute single unbiased accuracy
    loo_acc = float(np.mean(y_true == y_pred))
    n_correct = int((y_true == y_pred).sum())
    n_total   = len(y_true)

    print(f"\n{'═'*60}")
    print(f"  LOO-CV Results")
    print(f"{'═'*60}")
    print(f"  Overall accuracy : {loo_acc*100:.1f}%  ({n_correct}/{n_total})")
    print(f"  Chance level     : {100/len(class_names):.0f}%")
    print(f"\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                digits=3, zero_division=0))

    # 5. Save figures
    print()
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    plot_confusion_matrix(cm, class_names, loo_acc,
                          out_dir / "loo_confusion_matrix.png")

    plot_per_sample(fold_accs, fold_labels, class_names, loo_acc,
                    out_dir / "loo_per_sample.png")

    print(f"\n  Done.  Outputs → {out_dir.resolve()}")
    print(f"    loo_confusion_matrix.png")
    print(f"    loo_per_sample.png")


if __name__ == "__main__":
    main()