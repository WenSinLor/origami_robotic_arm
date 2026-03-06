"""
Cross-Validation Sweep for Payload Position Classifier
=======================================================
Exhaustively evaluates the 2D OLS classifier over every combination of
N_TRAIN_PER_CLASS training samples drawn from ALL_SAMPLE_FILES, with the
remaining samples used for testing.

Pipeline
--------
1. Enumerate all C(n_files, N_TRAIN_PER_CLASS) train/test splits.
2. For each split: fit OLS → compute test accuracy + per-class accuracy.
3. Flag outlier splits via the IQR rule.
4. Produce two publication-quality figures:
      cv_bar_errorbar.png     — mean ± SD accuracy (overall + per class)
      cv_confusion_matrix.png — mean confusion matrix over clean splits

Usage
-----
    python train_payload_cv_sweep.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import itertools
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state_long"

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

N_TRAIN_PER_CLASS = 7   # files per class used for training in each split

T_START = 0.0           # time window start (seconds)
T_END   = 5.0           # time window end   (seconds)

CLASS_LABELS  = [1, 2, 3, 4]
CLASS_NAMES   = ["coor_0", "coor_1", "coor_2", "coor_3"]
CLASS_TARGETS = np.array([[ 1.,  0.],
                           [ 0.,  1.],
                           [-1.,  0.],
                           [ 0., -1.]], dtype=float)

ACTIVE_CLASSES  = [1, 2, 3]   # subset of CLASS_LABELS to include
EXCLUDE_MARKERS = []              # 0-based marker indices to drop from features

# Outlier detection: flag splits outside Q1/Q3 ± IQR_MULTIPLIER × IQR
IQR_MULTIPLIER = 1.5

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
        keep = [c for n in range(N) if n not in exclude_markers for c in (2*n, 2*n+1)]
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
    Returns (Wout, mu, std) or None if no files are found.
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
    mu, std = X_all.mean(0), X_all.std(0)
    std[std < 1e-8] = 1.0
    Wout, _, _, _ = np.linalg.lstsq(add_bias((X_all - mu) / std),
                                     np.vstack(Yb), rcond=None)
    return Wout, mu, std


def evaluate(base, coor_dirs, class_labels, class_targets,
             test_files, Wout, mu, std):
    """Run inference on test_files. Returns (y_true, y_pred) arrays."""
    y_true, y_pred = [], []
    for label, cdir in zip(class_labels, coor_dirs):
        for fname in test_files:
            p = base / cdir / fname
            if not p.exists():
                continue
            disp, time = load_h5(p)
            X       = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            mean_xy = (add_bias((X - mu) / std) @ Wout).mean(axis=0)
            y_true.append(label)
            y_pred.append(nearest_class(mean_xy, class_targets, class_labels))
    return np.array(y_true), np.array(y_pred)


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(base, coor_dirs, class_labels, class_names, class_targets, valid_files):
    """
    Iterate over all C(n_files, N_TRAIN_PER_CLASS) splits.

    Returns
    -------
    accuracies     : list[float]            overall test accuracy per split
    per_class_accs : list[dict[str,float]]  per-class accuracy per split
    cm_list        : list[ndarray]          confusion matrix per split
    """
    combos = list(itertools.combinations(valid_files, N_TRAIN_PER_CLASS))
    accuracies, per_class_accs, cm_list = [], [], []

    print(f"  Running {len(combos)} splits …")
    for i, train_files in enumerate(combos):
        test_files = [f for f in valid_files if f not in train_files]

        result = fit_ols(base, coor_dirs, class_labels, class_targets,
                         list(train_files))
        if result is None:
            continue
        Wout, mu, std = result

        y_true, y_pred = evaluate(base, coor_dirs, class_labels, class_targets,
                                  test_files, Wout, mu, std)

        acc = float(np.mean(y_true == y_pred))
        pca = {name: float(np.mean(y_pred[y_true == lab] == lab))
               for lab, name in zip(class_labels, class_names)
               if (y_true == lab).any()}

        accuracies.append(acc)
        per_class_accs.append(pca)
        cm_list.append(confusion_matrix(y_true, y_pred, labels=class_labels))

        tag = "Tr:" + "|".join(f.replace("trajectories_sample_", "S")
                                for f in train_files)
        print(f"    [{i+1:>3}/{len(combos)}]  {tag:<25}  acc = {acc*100:.1f}%")

    return accuracies, per_class_accs, cm_list


# ══════════════════════════════════════════════════════════════════════════════
#  OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_outliers(accuracies):
    """Return boolean mask: True where split accuracy is outside IQR fence."""
    a      = np.array(accuracies)
    q1, q3 = np.percentile(a, [25, 75])
    iqr    = q3 - q1
    return (a < q1 - IQR_MULTIPLIER * iqr) | (a > q3 + IQR_MULTIPLIER * iqr)


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
PALETTE      = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                "#56B4E9", "#D55E00", "#F0E442", "#000000"]
OUTLIER_COL  = "#D55E00"


def plot_bar_errorbar(accuracies, per_class_accs, outlier_mask,
                      class_names, out_path):
    """
    Bar + error bar plot of cross-validation accuracy.

    Bar height  = mean over clean (non-outlier) splits
    Error bar   = ±1 SD
    Dots        = individual clean split values (jittered)
    Red crosses = outlier splits (excluded from bar statistics)
    """
    categories = ["Overall"] + list(class_names)
    n_cats     = len(categories)
    clean      = ~outlier_mask

    # Collect per-category value arrays
    vals_clean, vals_out = [], []
    vals_clean.append(np.array(accuracies)[clean] * 100)
    vals_out.append(np.array(accuracies)[outlier_mask] * 100)
    for cname in class_names:
        vc = np.array([r.get(cname, np.nan) for r in per_class_accs])
        vals_clean.append(vc[clean][~np.isnan(vc[clean])] * 100)
        vals_out.append(vc[outlier_mask][~np.isnan(vc[outlier_mask])] * 100)

    means = [v.mean() for v in vals_clean]
    stds  = [v.std()  for v in vals_clean]

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(1.5 + n_cats * 1.1, 3.2))
        xs = np.arange(n_cats)

        # Bars with error bars
        ax.bar(xs, means, yerr=stds, width=0.52,
               color=[PALETTE[i % len(PALETTE)] for i in range(n_cats)],
               alpha=0.78, linewidth=0, zorder=2,
               error_kw=dict(elinewidth=0.9, capsize=3.5, capthick=0.9,
                             ecolor="#222222", alpha=0.85))

        # Individual clean split dots (jittered)
        rng = np.random.default_rng(seed=42)
        for i, vc in enumerate(vals_clean):
            jx = xs[i] + rng.uniform(-0.15, 0.15, size=len(vc))
            ax.scatter(jx, vc, s=7, color=PALETTE[i % len(PALETTE)],
                       alpha=0.55, linewidths=0, zorder=3)

        # Outlier crosses
        for i, vo in enumerate(vals_out):
            if len(vo):
                jx = xs[i] + rng.uniform(-0.10, 0.10, size=len(vo))
                ax.scatter(jx, vo, s=20, color=OUTLIER_COL, marker="x",
                           linewidths=1.2, zorder=4)

        # Mean ± SD annotation above each bar (two lines)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(xs[i], m + s + 1.8, f"{m:.1f}%\n±{s:.1f}%",
                    ha="center", va="bottom", fontsize=5.8,
                    color=PALETTE[i % len(PALETTE)], fontweight="bold",
                    linespacing=1.4)

        # Subtle reference lines
        ax.axhline(100, lw=0.5, ls=":", color="#BBBBBB", zorder=0)
        chance = round(100 / len(class_names))
        ax.axhline(chance, lw=0.5, ls="--", color="#BBBBBB", zorder=0)
        ax.text(n_cats - 0.5, chance + 0.8, "Chance",
                ha="right", va="bottom", fontsize=5.5, color="#AAAAAA")

        ax.set_xticks(xs)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Test accuracy (%)")
        ax.set_ylim(0, max(means) + max(stds) + 26)
        ax.set_xlim(-0.55, n_cats - 0.45)
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
        ax.tick_params(axis="y", which="minor", length=1.5, width=0.4)

        n_clean = int(clean.sum())
        n_out   = int(outlier_mask.sum())
        ax.legend(handles=[
            mpatches.Patch(facecolor="#888888", alpha=0.78,
                           label=f"Mean ± SD  (n = {n_clean} splits)"),
            plt.Line2D([0],[0], marker="o", color="#888888", ls="None",
                       markersize=3.5, alpha=0.6, label="Individual split"),
            plt.Line2D([0],[0], marker="x", color=OUTLIER_COL, ls="None",
                       markersize=5, markeredgewidth=1.2,
                       label=f"Outlier  (n = {n_out})"),
        ], loc="lower right", frameon=True, framealpha=0.92,
           edgecolor="#DDDDDD", handlelength=1.2)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_confusion_matrix(cm_sum, n_splits, class_names, mean_acc, std_acc, out_path):
    """
    Side-by-side mean confusion matrices over clean splits.

    Left  : mean raw counts
    Right : row-normalised recall (0–1)
    """
    cm_mean = cm_sum / max(n_splits, 1)
    C       = len(class_names)
    row_sum = cm_mean.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_mean / row_sum, 0.0)

    cmap_count  = LinearSegmentedColormap.from_list(
        "nat_blue",  ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
    cmap_recall = LinearSegmentedColormap.from_list(
        "nat_green", ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"])

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4),
                                 gridspec_kw={"wspace": 0.65})

        panels = [
            (axes[0], cm_mean, cmap_count,  None, ".1f", "Mean count"),
            (axes[1], cm_norm, cmap_recall, 1.0,  ".2f", "Recall"),
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

            # Cell text + grid lines
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

        axes[0].set_title("Mean count")
        axes[1].set_title("Mean recall")
        fig.suptitle(
            f"Confusion matrix  ({n_splits} clean splits)  |  "
            f"mean test acc = {mean_acc*100:.1f}% ± {std_acc*100:.1f}%",
            fontsize=7.5, fontweight="bold", y=1.03)

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

    n_splits = len(list(itertools.combinations(valid_files, N_TRAIN_PER_CLASS)))
    print(f"\n{'═'*58}")
    print(f"  Cross-validation sweep")
    print(f"{'═'*58}")
    print(f"  Classes        : {class_names}")
    print(f"  Valid files    : {len(valid_files)}")
    print(f"  Train / split  : {N_TRAIN_PER_CLASS}  →  {n_splits} total splits")
    print(f"  Time window    : {T_START}–{T_END} s\n")

    # 3. Run CV
    accuracies, per_class_accs, cm_list = run_cv(
        base, coor_dirs, class_labels, class_names, class_targets, valid_files)

    # 4. Outlier detection
    outlier_mask = detect_outliers(accuracies)
    clean_mask   = ~outlier_mask
    accs         = np.array(accuracies)

    print(f"\n{'═'*58}")
    print(f"  Summary")
    print(f"{'═'*58}")
    print(f"  Mean acc  (all)   : {accs.mean()*100:.1f}%")
    print(f"  Mean acc  (clean) : {accs[clean_mask].mean()*100:.1f}%"
          f" ± {accs[clean_mask].std()*100:.1f}%")
    print(f"  Outliers flagged  : {outlier_mask.sum()} / {len(accs)}")
    print(f"\n  Per-class accuracy (clean splits):")
    for cname in class_names:
        v = [r[cname] for r, o in zip(per_class_accs, outlier_mask)
             if not o and cname in r]
        if v:
            print(f"    {cname:10s}: {np.mean(v)*100:.1f}% ± {np.std(v)*100:.1f}%")

    # 5. Aggregate confusion matrices from clean splits
    cm_clean = [cm for cm, o in zip(cm_list, outlier_mask) if not o]
    cm_sum   = (np.sum(cm_clean, axis=0).astype(float) if cm_clean
                else np.zeros((len(class_names),) * 2))

    # 6. Save figures
    print()
    plot_bar_errorbar(
        accuracies, per_class_accs, outlier_mask, class_names,
        out_dir / "cv_bar_errorbar.png")

    plot_confusion_matrix(
        cm_sum, len(cm_clean), class_names,
        accs[clean_mask].mean(), accs[clean_mask].std(),
        out_dir / "cv_confusion_matrix.png")

    print(f"\n  Done.  Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()