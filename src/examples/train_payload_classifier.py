"""
Mixed-Source Payload Classifier — 2D OLS Regression (Per-Class Sample Control)
===============================================================================
Freely assign any samples from any directory to the training and test sets,
with FULL INDEPENDENT CONTROL per coordinate class.

Each class in TRAIN_SAMPLES and TEST_SAMPLES has its own list of
(dir_key, filename) pairs — completely independent across classes.

Pipeline
--------
1. Load HDF5 trajectories per class from specified directories.
2. Stack all (class × train_sample) blocks → z-score → fit OLS.
3. Evaluate on train samples (sanity check) and test samples.
4. Produce diagnostic figures.

Usage
-----
    python per_class_classifier.py
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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit this block only
# ══════════════════════════════════════════════════════════════════════════════

# ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"
ROOT_DIR = "/home/wensin/Documents/origami_robotic_arm/data"

# Directory aliases
DIRS = {
    "base" : f"{ROOT_DIR}/soft_state_20g",
    "near" : f"{ROOT_DIR}/soft_state_100g_near",
}

# ── Train samples: fully independent per class ────────────────────────────────
# Format: { "coor_name": [("dir_key", "filename"), ...], ... }
TRAIN_SAMPLES = {
    "coor_0": [
        ("base", "trajectories_sample_0.h5"),
        ("base", "trajectories_sample_1.h5"),
        ("base", "trajectories_sample_2.h5"),
        ("base", "trajectories_sample_3.h5"),
        ("base", "trajectories_sample_4.h5"),
        ("base", "trajectories_sample_5.h5"),
        ("base", "trajectories_sample_6.h5"),
        ("base", "trajectories_sample_7.h5"),
        # ("near", "trajectories_sample_8.h5"),
        # ("near", "trajectories_sample_9.h5"),
        # ("near", "trajectories_sample_10.h5"),
        # ("near", "trajectories_sample_11.h5"),
        # ("near", "trajectories_sample_12.h5"),
        # ("near", "trajectories_sample_13.h5"),

    ],
    "coor_1": [
        ("base", "trajectories_sample_0.h5"),
        ("base", "trajectories_sample_1.h5"),
        ("base", "trajectories_sample_2.h5"),
        ("base", "trajectories_sample_3.h5"),
        ("base", "trajectories_sample_4.h5"),
        ("base", "trajectories_sample_5.h5"),
        ("base", "trajectories_sample_6.h5"),
        ("base", "trajectories_sample_7.h5"),
        # ("base", "trajectories_sample_8.h5"),
        # ("base", "trajectories_sample_9.h5"),
        # ("base", "trajectories_sample_10.h5"),
        # ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
    ],
    "coor_2": [
        ("base", "trajectories_sample_0.h5"),
        ("base", "trajectories_sample_1.h5"),
        ("base", "trajectories_sample_2.h5"),
        ("base", "trajectories_sample_3.h5"),
        ("base", "trajectories_sample_4.h5"),
        ("base", "trajectories_sample_5.h5"),
        ("base", "trajectories_sample_6.h5"),
        ("base", "trajectories_sample_7.h5"),
        # ("base", "trajectories_sample_8.h5"),
        # ("base", "trajectories_sample_9.h5"),
        # ("base", "trajectories_sample_10.h5"),
        # ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
    ],
    "coor_3": [
        ("base", "trajectories_sample_0.h5"),
        ("base", "trajectories_sample_1.h5"),
        ("base", "trajectories_sample_2.h5"),
        ("base", "trajectories_sample_3.h5"),
        ("base", "trajectories_sample_4.h5"),
        ("base", "trajectories_sample_5.h5"),
        ("base", "trajectories_sample_6.h5"),
        ("base", "trajectories_sample_7.h5"),
        # ("base", "trajectories_sample_8.h5"),
        # ("base", "trajectories_sample_9.h5"),
        # ("base", "trajectories_sample_10.h5"),
        # ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
    ],
}

# ── Test samples: fully independent per class ─────────────────────────────────
TEST_SAMPLES = {
    "coor_0": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
        # ("near", "trajectories_sample_14.h5"),
        # ("near", "trajectories_sample_15.h5"),
        # ("near", "trajectories_sample_16.h5"),
        # ("near", "trajectories_sample_17.h5"),
        # ("near", "trajectories_sample_18.h5"),
        # ("near", "trajectories_sample_19.h5"),
    ],
    "coor_1": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
        # ("near", "trajectories_sample_14.h5"),
        # ("near", "trajectories_sample_15.h5"),
        # ("near", "trajectories_sample_16.h5"),
        # ("near", "trajectories_sample_17.h5"),
        # ("near", "trajectories_sample_18.h5"),
        # ("near", "trajectories_sample_19.h5"),
    ],
    "coor_2": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
        # ("near", "trajectories_sample_14.h5"),
        # ("near", "trajectories_sample_15.h5"),
        # ("near", "trajectories_sample_16.h5"),
        # ("near", "trajectories_sample_17.h5"),
        # ("near", "trajectories_sample_18.h5"),
        # ("near", "trajectories_sample_19.h5"),
    ],
    "coor_3": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
        # ("base", "trajectories_sample_12.h5"),
        # ("base", "trajectories_sample_13.h5"),
        # ("near", "trajectories_sample_14.h5"),
        # ("near", "trajectories_sample_15.h5"),
        # ("near", "trajectories_sample_16.h5"),
        # ("near", "trajectories_sample_17.h5"),
        # ("near", "trajectories_sample_18.h5"),
        # ("near", "trajectories_sample_19.h5"),
    ],
}

# ── Everything else ───────────────────────────────────────────────────────────
COOR_DIRS   = ["coor_0", "coor_1", "coor_2", "coor_3"]

T_START = 0.0
T_END   = 3.0

CLASS_LABELS  = [1, 2, 3, 4]
CLASS_NAMES   = ["coor_0", "coor_1", "coor_2", "coor_3"]
CLASS_TARGETS = np.array([[ 1.,  0.],
                           [ 0.,  1.],
                           [-1.,  0.],
                           [ 0., -1.]], dtype=float)

ACTIVE_CLASSES  = [1, 2, 3, 4]
EXCLUDE_MARKERS = []

OUTPUT_DIR = f"{ROOT_DIR}/soft_state_20g"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_label(dir_key, fname):
    return f"{dir_key}/{fname.replace('trajectories_','').replace('.h5','')}"


def resolve(dir_key, coor_dir, fname):
    return Path(DIRS[dir_key]) / coor_dir / fname


def summarise_samples(samples_dict):
    """Print a neat per-class summary of sample sources."""
    for cname, slist in samples_dict.items():
        n_base = sum(1 for dk, _ in slist if dk == "base")
        n_near = sum(1 for dk, _ in slist if dk == "near")
        parts  = []
        if n_base: parts.append(f"{n_base}×base")
        if n_near: parts.append(f"{n_near}×near")
        print(f"    {cname:8s}: {' + '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_h5(path: Path, baseline_frames: int = 1):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]

    coord_dim = pos.shape[2]

    for n in range(pos.shape[1]):
        for ax in range(coord_dim):
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
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end, side="right")) if t_end is not None else len(time)

    X = disp[i0:i1].reshape(i1 - i0, -1)
    ts = time[i0:i1] - time[i0]

    if exclude_markers:
        coord_dim = disp.shape[2]
        n_markers = disp.shape[1]
        keep = [coord_dim*n + d
                for n in range(n_markers)
                if n not in exclude_markers
                for d in range(coord_dim)]
        X = X[:, keep]

    return X, ts


# ══════════════════════════════════════════════════════════════════════════════
#  OLS CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def nearest_class(p_xy, targets, labels):
    return int(labels[np.argmin(np.linalg.norm(targets - p_xy, axis=1))])


def fit_ols(coor_dirs, class_labels, class_names, class_targets,
            train_samples_dict, pca_variance=0.95):
    """
    Stack training data per class using its own sample list.
    train_samples_dict: { "coor_name": [(dir_key, fname), ...] }
    """
    Xb, Yb = [], []
    N = D = None
    ts_ref = None

    print("=" * 60)
    print("  Building training matrix (per-class samples)")
    print("=" * 60)

    for label, cdir, cname, tgt in zip(class_labels, coor_dirs,
                                        class_names, class_targets):
        sample_list = train_samples_dict.get(cname, [])
        if not sample_list:
            print(f"  [WARN] No training samples defined for {cname} — skipped")
            continue

        for dir_key, fname in sample_list:
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} — skipped")
                continue
            disp, time = load_h5(p)
            X, ts = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            if D is None:
                coord_dim = disp.shape[2]
                D, N, ts_ref = X.shape[1], X.shape[1] // coord_dim, ts
            Xb.append(X)
            Yb.append(np.tile(tgt, (len(X), 1)))
            print(f"  + [{dir_key}] {cdir}/{fname}  label={label}  T={len(X)}")

    if not Xb:
        raise RuntimeError("No training data found. Check TRAIN_SAMPLES and DIRS.")

    X_all = np.vstack(Xb)
    mu    = X_all.mean(axis=0)
    std   = X_all.std(axis=0); std[std < 1e-8] = 1.0
    Xn    = (X_all - mu) / std

    pca_full = PCA().fit(Xn)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    k        = min(int(np.searchsorted(cumvar, pca_variance)) + 1,
                   Xn.shape[1], max(1, Xn.shape[0] - 1))
    pca      = PCA(n_components=k).fit(Xn)

    Wout, _, _, _ = np.linalg.lstsq(add_bias(Xn), np.vstack(Yb), rcond=None)

    print(f"\n  Features  : {N} markers × 2 = {D}")
    print(f"  OLS shape : {add_bias(Xn).shape} → {np.vstack(Yb).shape}")
    print(f"  PCA       : {k} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
    return Wout, pca, mu, std, N, D, ts_ref


def run_inference(coor_dirs, class_labels, class_names, class_targets,
                  samples_dict, Wout, mu, std, label=""):
    """
    Run OLS inference using per-class sample lists.
    samples_dict: { "coor_name": [(dir_key, fname), ...] }
    """
    results = []
    y_true, y_pred = [], []

    print(f"\n{'=' * 60}")
    print(f"  Inference — {label}")
    print(f"{'=' * 60}")

    for lab, cdir, cname, tgt in zip(class_labels, coor_dirs,
                                      class_names, class_targets):
        sample_list = samples_dict.get(cname, [])
        if not sample_list:
            print(f"  [WARN] No samples defined for {cname} — skipped")
            continue

        for dir_key, fname in sample_list:
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} — skipped")
                continue
            disp, time = load_h5(p)
            X, ts   = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            Xn      = (X - mu) / std
            Y_hat   = add_bias(Xn) @ Wout
            mean_xy = Y_hat.mean(axis=0)
            pred    = nearest_class(mean_xy, class_targets, class_labels)
            tick    = "✓" if pred == lab else "✗"
            slabel  = sample_label(dir_key, fname)
            print(f"  {tick}  {cdir}/{slabel}  true={lab}  "
                  f"pred={pred}  mean=({mean_xy[0]:+.3f}, {mean_xy[1]:+.3f})")
            results.append(dict(label=lab, name=cname,
                                sample=slabel, dir_key=dir_key,
                                ts=ts, Y_hat=Y_hat, mean_xy=mean_xy,
                                pred_class=pred, target_xy=np.array(tgt, float)))
            y_true.append(lab)
            y_pred.append(pred)

    return results, np.array(y_true), np.array(y_pred)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

VT_MAROON = "#861F41"
VT_ORANGE = "#E5751F"
VT_STONE = "#75787B"
VT_DARK_STONE = "#54585A"
VT_LIGHT_STONE = "#D7D2CB"
VT_PALE_MAROON = "#F2E8ED"
VT_PALE_ORANGE = "#FBE9DC"
VT_GOLD = "#B3A369"
VT_TEAL = "#508590"

PLOT_FONT_SIZES = {
    "base": 9,
    "axis_label": 9,
    "tick": 8,
    "legend": 7.5,
    "title": 10,
    "panel_title": 9,
    "suptitle": 10,
    "annotation": 7,
    "matrix_cell": 8,
    "class_label": 8,
    "colorbar_label": 8,
    "colorbar_tick": 7,
}

NATURE_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          PLOT_FONT_SIZES["base"],
    "axes.titlesize":     PLOT_FONT_SIZES["panel_title"],
    "axes.labelsize":     PLOT_FONT_SIZES["axis_label"],
    "xtick.labelsize":    PLOT_FONT_SIZES["tick"],
    "ytick.labelsize":    PLOT_FONT_SIZES["tick"],
    "legend.fontsize":    PLOT_FONT_SIZES["legend"],
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

PALETTE = [VT_MAROON, VT_ORANGE, VT_STONE, VT_TEAL,
           VT_GOLD, "#C64600", VT_DARK_STONE, "#2C2A29"]

DIR_MARKERS = {"base": "o", "near": "s"}
DIR_BG      = {"base": VT_PALE_MAROON, "near": VT_PALE_ORANGE}


def plot_staircase(results, class_labels, class_names,
                   test_samples_dict, train_acc, test_acc, out_path):
    COL_TARGET = VT_DARK_STONE
    COL_X      = VT_MAROON
    COL_Y      = VT_ORANGE
    COL_OK     = VT_STONE
    COL_BAD    = "#C64600"

    # Build ordered list: class order → then sample order within class
    ordered = []
    for lab, cname in zip(class_labels, class_names):
        sample_list = test_samples_dict.get(cname, [])
        for dir_key, fname in sample_list:
            sl = sample_label(dir_key, fname)
            for r in results:
                if r["label"] == lab and r["sample"] == sl and r["dir_key"] == dir_key:
                    ordered.append(r)
                    break

    if not ordered:
        print("[WARN] No results to plot staircase.")
        return

    with plt.rc_context(NATURE_RC):
        fig, (axx, axy) = plt.subplots(2, 1, figsize=(11, 3.8), sharex=True,
                                        gridspec_kw={"hspace": 0.08})

        t_offset, sample_mids, first = 0.0, [], True
        for r in ordered:
            ts  = r["ts"]
            dt  = float(ts[1] - ts[0]) if len(ts) > 1 else 1/30
            ta  = ts + t_offset
            tx, ty = float(r["target_xy"][0]), float(r["target_xy"][1])
            mx, my = float(r["mean_xy"][0]),   float(r["mean_xy"][1])
            is_correct = r["pred_class"] == r["label"]
            status_col = COL_OK if is_correct else COL_BAD
            mean_ls = "-" if is_correct else ":"
            sx = float(r["Y_hat"][:,0].std())
            sy = float(r["Y_hat"][:,1].std())

            bg_col = DIR_BG.get(r["dir_key"], "#F5F5F5")
            for ax in (axx, axy):
                ax.axvspan(ta[0], ta[-1], color=bg_col, alpha=0.5, zorder=0)

            for ax, sig, tgt, mean_v, sd_v, col_sig, band_col, ylabel in [
                (axx, r["Y_hat"][:,0], tx, mx, sx, COL_X,
                 VT_PALE_MAROON, "x readout"),
                (axy, r["Y_hat"][:,1], ty, my, sy, COL_Y,
                 VT_PALE_ORANGE, "y readout"),
            ]:
                ax.hlines(tgt, ta[0], ta[-1], colors=COL_TARGET, lw=1.8,
                          ls="--", alpha=0.7,
                          label="Target" if first and ax is axx else "")
                ax.plot(ta, sig, color=col_sig, lw=0.8, alpha=0.8,
                        label=("x(t)" if ax is axx else "y(t)") if first else "")
                ax.fill_between(ta, mean_v - sd_v, mean_v + sd_v,
                                color=band_col, alpha=0.45, linewidth=0,
                                label="±1 SD" if first and ax is axx else "")
                ax.hlines(mean_v, ta[0], ta[-1], colors=col_sig, lw=1.5,
                          ls=mean_ls, alpha=0.95,
                          label="Mean" if first and ax is axx else "")
                ax.text(ta[len(ta)//2], mean_v + sd_v + 0.06,
                        f"{mean_v:+.2f}\n±{sd_v:.2f}",
                        ha="center", va="bottom",
                        fontsize=PLOT_FONT_SIZES["annotation"],
                        color=col_sig, fontweight="bold", linespacing=1.3,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white", edgecolor=status_col,
                                  linewidth=0.7, alpha=0.9))
                ax.set_ylabel(ylabel, fontsize=PLOT_FONT_SIZES["axis_label"])

            sample_mids.append(ta[len(ta)//2])
            t_offset += float(ts[-1]) + dt
            first = False

        # Sample separators
        t_cur = 0.0
        for i, r in enumerate(ordered):
            ts  = r["ts"]
            dur = float(ts[-1]) + (float(ts[1]-ts[0]) if len(ts) > 1 else 1/30)
            if i > 0:
                for ax in (axx, axy):
                    ax.axvline(t_cur, color=VT_LIGHT_STONE, lw=0.7, ls=":",
                               zorder=1)
            t_cur += dur

        for ax in (axx, axy):
            ax.set_ylim(-2.1, 2.1)
            ax.set_yticks([-1, 0, 1])
            ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
            ax.spines["bottom"].set_linewidth(0.7)

        axy.set_xlabel("Time (s)", fontsize=PLOT_FONT_SIZES["axis_label"])

        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        extra_handles = [
            Patch(facecolor=DIR_BG["base"], edgecolor=VT_STONE, label="base"),
            Patch(facecolor=DIR_BG["near"], edgecolor=VT_STONE, label="near"),
            Line2D([0], [0], color=COL_Y, lw=0.9, label="y(t)"),
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="white", markeredgecolor=COL_OK,
                   markersize=5, label="correct annotation"),
            Line2D([0], [0], marker="s", color="none",
                   markerfacecolor="white", markeredgecolor=COL_BAD,
                   markersize=5, label="wrong annotation"),
        ]
        base_handles, _ = axx.get_legend_handles_labels()
        axx.legend(loc="upper left", frameon=True, framealpha=0.9,
                   edgecolor=VT_LIGHT_STONE, ncol=5,
                   fontsize=PLOT_FONT_SIZES["legend"],
                   handles=base_handles + extra_handles)

        # Class labels on top axis — use midpoint of each class's block
        class_mids = []
        idx = 0
        for cname in class_names:
            n = len(test_samples_dict.get(cname, []))
            if n > 0:
                class_mids.append(np.mean(sample_mids[idx:idx+n]))
            idx += n

        ax_top = axx.secondary_xaxis("top")
        ax_top.set_xticks(class_mids)
        ax_top.set_xticklabels(class_names,
                               fontsize=PLOT_FONT_SIZES["class_label"],
                               fontweight="bold", color=VT_MAROON)
        ax_top.tick_params(length=0)

        fig.suptitle(
            f"Per-class mixed-source 2D readout — "
            f"train acc = {train_acc*100:.0f}%   test acc = {test_acc*100:.0f}%",
            fontsize=PLOT_FONT_SIZES["suptitle"], fontweight="bold",
            color=VT_MAROON, y=1.01)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_polar(results, class_labels, class_names, class_targets,
               train_acc, test_acc, out_path):
    COLORS = {lab: PALETTE[i % len(PALETTE)]
              for i, lab in enumerate(class_labels)}

    with plt.rc_context(NATURE_RC):
        fig = plt.figure(figsize=(3.0, 3.0))
        ax  = fig.add_subplot(111, projection="polar")

        for lab, name, tgt in zip(class_labels, class_names, class_targets):
            theta = np.arctan2(float(tgt[1]), float(tgt[0]))
            r     = np.hypot(float(tgt[0]), float(tgt[1]))
            ax.scatter(theta, r, marker="*", s=160,
                       color=COLORS[lab], edgecolor=VT_DARK_STONE,
                       linewidths=0.6, zorder=5)
            ax.text(theta, r + 0.10, name, ha="center", va="center",
                    fontsize=PLOT_FONT_SIZES["class_label"],
                    fontweight="bold", color=COLORS[lab])

        for r in results:
            x, y   = r["mean_xy"]
            theta  = np.arctan2(y, x)
            rr     = np.hypot(x, y)
            col    = COLORS[r["label"]]
            marker = DIR_MARKERS.get(r["dir_key"], "o")
            if r["pred_class"] == r["label"]:
                ax.scatter(theta, rr, s=20, marker=marker, color=col,
                           alpha=0.75, linewidths=0, zorder=4)
            else:
                ax.scatter(theta, rr, s=28, marker="x", color=col,
                           linewidths=1.0, alpha=0.9, zorder=6)

        ax.set_rlim(0, 1.15)
        ax.set_title(
            f"Polar readout (○=base, □=near, ×=wrong)\n"
            f"train acc={train_acc*100:.0f}%  test acc={test_acc*100:.0f}%",
            fontsize=PLOT_FONT_SIZES["title"], pad=10,
            color=VT_MAROON, fontweight="bold")

        legend_handles = (
            [plt.Line2D([0],[0], marker="o", color=PALETTE[i % len(PALETTE)],
                        ls="None", markersize=4, label=n)
             for i, (_, n) in enumerate(zip(class_labels, class_names))] +
            [plt.Line2D([0],[0], marker="o", color=VT_DARK_STONE, ls="None",
                        markersize=4, label="base correct"),
             plt.Line2D([0],[0], marker="s", color=VT_DARK_STONE, ls="None",
                        markersize=4, label="near correct"),
             plt.Line2D([0],[0], marker="x", color=VT_DARK_STONE, ls="None",
                        markersize=5, markeredgewidth=1.0, label="Wrong")]
        )
        ax.tick_params(labelsize=PLOT_FONT_SIZES["tick"])
        ax.legend(handles=legend_handles, fontsize=PLOT_FONT_SIZES["legend"],
                  loc="upper left", frameon=True, framealpha=0.9,
                  edgecolor=VT_LIGHT_STONE)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_confusion_matrix(cm_raw, class_names, train_acc, test_acc, out_path):
    C       = len(class_names)
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_raw / row_sum, 0.0)

    cmap_count  = LinearSegmentedColormap.from_list(
        "vt_maroon", ["#FFFFFF", VT_PALE_MAROON, "#C89AAE", VT_MAROON])
    cmap_recall = LinearSegmentedColormap.from_list(
        "vt_orange", ["#FFFFFF", VT_PALE_ORANGE, "#F0A66E", VT_ORANGE])

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
            cb.set_label(cbar_label, labelpad=4,
                         fontsize=PLOT_FONT_SIZES["colorbar_label"])
            cb.ax.tick_params(labelsize=PLOT_FONT_SIZES["colorbar_tick"],
                              width=0.6, length=2.2, pad=2)
            cb.outline.set_linewidth(0.4)
            ax.set_xticks(range(C))
            ax.set_xticklabels(class_names, rotation=30, ha="right",
                               fontsize=PLOT_FONT_SIZES["tick"])
            ax.set_yticks(range(C))
            ax.set_yticklabels(class_names, fontsize=PLOT_FONT_SIZES["tick"])
            ax.set_xlabel("Predicted class",
                          fontsize=PLOT_FONT_SIZES["axis_label"])
            ax.set_ylabel("True class", fontsize=PLOT_FONT_SIZES["axis_label"])
            thresh = vm * 0.55
            for i in range(C):
                for j in range(C):
                    v = data[i, j]
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=PLOT_FONT_SIZES["matrix_cell"],
                            fontweight="bold",
                            color="white" if v > thresh else VT_DARK_STONE)
            for k in range(C + 1):
                ax.axhline(k - 0.5, color="white", lw=0.5)
                ax.axvline(k - 0.5, color="white", lw=0.5)
            ax.set_xlim(-0.5, C - 0.5); ax.set_ylim(C - 0.5, -0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        axes[0].set_title("Count", fontsize=PLOT_FONT_SIZES["panel_title"])
        axes[1].set_title("Recall", fontsize=PLOT_FONT_SIZES["panel_title"])
        fig.suptitle(
            f"Confusion matrix  (train acc = {train_acc*100:.0f}%,  "
            f"test acc = {test_acc*100:.0f}%)",
            fontsize=PLOT_FONT_SIZES["suptitle"], fontweight="bold",
            color=VT_MAROON, y=1.04)
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_pca(pca, mu, std, coor_dirs, class_labels, class_names,
             train_samples_dict, test_samples_dict, N, out_path):
    COLORS = [PALETTE[i % len(PALETTE)] for i in range(len(class_labels))]
    PCX, PCY = 0, 1

    proj_train = {cdir: [] for cdir in coor_dirs}
    proj_test  = {cdir: [] for cdir in coor_dirs}

    for cdir, cname in zip(coor_dirs, class_names):
        for dir_key, fname in train_samples_dict.get(cname, []):
            p = resolve(dir_key, cdir, fname)
            if not p.exists(): continue
            disp, time = load_h5(p)
            X, _ = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            proj_train[cdir].append(pca.transform((X - mu) / std))

        for dir_key, fname in test_samples_dict.get(cname, []):
            p = resolve(dir_key, cdir, fname)
            if not p.exists(): continue
            disp, time = load_h5(p)
            X, _ = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            proj_test[cdir].append(pca.transform((X - mu) / std))

    for cdir in coor_dirs:
        proj_train[cdir] = np.vstack(proj_train[cdir]) if proj_train[cdir] else np.empty((0, pca.n_components_))
        proj_test[cdir]  = np.vstack(proj_test[cdir])  if proj_test[cdir]  else np.empty((0, pca.n_components_))

    k      = pca.n_components_
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k_95   = int(np.searchsorted(cumvar, 0.95)) + 1

    with plt.rc_context(NATURE_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2),
                                        gridspec_kw={"wspace": 0.42})
        xs = np.arange(1, k + 1)
        ax1.bar(xs, pca.explained_variance_ratio_ * 100,
                color=VT_PALE_ORANGE, edgecolor="white", lw=0.5,
                label="Individual")
        ax1.plot(xs, cumvar * 100, color=PALETTE[0], lw=1.5,
                 marker="o", ms=3, label="Cumulative")
        ax1.axhline(95, color=VT_STONE, lw=1.0, ls="--")
        ax1.axvline(k_95, color=VT_ORANGE, lw=1.2, ls="--",
                    label=f"95% @ PC{k_95}")
        ax1.set_xlabel("Principal component",
                       fontsize=PLOT_FONT_SIZES["axis_label"])
        ax1.set_ylabel("Explained variance (%)",
                       fontsize=PLOT_FONT_SIZES["axis_label"])
        ax1.set_title("Scree plot", fontsize=PLOT_FONT_SIZES["panel_title"],
                      fontweight="bold", color=VT_MAROON)
        ax1.set_xlim(0.5, min(k + 0.5, 20.5)); ax1.set_ylim(0, 110)
        ax1.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        ax1.legend(fontsize=PLOT_FONT_SIZES["legend"])

        for (cdir, cname), col in zip(zip(coor_dirs, class_names), COLORS):
            Xp = proj_train[cdir]
            Xt = proj_test[cdir]
            if len(Xp) > 0:
                pcy = Xp[:, PCY] if k > PCY else np.zeros(len(Xp))
                ax2.scatter(Xp[:, PCX], pcy, c=col, s=4, alpha=0.7,
                            label=f"{cname} train", linewidths=0)
            if len(Xt) > 0:
                pcyt = Xt[:, PCY] if k > PCY else np.zeros(len(Xt))
                ax2.scatter(Xt[:, PCX], pcyt, facecolors="none",
                            edgecolors=col, s=5, alpha=0.35, lw=0.6,
                            label=f"{cname} test")
            if len(Xp) > 0 or len(Xt) > 0:
                pool = np.vstack([arr for arr in (Xp, Xt) if len(arr) > 0])
                ax2.text(pool[:, PCX].mean(),
                         pool[:, PCY].mean() if k > PCY else 0,
                         cname, fontsize=PLOT_FONT_SIZES["class_label"],
                         color=col, fontweight="bold",
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                   alpha=0.85, ec=col, lw=0.6))

        ax2.set_xlabel(f"PC{PCX+1}  ({pca.explained_variance_ratio_[PCX]*100:.1f}%)",
                       fontsize=PLOT_FONT_SIZES["axis_label"])
        ax2.set_ylabel(f"PC{PCY+1}  ({pca.explained_variance_ratio_[PCY]*100:.1f}%)"
                       if k > PCY else f"PC{PCY+1} (n/a)",
                       fontsize=PLOT_FONT_SIZES["axis_label"])
        ax2.set_title(f"PC{PCX+1} vs PC{PCY+1}  (filled=train, hollow=test)",
                      fontsize=PLOT_FONT_SIZES["panel_title"],
                      fontweight="bold", color=VT_MAROON)
        ax2.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        ax2.legend(fontsize=PLOT_FONT_SIZES["legend"], markerscale=2,
                   loc="best")
        fig.suptitle(
            f"PCA diagnostic  |  {k} components  |  {N} markers × 2 = {N*2} features",
            fontsize=PLOT_FONT_SIZES["suptitle"], fontweight="bold",
            color=VT_MAROON, y=1.03)
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx           = [i for i, l in enumerate(CLASS_LABELS) if l in ACTIVE_CLASSES]
    coor_dirs     = [COOR_DIRS[i]    for i in idx]
    class_labels  = [CLASS_LABELS[i] for i in idx]
    class_names   = [CLASS_NAMES[i]  for i in idx]
    class_targets = CLASS_TARGETS[idx]

    print(f"\n{'═'*60}")
    print(f"  Per-Class Mixed-Source Payload Classifier — 2D OLS")
    print(f"{'═'*60}")
    print(f"  Train samples per class:")
    summarise_samples({k: v for k, v in TRAIN_SAMPLES.items() if k in class_names})
    print(f"  Test samples per class:")
    summarise_samples({k: v for k, v in TEST_SAMPLES.items()  if k in class_names})
    print(f"  Time window : {T_START}–{T_END} s\n")

    # 1. Fit OLS
    Wout, pca, mu, std, N, D, _ = fit_ols(
        coor_dirs, class_labels, class_names, class_targets, TRAIN_SAMPLES)

    # 2. Sanity check on train set
    _, tr_true, tr_pred = run_inference(
        coor_dirs, class_labels, class_names, class_targets,
        TRAIN_SAMPLES, Wout, mu, std, label="Training set")
    train_acc = float(np.mean(tr_true == tr_pred))
    print(f"\n  Train accuracy : {train_acc*100:.1f}%")

    # 3. Evaluate on test set
    results, te_true, te_pred = run_inference(
        coor_dirs, class_labels, class_names, class_targets,
        TEST_SAMPLES, Wout, mu, std, label="Test set")
    test_acc = float(np.mean(te_true == te_pred)) if len(te_true) else 0.0
    print(f"\n  Test accuracy  : {test_acc*100:.1f}%")

    if len(te_true):
        print("\n  Per-class report (test):")
        print(classification_report(te_true, te_pred,
                                    target_names=class_names,
                                    digits=3, zero_division=0))

    # 4. Save figures
    print()
    cm = confusion_matrix(te_true, te_pred, labels=class_labels)

    plot_staircase(results, class_labels, class_names,
                   TEST_SAMPLES, train_acc, test_acc,
                   out_dir / "perclass_staircase.pdf")

    plot_polar(results, class_labels, class_names, class_targets,
               train_acc, test_acc,
               out_dir / "perclass_polar.pdf")

    plot_confusion_matrix(cm, class_names, train_acc, test_acc,
                          out_dir / "perclass_confusion_matrix.pdf")

    plot_pca(pca, mu, std, coor_dirs, class_labels, class_names,
             TRAIN_SAMPLES, TEST_SAMPLES, N,
             out_dir / "perclass_pca.pdf")

    print(f"\n  Done.  Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
