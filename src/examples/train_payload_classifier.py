"""
Mixed-Source Payload Classifier — 2D OLS Regression
====================================================
Freely assign any samples from soft_state_100g and/or
soft_state_100g_near to the training and test sets.

Each sample is specified as a (directory, filename) pair,
giving full flexibility over the train/test composition.

Pipeline
--------
1. Load HDF5 trajectories from specified directories → baseline-subtract → slice window.
2. Stack all (class × train_sample) blocks → z-score → fit OLS.
3. Evaluate on train samples (sanity check) and test samples.
4. Produce diagnostic figures.

Usage
-----
    python mixed_source_classifier.py
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

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

# Shorthand aliases for directories — use these keys in TRAIN_SAMPLES / TEST_SAMPLES
DIRS = {
    "base" : f"{ROOT_DIR}/soft_state_40g",
    # "near" : f"{ROOT_DIR}/soft_state_40g",
}

# ── Define your train set here ────────────────────────────────────────────────
# Each entry: ("dir_key", "filename")
# Mix freely from any directory
TRAIN_SAMPLES = [
    ("base", "trajectories_sample_0.h5"),
    ("base", "trajectories_sample_1.h5"),
    ("base", "trajectories_sample_2.h5"),
    ("base", "trajectories_sample_3.h5"),
    ("base", "trajectories_sample_4.h5"),
    ("base", "trajectories_sample_5.h5"),
    ("base", "trajectories_sample_6.h5"),
    ("base", "trajectories_sample_7.h5"),
]

# ── Define your test set here ─────────────────────────────────────────────────
TEST_SAMPLES = [
    ("base", "trajectories_sample_8.h5"),
    ("base", "trajectories_sample_9.h5"),
    ("base", "trajectories_sample_10.h5"),
    ("base", "trajectories_sample_11.h5"),
    # ("base", "trajectories_sample_12.h5"),
    # ("base", "trajectories_sample_13.h5"),
    # ("base", "trajectories_sample_14.h5"),
    # ("base", "trajectories_sample_15.h5"),
]

# ── Everything else ───────────────────────────────────────────────────────────
COOR_DIRS = ["coor_0", "coor_1", "coor_2", "coor_3"]

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

OUTPUT_DIR = f"{ROOT_DIR}/soft_state_40g"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_label(dir_key, fname):
    """Short human-readable label e.g. 'base/sample_3'."""
    return f"{dir_key}/{fname.replace('trajectories_','').replace('.h5','')}"


def resolve(dir_key, coor_dir, fname):
    """Resolve full path from dir key, coordinate subdir, and filename."""
    return Path(DIRS[dir_key]) / coor_dir / fname


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def load_h5(path: Path, baseline_frames: int = 1):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]

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
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end, side="right")) if t_end is not None else len(time)
    X  = disp[i0:i1].reshape(i1 - i0, -1)
    ts = time[i0:i1] - time[i0]
    if exclude_markers:
        N    = X.shape[1] // 2
        keep = [c for n in range(N) if n not in exclude_markers
                for c in (2*n, 2*n+1)]
        X    = X[:, keep]
    return X, ts


# ══════════════════════════════════════════════════════════════════════════════
#  OLS CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def nearest_class(p_xy, targets, labels):
    return int(labels[np.argmin(np.linalg.norm(targets - p_xy, axis=1))])


def fit_ols(coor_dirs, class_labels, class_targets, train_samples,
            pca_variance=0.95):
    """
    Stack training data from all classes × train_samples (mixed sources).
    train_samples: list of (dir_key, filename) tuples.
    """
    Xb, Yb = [], []
    N = D = None
    ts_ref = None

    print("=" * 60)
    print("  Building training matrix")
    print("=" * 60)

    for label, cdir, tgt in zip(class_labels, coor_dirs, class_targets):
        for dir_key, fname in train_samples:
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} — skipped")
                continue
            disp, time = load_h5(p)
            X, ts = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            if D is None:
                D, N, ts_ref = X.shape[1], X.shape[1] // 2, ts
            Xb.append(X)
            Yb.append(np.tile(tgt, (len(X), 1)))
            print(f"  + [{dir_key}] {cdir}/{fname}  label={label}  T={len(X)}")

    if not Xb:
        raise RuntimeError("No training files found. Check TRAIN_SAMPLES and DIRS.")

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
                  samples, Wout, mu, std, label=""):
    """
    Run OLS inference.
    samples: list of (dir_key, filename) tuples.
    """
    results = []
    y_true, y_pred = [], []

    print(f"\n{'=' * 60}")
    print(f"  Inference — {label}")
    print(f"{'=' * 60}")

    for lab, cdir, cname, tgt in zip(class_labels, coor_dirs, class_names, class_targets):
        for dir_key, fname in samples:
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

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]

# Marker style per directory key so you can visually distinguish sources
DIR_MARKERS = {
    "base" : "o",
    "near" : "s",   # square for near
}


def plot_staircase(results, class_labels, class_names,
                   train_samples, test_samples,
                   train_acc, test_acc, out_path):
    COL_TARGET = "#333333"
    COL_X      = "#0072B2"
    COL_Y      = "#E69F00"
    COL_OK     = "#009E73"
    COL_BAD    = "#D55E00"

    # Order results to match test_samples list
    test_labels = [sample_label(dk, fn) for dk, fn in test_samples]
    ordered = [r for lab in class_labels
                 for sl in test_labels
                 for r in results
                 if r["label"] == lab and r["sample"] == sl]

    if not ordered:
        print("[WARN] No results to plot staircase.")
        return

    with plt.rc_context({**NATURE_RC,
                         "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7.5}):
        fig, (axx, axy) = plt.subplots(2, 1, figsize=(11, 3.8), sharex=True,
                                        gridspec_kw={"hspace": 0.08})

        t_offset, sample_mids, first = 0.0, [], True
        for r in ordered:
            ts  = r["ts"]
            dt  = float(ts[1] - ts[0]) if len(ts) > 1 else 1/30
            ta  = ts + t_offset
            tx, ty = float(r["target_xy"][0]), float(r["target_xy"][1])
            mx, my = float(r["mean_xy"][0]),   float(r["mean_xy"][1])
            col_mu = COL_OK if r["pred_class"] == r["label"] else COL_BAD
            sx = float(r["Y_hat"][:,0].std())
            sy = float(r["Y_hat"][:,1].std())

            # Shade background by source directory
            bg_col = "#EEF4FB" if r["dir_key"] == "base" else "#FFF8EE"
            for ax in (axx, axy):
                ax.axvspan(ta[0], ta[-1], color=bg_col, alpha=0.5, zorder=0)

            for ax, sig, tgt, mean_v, sd_v, col_sig, ylabel in [
                (axx, r["Y_hat"][:,0], tx, mx, sx, COL_X, "x readout"),
                (axy, r["Y_hat"][:,1], ty, my, sy, COL_Y, "y readout"),
            ]:
                ax.hlines(tgt, ta[0], ta[-1], colors=COL_TARGET, lw=1.8,
                          ls="--", alpha=0.7,
                          label="Target" if first and ax is axx else "")
                ax.plot(ta, sig, color=col_sig, lw=0.8, alpha=0.8,
                        label=("x̂(t)" if ax is axx else "ŷ(t)") if first else "")
                ax.fill_between(ta, mean_v - sd_v, mean_v + sd_v,
                                color=col_mu, alpha=0.15, linewidth=0,
                                label="±1 SD" if first and ax is axx else "")
                ax.hlines(mean_v, ta[0], ta[-1], colors=col_mu, lw=1.5,
                          ls="-", alpha=0.9,
                          label="Mean" if first and ax is axx else "")
                ax.text(ta[len(ta)//2], mean_v + sd_v + 0.06,
                        f"{mean_v:+.2f}\n±{sd_v:.2f}",
                        ha="center", va="bottom", fontsize=5.0,
                        color=col_mu, fontweight="bold", linespacing=1.3)
                ax.set_ylabel(ylabel)

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
                    ax.axvline(t_cur, color="#CCCCCC", lw=0.6, ls=":", zorder=1)
            t_cur += dur

        for ax in (axx, axy):
            ax.set_ylim(-2.1, 2.1)
            ax.set_yticks([-1, 0, 1])
            ax.spines["bottom"].set_linewidth(0.6)

        axy.set_xlabel("Time (s)")

        # Add source legend patches
        from matplotlib.patches import Patch
        extra_handles = [
            Patch(facecolor="#EEF4FB", edgecolor="#AAAAAA", label="base"),
            Patch(facecolor="#FFF8EE", edgecolor="#AAAAAA", label="near"),
        ]
        axx.legend(loc="upper left", frameon=True, framealpha=0.9,
                   edgecolor="#DDDDDD", ncol=5,
                   handles=axx.get_legend_handles_labels()[0] + extra_handles)

        # Class labels on top axis
        n_per = len([r for r in ordered if r["label"] == class_labels[0]])
        ax_top = axx.secondary_xaxis("top")
        if n_per > 0 and len(sample_mids) >= n_per * len(class_labels):
            ax_top.set_xticks([sample_mids[i*n_per + n_per//2]
                               for i in range(len(class_labels))])
            ax_top.set_xticklabels(class_names, fontsize=7.5, fontweight="bold")
        ax_top.tick_params(length=0)

        train_summary = ", ".join([sample_label(dk, fn) for dk, fn in train_samples])
        test_summary  = ", ".join([sample_label(dk, fn) for dk, fn in test_samples])
        fig.suptitle(
            f"Mixed-source 2D readout — "
            f"train acc = {train_acc*100:.0f}%   test acc = {test_acc*100:.0f}%\n"
            f"train: [{train_summary}]   test: [{test_summary}]",
            fontsize=6.5, y=1.01)

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
                       color=COLORS[lab], edgecolor="#333333",
                       linewidths=0.6, zorder=5)
            ax.text(theta, r + 0.10, name, ha="center", va="center",
                    fontsize=6, fontweight="bold", color=COLORS[lab])

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
            fontsize=6.5, pad=8)

        legend_handles = (
            [plt.Line2D([0],[0], marker="o", color=PALETTE[i % len(PALETTE)],
                        ls="None", markersize=4, label=n)
             for i, (_, n) in enumerate(zip(class_labels, class_names))] +
            [plt.Line2D([0],[0], marker="o", color="#555", ls="None",
                        markersize=4, label="base correct"),
             plt.Line2D([0],[0], marker="s", color="#555", ls="None",
                        markersize=4, label="near correct"),
             plt.Line2D([0],[0], marker="x", color="#555", ls="None",
                        markersize=5, markeredgewidth=1.0, label="Wrong")]
        )
        ax.legend(handles=legend_handles, fontsize=5.5,
                  loc="upper left", frameon=True, framealpha=0.9,
                  edgecolor="#DDDDDD")

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_confusion_matrix(cm_raw, class_names, train_acc, test_acc, out_path):
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
            ax.set_xticks(range(C)); ax.set_xticklabels(class_names, rotation=30, ha="right")
            ax.set_yticks(range(C)); ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
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
            ax.set_xlim(-0.5, C - 0.5); ax.set_ylim(C - 0.5, -0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
        axes[0].set_title("Count"); axes[1].set_title("Recall")
        fig.suptitle(
            f"Confusion matrix  (train acc = {train_acc*100:.0f}%,  "
            f"test acc = {test_acc*100:.0f}%)",
            fontsize=7.5, fontweight="bold", y=1.03)
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_pca(pca, mu, std, coor_dirs, class_labels, class_names,
             train_samples, test_samples, N, out_path):
    COLORS = [PALETTE[i % len(PALETTE)] for i in range(len(class_labels))]
    PCX, PCY = 0, 2

    proj_train = {cdir: [] for cdir in coor_dirs}
    proj_test  = {cdir: [] for cdir in coor_dirs}

    for cdir in coor_dirs:
        for dir_key, fname in train_samples:
            p = resolve(dir_key, cdir, fname)
            if not p.exists(): continue
            disp, time = load_h5(p)
            X, _ = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            proj_train[cdir].append(pca.transform((X - mu) / std))

        for dir_key, fname in test_samples:
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
                color="#9ECAE1", edgecolor="white", lw=0.4, label="Individual")
        ax1.plot(xs, cumvar * 100, color=PALETTE[0], lw=1.5,
                 marker="o", ms=3, label="Cumulative")
        ax1.axhline(95, color="#AAAAAA", lw=0.8, ls="--")
        ax1.axvline(k_95, color=PALETTE[2], lw=1.2, ls="--",
                    label=f"95% @ PC{k_95}")
        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Explained variance (%)")
        ax1.set_title("Scree plot", fontweight="bold")
        ax1.set_xlim(0.5, min(k + 0.5, 20.5)); ax1.set_ylim(0, 110)
        ax1.legend(fontsize=6)

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
                         cname, fontsize=6, color=col, fontweight="bold",
                         ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                   alpha=0.85, ec=col, lw=0.6))

        n_base_train = sum(1 for dk, _ in train_samples if dk == "base")
        n_near_train = sum(1 for dk, _ in train_samples if dk == "near")
        ax2.set_xlabel(f"PC{PCX+1}  ({pca.explained_variance_ratio_[PCX]*100:.1f}%)")
        ax2.set_ylabel(f"PC{PCY+1}  ({pca.explained_variance_ratio_[PCY]*100:.1f}%)"
                       if k > PCY else f"PC{PCY+1} (n/a)")
        ax2.set_title(f"PC{PCX+1} vs PC{PCY+1}  (filled=train, hollow=test)",
                      fontweight="bold")
        ax2.legend(fontsize=5.5, markerscale=2, loc="best")

        fig.suptitle(
            f"PCA diagnostic  |  {k} components  |  {N} markers × 2 = {N*2} features\n"
            f"train: {n_base_train}×base + {n_near_train}×near  |  "
            f"test: {sum(1 for dk,_ in test_samples if dk=='base')}×base + "
            f"{sum(1 for dk,_ in test_samples if dk=='near')}×near",
            fontsize=7, fontweight="bold", y=1.02)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to active classes
    idx           = [i for i, l in enumerate(CLASS_LABELS) if l in ACTIVE_CLASSES]
    coor_dirs     = [COOR_DIRS[i]    for i in idx]
    class_labels  = [CLASS_LABELS[i] for i in idx]
    class_names   = [CLASS_NAMES[i]  for i in idx]
    class_targets = CLASS_TARGETS[idx]

    n_base_train = sum(1 for dk, _ in TRAIN_SAMPLES if dk == "base")
    n_near_train = sum(1 for dk, _ in TRAIN_SAMPLES if dk == "near")
    n_base_test  = sum(1 for dk, _ in TEST_SAMPLES  if dk == "base")
    n_near_test  = sum(1 for dk, _ in TEST_SAMPLES  if dk == "near")

    print(f"\n{'═'*60}")
    print(f"  Mixed-Source Payload Classifier — 2D OLS")
    print(f"{'═'*60}")
    print(f"  Train : {n_base_train}×base + {n_near_train}×near = {len(TRAIN_SAMPLES)} samples")
    print(f"  Test  : {n_base_test}×base  + {n_near_test}×near  = {len(TEST_SAMPLES)} samples")
    print(f"  Classes     : {class_names}")
    print(f"  Time window : {T_START}–{T_END} s\n")

    # 1. Fit OLS on mixed training set
    Wout, pca, mu, std, N, D, _ = fit_ols(
        coor_dirs, class_labels, class_targets, TRAIN_SAMPLES)

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
                   TRAIN_SAMPLES, TEST_SAMPLES, train_acc, test_acc,
                   out_dir / "mixed_staircase.png")

    plot_polar(results, class_labels, class_names, class_targets,
               train_acc, test_acc,
               out_dir / "mixed_polar.png")

    plot_confusion_matrix(cm, class_names, train_acc, test_acc,
                          out_dir / "mixed_confusion_matrix.png")

    plot_pca(pca, mu, std, coor_dirs, class_labels, class_names,
             TRAIN_SAMPLES, TEST_SAMPLES, N,
             out_dir / "mixed_pca.png")

    print(f"\n  Done.  Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()