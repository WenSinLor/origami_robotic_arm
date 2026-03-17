"""
Sensor Ablation Study — Leave-One-Out (LOO) by file
===================================================

This version upgrades the original single 8/4 train-test split to a
leave-one-out (LOO) protocol by file index.

Protocol
--------
Assume each class has files:
    trajectories_sample_0.h5 ... trajectories_sample_11.h5

For each LOO fold h = 0..11:
  - Hold out trajectories_sample_h.h5 from EACH class for test
  - Train on the other 11 files from EACH class
  - Fit OLS on all markers (same as original)
  - Run sensor ablation on the held-out test files
  - Optionally run LCS-guided and Wout-guided ablations

Final figures aggregate across:
  - all LOO folds
  - all random sensor subsets of the same size

Important
---------
This code keeps your ORIGINAL "trial MSE" definition unchanged:
    || mean_t y_hat(t) - y* ||^2
So the only methodological change is the train/test protocol (LOO),
not the metric definition.

Usage
-----
    python sensor_ablation_loo.py
"""

# -- Standard library --
from itertools import combinations
from pathlib import Path
import random

# -- Third-party --
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score

# ==============================================================================
#  CONFIG
# ==============================================================================

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "base": f"{ROOT_DIR}/soft_state_100g",
    # "near": f"{ROOT_DIR}/soft_state_100g_near",
}

COOR_DIRS     = ["coor_0", "coor_1", "coor_2", "coor_3"]
CLASS_LABELS  = [1, 2, 3, 4]
CLASS_NAMES   = ["coor_0", "coor_1", "coor_2", "coor_3"]
CLASS_TARGETS = np.array([[ 1.,  0.],
                          [ 0.,  1.],
                          [-1.,  0.],
                          [ 0., -1.]], dtype=float)

T_START = 0.0
T_END   = 3.0

# Total available files per class:
# trajectories_sample_0.h5 ... trajectories_sample_11.h5
N_TOTAL_SAMPLES_PER_CLASS = 12

# Cap on combinations evaluated per k.
# Set to None to evaluate exhaustively (exact but potentially very slow).
MAX_COMBOS  = 92378
RANDOM_SEED = 42

# Maximum number of sensors allowed to fail.
MAX_INVALID = 6

OUTPUT_DIR = f"{ROOT_DIR}/ablation_study"

# ==============================================================================
#  DATA LOADING
# ==============================================================================

def load_h5(path, baseline_frames=1):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]

    # Fill NaNs by nearest-hold logic
    for n in range(pos.shape[1]):
        for ax in range(2):
            col      = pos[:, n, ax]
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


def extract_features(disp, time, t_start, t_end):
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end,   side="right"))
    return disp[i0:i1].reshape(i1 - i0, -1)   # [T, N_markers * 2]


def resolve(dir_key, coor_dir, fname):
    return Path(DIRS[dir_key]) / coor_dir / fname


# ==============================================================================
#  LOO SPLIT
# ==============================================================================

def build_loo_split(held_out_idx):
    """
    Leave-one-out split by file index.

    For each class:
      - test  = trajectories_sample_{held_out_idx}.h5
      - train = all other sample_i.h5, i != held_out_idx
    """
    train_samples = {}
    test_samples  = {}

    for cname in CLASS_NAMES:
        train_samples[cname] = []
        test_samples[cname]  = []

        for i in range(N_TOTAL_SAMPLES_PER_CLASS):
            item = ("base", f"trajectories_sample_{i}.h5")
            if i == held_out_idx:
                test_samples[cname].append(item)
            else:
                train_samples[cname].append(item)

    return train_samples, test_samples


# ==============================================================================
#  OLS TRAINING
# ==============================================================================

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def nearest_class(p_xy, targets, labels):
    return int(labels[np.argmin(np.linalg.norm(targets - p_xy, axis=1))])


def fit_ols_full(train_samples_dict):
    """
    Train on all markers. Returns:
      Wout, mu, std, n_markers
    """
    Xb, Yb    = [], []
    n_markers = None

    print("=" * 60)
    print("  Fitting OLS on all markers (full reservoir)")
    print("=" * 60)

    for label, cdir, cname, tgt in zip(CLASS_LABELS, COOR_DIRS, CLASS_NAMES, CLASS_TARGETS):
        for dir_key, fname in train_samples_dict.get(cname, []):
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} -- skipped")
                continue

            disp, time = load_h5(p)
            X          = extract_features(disp, time, T_START, T_END)

            if n_markers is None:
                n_markers = X.shape[1] // 2

            Xb.append(X)
            Yb.append(np.tile(tgt, (len(X), 1)))
            print(f"  + [{dir_key}] {cdir}/{fname}  label={label}  T={len(X)}")

    if not Xb:
        raise RuntimeError("No training data found. Check DIRS or data files.")

    X_all = np.vstack(Xb)
    Y_all = np.vstack(Yb)

    mu  = X_all.mean(axis=0)
    std = X_all.std(axis=0)
    std[std < 1e-8] = 1.0

    Xn = (X_all - mu) / std

    Wout, _, _, _ = np.linalg.lstsq(add_bias(Xn), Y_all, rcond=None)

    print(f"\n  Detected n_markers = {n_markers}")
    print(f"  Feature dim        = {n_markers * 2}")
    print(f"  Wout shape         = {Wout.shape}")

    return Wout, mu, std, n_markers


# ==============================================================================
#  TEST DATA LOADING
# ==============================================================================

def load_test_data(test_samples_dict):
    """
    Returns list of tuples:
      (true_label, class_target [2], X [T, N*2])
    One item per test file.
    """
    data = []

    for label, cdir, cname, tgt in zip(CLASS_LABELS, COOR_DIRS, CLASS_NAMES, CLASS_TARGETS):
        for dir_key, fname in test_samples_dict.get(cname, []):
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                print(f"  [WARN] Missing test file: {p} -- skipped")
                continue

            disp, time = load_h5(p)
            X = extract_features(disp, time, T_START, T_END)
            data.append((label, tgt, X))

    return data


# ==============================================================================
#  MASKED INFERENCE
# ==============================================================================

def evaluate_subset(test_data, mu, std, Wout, active_markers, n_markers):
    """
    Evaluate one marker subset on all test trials.

    Inactive markers are simulated as physically dead sensors (zero raw
    displacement) using the two-step approach:
      1. Zero inactive columns in raw displacement space (X_masked).
      2. Normalise, then re-zero those columns to cancel the -mu/std residual.

    Returns dict:
      acc       : float, classification accuracy over all trials
      trial_mse : np.array [n_trials], each = ||mean_xy - target||^2
    """
    y_true    = []
    y_pred    = []
    trial_mse = []

    inactive = [m for m in range(n_markers) if m not in active_markers]

    for true_label, target, X in test_data:
        # Step 1: zero raw displacement for inactive markers
        X_masked = X.copy()
        for m in inactive:
            X_masked[:, 2 * m]     = 0.0
            X_masked[:, 2 * m + 1] = 0.0

        # Step 2: normalise
        Xn = (X_masked - mu) / std

        # Step 3: re-zero to fully remove inactive-marker contribution
        for m in inactive:
            Xn[:, 2 * m]     = 0.0
            Xn[:, 2 * m + 1] = 0.0

        Y_hat   = add_bias(Xn) @ Wout       # [T, 2]
        mean_xy = Y_hat.mean(axis=0)        # [2]

        pred = nearest_class(mean_xy, CLASS_TARGETS, CLASS_LABELS)
        y_true.append(true_label)
        y_pred.append(pred)

        # Original metric retained unchanged
        trial_mse.append(float(np.sum((mean_xy - target) ** 2)))

    return {
        "acc": accuracy_score(y_true, y_pred),
        "trial_mse": np.array(trial_mse),
    }


# ==============================================================================
#  RANDOM ABLATION SWEEP
# ==============================================================================

def run_ablation_sweep(Wout, mu, std, n_markers, test_data, rng):
    """
    For k = N down to max(1, N-MAX_INVALID):
      - Enumerate all C(N, k) subsets (or subsample MAX_COMBOS)
      - Evaluate each subset on the test set

    Returns:
      results[k] = list of eval dicts, one per tested subset
    """
    all_markers = list(range(n_markers))
    results     = {}

    # Full reservoir
    full = evaluate_subset(test_data, mu, std, Wout, set(all_markers), n_markers)
    results[n_markers] = [full]
    print(f"  k={n_markers:2d} (0 invalid, full reservoir) : "
          f"acc={full['acc']*100:.1f}%  "
          f"trial_mse mean={full['trial_mse'].mean():.4f}")

    # Reduced subsets
    k_min = max(1, n_markers - MAX_INVALID)

    for k in range(n_markers - 1, k_min - 1, -1):
        all_combos = list(combinations(all_markers, k))

        if MAX_COMBOS is not None and len(all_combos) > MAX_COMBOS:
            sampled = rng.sample(all_combos, MAX_COMBOS)
            tag = f"sampled {MAX_COMBOS} of {len(all_combos)}"
        else:
            sampled = all_combos
            tag = f"{len(sampled)} combos"

        k_results = []
        for subset in sampled:
            ev = evaluate_subset(test_data, mu, std, Wout, set(subset), n_markers)
            k_results.append(ev)

        results[k] = k_results

        accs  = [r["acc"] for r in k_results]
        tmses = [r["trial_mse"].mean() for r in k_results]
        n_invalid = n_markers - k

        print(f"  k={k:2d} ({n_invalid} invalid)  ({tag}) :  "
              f"acc mean={np.mean(accs)*100:.1f}%  "
              f"trial_mse mean={np.mean(tmses):.4f}  "
              f"max={np.max(tmses):.4f}  min={np.min(tmses):.4f}")

    return results


# ==============================================================================
#  MARKER CORRELATION ANALYSIS (LCS)
# ==============================================================================

def compute_lcs(train_samples_dict):
    """
    Compute Linear Correlation Score (LCS) for each marker.

    Returns:
      lcs      : [N_markers]
      lcs_rank : marker indices sorted ascending by LCS
      M        : correlation matrix [N_markers, N_markers]
    """
    marker_vecs = None
    n_markers   = None

    for cdir, cname in zip(COOR_DIRS, CLASS_NAMES):
        for dir_key, fname in train_samples_dict.get(cname, []):
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                continue

            pos, time = load_h5(p)
            i0 = int(np.searchsorted(time, T_START, side="left"))
            i1 = int(np.searchsorted(time, T_END,   side="right"))
            win = pos[i0:i1]   # [T, N_markers, 2]

            if n_markers is None:
                n_markers   = win.shape[1]
                marker_vecs = [[] for _ in range(n_markers)]

            for m in range(n_markers):
                marker_vecs[m].append(win[:, m, :])

    if n_markers is None:
        raise RuntimeError("No training data found for LCS computation.")

    rows = []
    for m in range(n_markers):
        vec  = np.concatenate(marker_vecs[m], axis=0).flatten()
        norm = np.linalg.norm(vec)
        rows.append(vec / norm if norm > 1e-8 else vec)

    delta_X = np.vstack(rows)     # [N, D]
    M       = delta_X @ delta_X.T
    lcs     = np.sum(np.abs(M), axis=1)
    lcs_rank = np.argsort(lcs)

    return lcs, lcs_rank, M


def compute_wout_importance(Wout, n_markers):
    """
    importance[m] = Frobenius norm of the two weight rows for marker m
    """
    importance = np.array([
        np.linalg.norm(Wout[2*m+1 : 2*m+3])
        for m in range(n_markers)
    ])
    importance_rank = np.argsort(importance)   # smallest first = remove first
    return importance, importance_rank


# ==============================================================================
#  PLOT STYLING
# ==============================================================================

NATURE_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "legend.fontsize":    7,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
}

BLUE       = "#0072B2"
BLUE_LIGHT = "#C6DCEF"
GREEN      = "#009E73"
ORANGE     = "#E69F00"
GRAY       = "#999999"


# ==============================================================================
#  RANDOM-ONLY FIGURE
# ==============================================================================

def plot_ablation_figure(results, n_markers, out_path):
    """
    Plot random-removal aggregated results.
    results[k] is a list of eval dicts. Under LOO aggregation, that list already
    contains data from all folds and all subsets.
    """
    chance = 1.0 / len(CLASS_LABELS)

    ks         = sorted(results.keys(), reverse=True)
    n_invalids = np.array([n_markers - k for k in ks])

    # Accuracy
    acc_means = np.array([np.mean([r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_stds  = np.array([np.std( [r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_maxs  = np.array([np.max( [r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_mins  = np.array([np.min( [r["acc"] for r in results[k]]) for k in ks]) * 100
    full_acc  = np.mean([r["acc"] for r in results[n_markers]]) * 100

    # MSE
    mse_means = np.array([np.mean([r["trial_mse"].mean() for r in results[k]]) for k in ks])
    mse_stds  = np.array([np.std( [r["trial_mse"].mean() for r in results[k]]) for k in ks])
    mse_maxs  = np.array([np.max( [r["trial_mse"].mean() for r in results[k]]) for k in ks])
    mse_mins  = np.array([np.min( [r["trial_mse"].mean() for r in results[k]]) for k in ks])
    full_mse  = np.mean([r["trial_mse"].mean() for r in results[n_markers]])

    with plt.rc_context(NATURE_RC):
        fig, (ax_acc, ax_mse) = plt.subplots(
            2, 1,
            figsize=(max(5.5, 0.8 * (MAX_INVALID + 1) + 1.5), 6.0),
            sharex=True,
            gridspec_kw={
                "hspace": 0.10,
                "height_ratios": [1, 1],
                "left": 0.12,
                "right": 0.97,
                "top": 0.93,
                "bottom": 0.10,
            }
        )

        # Panel a: accuracy
        ax_acc.text(-0.10, 1.06, "a", transform=ax_acc.transAxes,
                    fontsize=10, fontweight="bold", va="top")

        ax_acc.fill_between(n_invalids, acc_mins, acc_maxs,
                            color=BLUE, alpha=0.10, linewidth=0,
                            label="Min-max range")
        ax_acc.fill_between(n_invalids, acc_means - acc_stds, acc_means + acc_stds,
                            color=BLUE, alpha=0.22, linewidth=0,
                            label="Mean +/- 1 SD")
        ax_acc.plot(n_invalids, acc_maxs, color=GREEN,  lw=1.0, ls="--",
                    alpha=0.85, label="Best subset")
        ax_acc.plot(n_invalids, acc_mins, color=ORANGE, lw=1.0, ls=":",
                    alpha=0.85, label="Worst subset")
        ax_acc.plot(n_invalids, acc_means, color=BLUE, lw=2.0,
                    marker="o", ms=4.5, zorder=5, label="Mean accuracy")
        ax_acc.axhline(full_acc, color=BLUE, lw=1.2, ls="-", alpha=0.4,
                       label=f"Full reservoir ({full_acc:.0f}%)")
        ax_acc.axhline(chance * 100, color=GRAY, lw=1.0, ls="--",
                       label=f"Chance ({chance*100:.0f}%)")

        ax_acc.set_ylabel("Classification accuracy (%)")
        ax_acc.set_ylim(max(0, acc_mins.min() - 8), min(105, acc_maxs.max() + 12))
        ax_acc.legend(loc="lower right", frameon=True, framealpha=0.9,
                      edgecolor="#DDDDDD")

        # Panel b: MSE
        ax_mse.text(-0.10, 1.06, "b", transform=ax_mse.transAxes,
                    fontsize=10, fontweight="bold", va="top")

        ax_mse.fill_between(n_invalids, mse_mins, mse_maxs,
                            color=BLUE, alpha=0.10, linewidth=0,
                            label="Min-max range")
        ax_mse.fill_between(n_invalids, mse_means - mse_stds, mse_means + mse_stds,
                            color=BLUE, alpha=0.22, linewidth=0,
                            label="Mean +/- 1 SD")
        ax_mse.plot(n_invalids, mse_maxs, color=GREEN, lw=1.0, ls="--",
                    alpha=0.85, label="Worst subset (highest MSE)")
        ax_mse.plot(n_invalids, mse_mins, color=ORANGE, lw=1.0, ls=":",
                    alpha=0.85, label="Best subset (lowest MSE)")
        ax_mse.plot(n_invalids, mse_means, color=BLUE, lw=2.0,
                    marker="o", ms=4.5, zorder=5, label="Mean trial MSE")
        ax_mse.axhline(full_mse, color=BLUE, lw=1.2, ls="-", alpha=0.4,
                       label=f"Full reservoir ({full_mse:.3f})")

        ax_mse.set_xlabel("Number of invalid sensors")
        ax_mse.set_ylabel(r"Per-trial MSE  $\|\bar{\mathbf{y}} - \mathbf{y}^*\|^2$")
        ax_mse.set_ylim(0, max(mse_maxs.max() * 1.3, full_mse * 2.0))
        ax_mse.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax_mse.set_xticks(n_invalids)
        ax_mse.tick_params(axis="x", length=0)
        ax_mse.spines["bottom"].set_linewidth(0.4)
        ax_mse.legend(loc="upper right", frameon=True, framealpha=0.9,
                      edgecolor="#DDDDDD")

        fig.suptitle(
            "LOO sensor count ablation\nTrained on all markers — tested with k active markers",
            fontweight="bold", fontsize=9, y=0.99
        )

        fig.savefig(str(out_path))
        plt.close(fig)

    print(f"  Saved -> {out_path}")


# ==============================================================================
#  LCS-GUIDED / WOUT-GUIDED SWEEPS
# ==============================================================================

def run_lcs_guided_sweep(Wout, mu, std, n_markers, test_data, lcs_rank):
    """
    Deterministic ablation: remove highest-LCS markers first
    (most redundant by correlation criterion).
    """
    results_guided = {}
    all_markers    = set(range(n_markers))

    removal_order = list(reversed(lcs_rank.tolist()))  # highest LCS first

    for n_invalid in range(MAX_INVALID + 1):
        inactive = set(removal_order[:n_invalid])
        active   = all_markers - inactive

        ev = evaluate_subset(test_data, mu, std, Wout, active, n_markers)
        results_guided[n_invalid] = ev

        tag = f"remove {list(inactive)}" if n_invalid > 0 else "full reservoir"
        print(f"  n_invalid={n_invalid}  ({tag}) : "
              f"acc={ev['acc']*100:.1f}%  "
              f"trial_mse mean={ev['trial_mse'].mean():.4f}")

    return results_guided


def run_wout_guided_sweep(Wout, mu, std, n_markers, test_data, importance_rank):
    """
    Deterministic ablation: remove smallest-Wout-importance markers first.
    """
    results_wout = {}
    all_markers  = set(range(n_markers))

    removal_order = importance_rank.tolist()

    for n_invalid in range(MAX_INVALID + 1):
        inactive = set(removal_order[:n_invalid])
        active   = all_markers - inactive

        ev = evaluate_subset(test_data, mu, std, Wout, active, n_markers)
        results_wout[n_invalid] = ev

        tag = f"remove {list(inactive)}" if n_invalid > 0 else "full reservoir"
        print(f"  n_invalid={n_invalid}  ({tag}) : "
              f"acc={ev['acc']*100:.1f}%  "
              f"trial_mse mean={ev['trial_mse'].mean():.4f}")

    return results_wout


# ==============================================================================
#  LCS FIGURE
# ==============================================================================

def _lcs_threshold(lcs, lcs_rank):
    lcs_sorted = lcs[lcs_rank]
    gaps = np.diff(lcs_sorted)
    if gaps.std() > 1e-8:
        gap_z   = (gaps - gaps.mean()) / gaps.std()
        outlier = np.where(gap_z > 2.0)[0]
    else:
        outlier = np.array([])
    if len(outlier) > 0:
        return int(outlier[0]) + 1, True
    return 0, False


def plot_lcs_figure(lcs, lcs_rank, out_path):
    """
    Standalone LCS bar chart for the first fold (or representative fold).
    """
    n_m        = len(lcs_rank)
    lcs_sorted = lcs[lcs_rank]
    n_crucial, threshold_found = _lcs_threshold(lcs, lcs_rank)

    if threshold_found:
        bar_colours = [BLUE if i < n_crucial else BLUE_LIGHT for i in range(n_m)]
    else:
        cmap     = plt.cm.Blues
        norm_lcs = plt.Normalize(vmin=lcs_sorted.min(), vmax=lcs_sorted.max())
        bar_colours = [cmap(1.0 - norm_lcs(v) * 0.7 + 0.2) for v in lcs_sorted]

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(
            figsize=(max(6.0, 0.45 * n_m + 2.0), 3.8),
            gridspec_kw={"left": 0.10, "right": 0.97, "top": 0.88, "bottom": 0.22}
        )

        ax.bar(range(n_m), lcs_sorted, color=bar_colours,
               edgecolor="white", linewidth=0.4, zorder=2)

        if threshold_found:
            ax.axvline(n_crucial - 0.5, color=GRAY, lw=1.0, ls="--", alpha=0.7,
                       label=f"Gap z-score > 2.0  (n={n_crucial} crucial)")
            for i in range(n_crucial):
                m_id = lcs_rank[i]
                ax.text(i, lcs_sorted[i] + lcs.max() * 0.015,
                        f"M{m_id}", ha="center", va="bottom",
                        fontsize=6, color=BLUE, fontweight="bold")
            title = f"Linear Correlation Score — {n_crucial} crucial markers"
        else:
            for i in range(min(3, n_m)):
                m_id = lcs_rank[i]
                ax.text(i, lcs_sorted[i] + lcs.max() * 0.015,
                        f"M{m_id}", ha="center", va="bottom",
                        fontsize=6, color=BLUE, fontweight="bold")
            title = "Linear Correlation Score — no statistically significant gap"

        ax.set_xlabel("Marker rank (0 = most crucial / least correlated)")
        ax.set_ylabel("LCS")
        ax.set_title(title, fontweight="bold", pad=4)
        ax.set_xticks(range(n_m))
        ax.set_xticklabels([f"M{lcs_rank[i]}" for i in range(n_m)],
                           rotation=45, ha="right", fontsize=6)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        if threshold_found:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9,
                      edgecolor="#DDDDDD")

        fig.savefig(str(out_path))
        plt.close(fig)

    print(f"  Saved -> {out_path}")


# ==============================================================================
#  LOO-AGGREGATED COMPARISON FIGURE
# ==============================================================================

def plot_lcs_guided_figure_loo(results, results_guided, results_wout, n_markers, out_path):
    """
    Compare random removal (aggregated over folds and subsets) with:
      - LCS-guided removal (aggregated over folds)
      - Wout-guided removal (aggregated over folds)
    """
    chance     = 1.0 / len(CLASS_LABELS)
    ks         = sorted(results.keys(), reverse=True)
    n_invalids = np.array([n_markers - k for k in ks])

    # Random removal
    acc_means = np.array([np.mean([r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_stds  = np.array([np.std( [r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_maxs  = np.array([np.max( [r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_mins  = np.array([np.min( [r["acc"] for r in results[k]]) for k in ks]) * 100
    full_acc  = np.mean([r["acc"] for r in results[n_markers]]) * 100

    mse_means = np.array([np.mean([r["trial_mse"].mean() for r in results[k]]) for k in ks])
    mse_stds  = np.array([np.std( [r["trial_mse"].mean() for r in results[k]]) for k in ks])
    mse_maxs  = np.array([np.max( [r["trial_mse"].mean() for r in results[k]]) for k in ks])
    mse_mins  = np.array([np.min( [r["trial_mse"].mean() for r in results[k]]) for k in ks])
    full_mse  = np.mean([r["trial_mse"].mean() for r in results[n_markers]])

    # LCS-guided
    guided_n        = sorted(results_guided.keys())
    guided_acc_mean = np.array([np.mean([ev["acc"] for ev in results_guided[n]]) for n in guided_n]) * 100
    guided_acc_std  = np.array([np.std( [ev["acc"] for ev in results_guided[n]]) for n in guided_n]) * 100
    guided_mse_mean = np.array([np.mean([ev["trial_mse"].mean() for ev in results_guided[n]]) for n in guided_n])
    guided_mse_std  = np.array([np.std( [ev["trial_mse"].mean() for ev in results_guided[n]]) for n in guided_n])

    # Wout-guided
    wout_n        = sorted(results_wout.keys())
    wout_acc_mean = np.array([np.mean([ev["acc"] for ev in results_wout[n]]) for n in wout_n]) * 100
    wout_acc_std  = np.array([np.std( [ev["acc"] for ev in results_wout[n]]) for n in wout_n]) * 100
    wout_mse_mean = np.array([np.mean([ev["trial_mse"].mean() for ev in results_wout[n]]) for n in wout_n])
    wout_mse_std  = np.array([np.std( [ev["trial_mse"].mean() for ev in results_wout[n]]) for n in wout_n])

    RED    = "#D55E00"
    PURPLE = "#CC79A7"

    with plt.rc_context(NATURE_RC):
        fig, (ax_acc, ax_mse) = plt.subplots(
            2, 1,
            figsize=(max(5.5, 0.8 * (MAX_INVALID + 1) + 1.5), 6.0),
            sharex=True,
            gridspec_kw={
                "hspace": 0.10,
                "height_ratios": [1, 1],
                "left": 0.12,
                "right": 0.97,
                "top": 0.93,
                "bottom": 0.10,
            }
        )

        # Accuracy
        ax_acc.text(-0.10, 1.06, "a", transform=ax_acc.transAxes,
                    fontsize=10, fontweight="bold", va="top")

        ax_acc.fill_between(n_invalids, acc_mins, acc_maxs,
                            color=BLUE, alpha=0.10, linewidth=0,
                            label="Random: min-max range")
        ax_acc.fill_between(n_invalids, acc_means - acc_stds, acc_means + acc_stds,
                            color=BLUE, alpha=0.22, linewidth=0,
                            label="Random: mean +/- 1 SD")
        ax_acc.plot(n_invalids, acc_maxs, color=GREEN, lw=1.0, ls="--",
                    alpha=0.85, label="Random: best subset")
        ax_acc.plot(n_invalids, acc_mins, color=ORANGE, lw=1.0, ls=":",
                    alpha=0.85, label="Random: worst subset")
        ax_acc.plot(n_invalids, acc_means, color=BLUE, lw=1.5,
                    marker="o", ms=3.5, alpha=0.85, label="Random: mean")

        ax_acc.fill_between(guided_n, guided_acc_mean - guided_acc_std, guided_acc_mean + guided_acc_std,
                            color=RED, alpha=0.18, linewidth=0)
        ax_acc.plot(guided_n, guided_acc_mean, color=RED, lw=2.0,
                    marker="s", ms=4.5, label="LCS-guided")

        ax_acc.fill_between(wout_n, wout_acc_mean - wout_acc_std, wout_acc_mean + wout_acc_std,
                            color=PURPLE, alpha=0.18, linewidth=0)
        ax_acc.plot(wout_n, wout_acc_mean, color=PURPLE, lw=2.0,
                    marker="^", ms=4.5, label="Wout-guided")

        ax_acc.axhline(full_acc, color=BLUE, lw=1.0, ls="-", alpha=0.35,
                       label=f"Full reservoir ({full_acc:.0f}%)")
        ax_acc.axhline(chance * 100, color=GRAY, lw=1.0, ls="--",
                       label=f"Chance ({chance*100:.0f}%)")

        ax_acc.set_ylabel("Classification accuracy (%)")
        ax_acc.set_ylim(max(0, acc_mins.min() - 8), min(105, acc_maxs.max() + 12))
        ax_acc.legend(loc="lower left", frameon=True, framealpha=0.9,
                      edgecolor="#DDDDDD", fontsize=6)

        # MSE
        ax_mse.text(-0.10, 1.06, "b", transform=ax_mse.transAxes,
                    fontsize=10, fontweight="bold", va="top")

        ax_mse.fill_between(n_invalids, mse_mins, mse_maxs,
                            color=BLUE, alpha=0.10, linewidth=0,
                            label="Random: min-max range")
        ax_mse.fill_between(n_invalids, mse_means - mse_stds, mse_means + mse_stds,
                            color=BLUE, alpha=0.22, linewidth=0,
                            label="Random: mean +/- 1 SD")
        ax_mse.plot(n_invalids, mse_maxs, color=GREEN, lw=1.0, ls="--",
                    alpha=0.85, label="Random: worst subset")
        ax_mse.plot(n_invalids, mse_mins, color=ORANGE, lw=1.0, ls=":",
                    alpha=0.85, label="Random: best subset")
        ax_mse.plot(n_invalids, mse_means, color=BLUE, lw=1.5,
                    marker="o", ms=3.5, alpha=0.85, label="Random: mean")

        ax_mse.fill_between(guided_n, guided_mse_mean - guided_mse_std, guided_mse_mean + guided_mse_std,
                            color=RED, alpha=0.18, linewidth=0)
        ax_mse.plot(guided_n, guided_mse_mean, color=RED, lw=2.0,
                    marker="s", ms=4.5, label="LCS-guided")

        ax_mse.fill_between(wout_n, wout_mse_mean - wout_mse_std, wout_mse_mean + wout_mse_std,
                            color=PURPLE, alpha=0.18, linewidth=0)
        ax_mse.plot(wout_n, wout_mse_mean, color=PURPLE, lw=2.0,
                    marker="^", ms=4.5, label="Wout-guided")

        ax_mse.axhline(full_mse, color=BLUE, lw=1.0, ls="-", alpha=0.35,
                       label=f"Full reservoir ({full_mse:.3f})")

        ax_mse.set_xlabel("Number of invalid sensors")
        ax_mse.set_ylabel(r"Per-trial MSE  $\|\bar{\mathbf{y}} - \mathbf{y}^*\|^2$")
        ax_mse.set_ylim(0, max(mse_maxs.max() * 1.3, full_mse * 2.0))
        ax_mse.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax_mse.set_xticks(n_invalids)
        ax_mse.tick_params(axis="x", length=0)
        ax_mse.spines["bottom"].set_linewidth(0.4)
        ax_mse.legend(loc="upper left", frameon=True, framealpha=0.9,
                      edgecolor="#DDDDDD", fontsize=6)

        fig.suptitle(
            "LOO guided vs random sensor failure\n"
            "LCS-guided and Wout-guided vs random removal",
            fontweight="bold", fontsize=9, y=0.99
        )

        fig.savefig(str(out_path))
        plt.close(fig)

    print(f"  Saved -> {out_path}")


# ==============================================================================
#  ONE LOO FOLD
# ==============================================================================

def run_one_loo_fold(held_out_idx, rng):
    """
    Run one LOO fold.
    """
    print(f"\n{'='*70}")
    print(f"  LOO fold: hold out sample_{held_out_idx}.h5 from each class")
    print(f"{'='*70}")

    train_samples, test_samples = build_loo_split(held_out_idx)

    # Train
    Wout, mu, std, n_markers = fit_ols_full(train_samples)

    # Rankings from training data only
    print("\n  Computing Linear Correlation Scores...")
    lcs, lcs_rank, M_corr = compute_lcs(train_samples)

    print("\n  Computing Wout importance scores...")
    importance, importance_rank = compute_wout_importance(Wout, n_markers)

    # Load test
    print("\n  Pre-loading test data...")
    test_data = load_test_data(test_samples)
    print(f"  {len(test_data)} test trials loaded for this fold.")

    # Random ablation
    print("\n  Running random ablation sweep...")
    results = run_ablation_sweep(Wout, mu, std, n_markers, test_data, rng)

    # Guided ablations
    print("\n  Running LCS-guided ablation sweep...")
    results_guided = run_lcs_guided_sweep(Wout, mu, std, n_markers, test_data, lcs_rank)

    print("\n  Running Wout-importance guided ablation sweep...")
    results_wout = run_wout_guided_sweep(Wout, mu, std, n_markers, test_data, importance_rank)

    return {
        "held_out_idx": held_out_idx,
        "n_markers": n_markers,
        "results": results,
        "results_guided": results_guided,
        "results_wout": results_wout,
        "lcs": lcs,
        "lcs_rank": lcs_rank,
        "importance": importance,
        "importance_rank": importance_rank,
    }


# ==============================================================================
#  AGGREGATE ACROSS LOO FOLDS
# ==============================================================================

def aggregate_loo_results(all_folds):
    """
    Aggregate:
      - random ablation results across folds and subsets
      - guided results across folds
    """
    if len(all_folds) == 0:
        raise RuntimeError("No fold results to aggregate.")

    agg_random = {}
    agg_guided = {}
    agg_wout   = {}

    n_markers = all_folds[0]["n_markers"]

    for fold in all_folds:
        for k, eval_list in fold["results"].items():
            agg_random.setdefault(k, []).extend(eval_list)

        for n_invalid, ev in fold["results_guided"].items():
            agg_guided.setdefault(n_invalid, []).append(ev)

        for n_invalid, ev in fold["results_wout"].items():
            agg_wout.setdefault(n_invalid, []).append(ev)

    return agg_random, agg_guided, agg_wout, n_markers


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    rng     = random.Random(RANDOM_SEED)
    out_dir = Path(OUTPUT_DIR) / "loo_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sensor Ablation Study -- LOO by file")
    print(f"{'='*60}")
    print(f"  Time window : {T_START}-{T_END} s")
    print(f"  MAX_COMBOS  : {MAX_COMBOS}")
    print(f"  Random seed : {RANDOM_SEED}")
    print(f"  LOO folds   : {N_TOTAL_SAMPLES_PER_CLASS}\n")

    all_folds = []

    for held_out_idx in range(N_TOTAL_SAMPLES_PER_CLASS):
        fold_result = run_one_loo_fold(held_out_idx, rng)
        all_folds.append(fold_result)

    # Aggregate across folds
    agg_random, agg_guided, agg_wout, n_markers = aggregate_loo_results(all_folds)

    # Summary
    full_acc = np.mean([r["acc"] for r in agg_random[n_markers]])
    full_mse = np.mean([r["trial_mse"].mean() for r in agg_random[n_markers]])

    print(f"\n{'--'*25}")
    print("  LOO aggregated summary")
    print(f"    Full reservoir mean accuracy : {full_acc*100:.2f}%")
    print(f"    Full reservoir mean trial MSE: {full_mse:.4f}")
    print(f"{'--'*25}\n")

    # Random-only figure
    print("  Generating aggregated random-ablation figure...")
    plot_ablation_figure(agg_random, n_markers, out_dir / "ablation_curve_loo.png")

    # Guided comparison figure
    print("  Generating aggregated guided-comparison figure...")
    plot_lcs_guided_figure_loo(
        agg_random, agg_guided, agg_wout, n_markers,
        out_dir / "ablation_lcs_guided_loo.png"
    )

    # Optional: save LCS figure from first fold as representative ranking
    print("  Generating representative LCS figure from fold 0...")
    plot_lcs_figure(
        all_folds[0]["lcs"],
        all_folds[0]["lcs_rank"],
        out_dir / "ablation_lcs_fold0.png"
    )

    # Save numeric results
    np.save(
        str(out_dir / "ablation_results_loo.npy"),
        {
            "all_folds": all_folds,
            "agg_random": agg_random,
            "agg_guided": agg_guided,
            "agg_wout": agg_wout,
            "n_markers": n_markers,
            "MAX_INVALID": MAX_INVALID,
            "full_acc": full_acc,
            "full_mse": full_mse,
        },
        allow_pickle=True
    )

    print(f"\n  Done. Output -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()