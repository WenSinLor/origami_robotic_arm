"""
Sensor Ablation Study — Marker Count vs Accuracy and Frame-level MSE
=====================================================================
Trains the 2D OLS classifier on ALL markers (identical to per_class_classifier.py),
then evaluates on the test set with progressively fewer active markers:

    all N markers -> N-1 -> N-2 -> ... -> 1

For each k (number of active markers):
  - Every combination C(N, k) is tested (or MAX_COMBOS random subsets).
  - Per each combination two quantities are recorded:
      acc       : classification accuracy across all test trials
      frame_mse : frame-level MSE ||Y_hat(t) - target||^2, one value per frame
                  across all test trials (n_trials * T values per subset)

Two-panel figure:
  Panel a (top)    : accuracy vs k — mean +/- SD band, best/worst envelopes
  Panel b (bottom) : frame-level MSE vs k — violin per k with mean +/- std
                     error bar overlaid, matching the style of mse_comparison.py

Usage
-----
    python sensor_ablation.py
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
#  CONFIG  -- edit this block only
# ==============================================================================

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "base": f"{ROOT_DIR}/soft_state_100g",
    # "near": f"{ROOT_DIR}/soft_state_100g_near",
}

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
    ],
}

TEST_SAMPLES = {
    "coor_0": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
    ],
    "coor_1": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
    ],
    "coor_2": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
    ],
    "coor_3": [
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
    ],
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

# Cap on combinations evaluated per k.
# Set to None to evaluate exhaustively (exact but slow for large N).
MAX_COMBOS  = 200
RANDOM_SEED = 42

OUTPUT_DIR = f"{ROOT_DIR}/ablation_study"


# ==============================================================================
#  DATA LOADING  (identical to per_class_classifier.py)
# ==============================================================================

def load_h5(path, baseline_frames=1):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]
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
#  OLS TRAINING  (identical to per_class_classifier.py)
# ==============================================================================

def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def nearest_class(p_xy, targets, labels):
    return int(labels[np.argmin(np.linalg.norm(targets - p_xy, axis=1))])


def fit_ols_full(train_samples_dict):
    """Train on all markers. Returns Wout, mu, std, n_markers."""
    Xb, Yb    = [], []
    n_markers = None

    print("=" * 60)
    print("  Fitting OLS on all markers (full reservoir)")
    print("=" * 60)

    for label, cdir, cname, tgt in zip(CLASS_LABELS, COOR_DIRS,
                                        CLASS_NAMES, CLASS_TARGETS):
        for dir_key, fname in train_samples_dict.get(cname, []):
            p = resolve(dir_key, cdir, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} -- skipped")
                continue
            disp, time = load_h5(p)
            X          = extract_features(disp, time, T_START, T_END)
            if n_markers is None:
                n_markers = X.shape[1] // 2   # auto-detected from data
            Xb.append(X)
            Yb.append(np.tile(tgt, (len(X), 1)))
            print(f"  + [{dir_key}] {cdir}/{fname}  label={label}  T={len(X)}")

    if not Xb:
        raise RuntimeError("No training data found. Check TRAIN_SAMPLES and DIRS.")

    X_all = np.vstack(Xb)
    mu    = X_all.mean(axis=0)
    std   = X_all.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn    = (X_all - mu) / std

    Wout, _, _, _ = np.linalg.lstsq(add_bias(Xn), np.vstack(Yb), rcond=None)

    print(f"\n  Detected n_markers = {n_markers}")
    print(f"  Feature dim        = {n_markers * 2}")
    print(f"  Wout shape         = {Wout.shape}")
    return Wout, mu, std, n_markers


# ==============================================================================
#  TEST DATA LOADING
# ==============================================================================

def load_test_data(test_samples_dict):
    """Returns list of (true_label, class_target [2], X [T, N*2]) per trial."""
    data = []
    for label, cdir, cname, tgt in zip(CLASS_LABELS, COOR_DIRS,
                                        CLASS_NAMES, CLASS_TARGETS):
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

    Inactive markers are silenced by zeroing their rows in a copy of Wout.
    Wout shape: [1 + N*2, 2] — row 0 is the bias; marker m occupies
    rows 2*m+1 (x weight) and 2*m+2 (y weight).

    Returns dict:
      acc       : float, classification accuracy over all trials
      frame_mse : np.array [n_trials * T], ||Y_hat(t) - target||^2 per frame
                  across all test trials concatenated. This is the full
                  distribution of per-frame squared errors — used for the
                  violin plot in the bottom panel.
    """
    y_true    = []
    y_pred    = []
    frame_mse = []

    # Build masked Wout once for this subset.
    Wout_masked = Wout.copy()
    for m in range(n_markers):
        if m not in active_markers:
            Wout_masked[2 * m + 1] = 0.0   # x-weight row for marker m
            Wout_masked[2 * m + 2] = 0.0   # y-weight row for marker m

    for true_label, target, X in test_data:
        Xn      = (X - mu) / std               # normalise normally, X untouched
        Y_hat   = add_bias(Xn) @ Wout_masked   # [T, 2]
        mean_xy = Y_hat.mean(axis=0)            # [2] time-averaged prediction

        # Classification
        pred = nearest_class(mean_xy, CLASS_TARGETS, CLASS_LABELS)
        y_true.append(true_label)
        y_pred.append(pred)

        # Frame-level MSE: ||Y_hat(t) - target||^2 per frame  [T]
        frame_errors = np.sum((Y_hat - target) ** 2, axis=1)
        frame_mse.extend(frame_errors.tolist())

    return {
        "acc":       accuracy_score(y_true, y_pred),
        "frame_mse": np.array(frame_mse),
    }


# ==============================================================================
#  ABLATION SWEEP
# ==============================================================================

def run_ablation_sweep(Wout, mu, std, n_markers, test_data, rng):
    """
    For k = N down to 1:
      - Enumerate all C(N, k) subsets (or subsample MAX_COMBOS).
      - Evaluate accuracy and frame-level MSE for each subset.

    Returns dict: k -> list of dicts, one per evaluated subset.
    Each dict has keys:
      acc           : float, classification accuracy
      frame_mse     : np.array [n_trials * T], full per-frame MSE distribution

    The full frame_mse array is kept (not just the mean) so the violin plot
    can show the true distribution of per-frame errors at each k.
    """
    all_markers = list(range(n_markers))
    results     = {}

    # k = N: full reservoir
    full = evaluate_subset(test_data, mu, std, Wout,
                           set(all_markers), n_markers)
    results[n_markers] = [full]
    print(f"  k={n_markers:2d} (full reservoir) : "
          f"acc={full['acc']*100:.1f}%  "
          f"frame_mse mean={full['frame_mse'].mean():.4f}  "
          f"median={np.median(full['frame_mse']):.4f}")

    for k in range(n_markers - 1, 0, -1):
        all_combos = list(combinations(all_markers, k))

        if MAX_COMBOS is not None and len(all_combos) > MAX_COMBOS:
            sampled = rng.sample(all_combos, MAX_COMBOS)
            tag = f"sampled {MAX_COMBOS} of {len(all_combos)}"
        else:
            sampled = all_combos
            tag = f"{len(sampled)} combos"

        k_results = []
        for subset in sampled:
            ev = evaluate_subset(test_data, mu, std, Wout,
                                 set(subset), n_markers)
            k_results.append(ev)

        results[k] = k_results
        accs  = [r["acc"]                  for r in k_results]
        means = [r["frame_mse"].mean()     for r in k_results]
        print(f"  k={k:2d}  ({tag}) :  "
              f"acc mean={np.mean(accs)*100:.1f}%  "
              f"frame_mse mean={np.mean(means):.4f}  "
              f"max={np.max(means):.4f}  min={np.min(means):.4f}")

    return results


# ==============================================================================
#  PLOT
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


def plot_ablation_figure(results, n_markers, out_path):
    """
    Two-panel figure sharing the x-axis.

    Panel a (top): accuracy vs k
      Mean +/- SD shaded band, best/worst subset envelopes, full-reservoir
      reference line, chance baseline.

    Panel b (bottom): frame-level MSE vs k
      One violin per k, showing the full distribution of ||Y_hat(t)-target||^2
      across all frames and all evaluated subsets at that k.
      Mean +/- std error bar overlaid per violin.
      Full-reservoir reference line (horizontal dashed).
      Style matches mse_comparison.py.
    """
    chance   = 1.0 / len(CLASS_LABELS)
    ks       = sorted(results.keys(), reverse=True)   # N -> 1
    ks_arr   = np.array(ks)

    # -- Accuracy summary arrays --
    acc_means = np.array([np.mean([r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_stds  = np.array([np.std( [r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_maxs  = np.array([np.max( [r["acc"] for r in results[k]]) for k in ks]) * 100
    acc_mins  = np.array([np.min( [r["acc"] for r in results[k]]) for k in ks]) * 100
    full_acc  = results[n_markers][0]["acc"] * 100

    # -- Frame MSE: pool all subsets' frame arrays into one distribution per k --
    # At each k we concatenate frame_mse across all evaluated subsets,
    # giving the full empirical distribution of per-frame errors at that k.
    frame_distributions = [
        np.concatenate([r["frame_mse"] for r in results[k]])
        for k in ks
    ]
    full_frame_mse = results[n_markers][0]["frame_mse"]   # k=N reference

    with plt.rc_context(NATURE_RC):
        fig, (ax_acc, ax_mse) = plt.subplots(
            2, 1, figsize=(max(6.0, 0.45 * n_markers + 2.5), 6.0),
            sharex=True,
            gridspec_kw={"hspace": 0.10, "height_ratios": [1, 1]})

        # ── Panel a: accuracy ──────────────────────────────────────────────
        ax_acc.text(-0.10, 1.06, "a", transform=ax_acc.transAxes,
                    fontsize=10, fontweight="bold", va="top")

        ax_acc.fill_between(ks_arr, acc_mins, acc_maxs,
                            color=BLUE, alpha=0.10, linewidth=0,
                            label="Min-max range")
        ax_acc.fill_between(ks_arr, acc_means - acc_stds, acc_means + acc_stds,
                            color=BLUE, alpha=0.22, linewidth=0,
                            label="Mean +/- 1 SD")
        ax_acc.plot(ks_arr, acc_maxs, color=GREEN,  lw=1.0, ls="--",
                    alpha=0.85, label="Best subset")
        ax_acc.plot(ks_arr, acc_mins, color=ORANGE, lw=1.0, ls=":",
                    alpha=0.85, label="Worst subset")
        ax_acc.plot(ks_arr, acc_means, color=BLUE, lw=2.0,
                    marker="o", ms=4.5, zorder=5, label="Mean accuracy")
        ax_acc.axhline(full_acc, color=BLUE, lw=1.2, ls="-", alpha=0.4,
                       label=f"Full reservoir ({full_acc:.0f}%)")
        ax_acc.axhline(chance * 100, color=GRAY, lw=1.0, ls="--",
                       label=f"Chance ({chance*100:.0f}%)")

        ax_acc.set_ylabel("Classification accuracy (%)")
        ax_acc.set_ylim(max(0, acc_mins.min() - 8),
                        min(105, acc_maxs.max() + 12))
        ax_acc.legend(loc="lower right", frameon=True,
                      framealpha=0.9, edgecolor="#DDDDDD")

        # ── Panel b: frame-level MSE — violin per k ────────────────────────
        ax_mse.text(-0.10, 1.06, "b", transform=ax_mse.transAxes,
                    fontsize=10, fontweight="bold", va="top")

        # Reverse so violins are drawn left-to-right as k increases (1 -> N)
        for xi, (k, fdata) in enumerate(zip(reversed(ks), reversed(frame_distributions))):
            parts = ax_mse.violinplot(fdata, positions=[xi], widths=0.6,
                                      showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(BLUE_LIGHT)
                pc.set_edgecolor(BLUE)
                pc.set_alpha(0.65)
            parts["cmedians"].set_color(BLUE)
            parts["cmedians"].set_linewidth(1.5)

            # Mean +/- std error bar overlaid on violin
            ax_mse.errorbar(xi, fdata.mean(), yerr=fdata.std(),
                            fmt="none", ecolor=BLUE,
                            elinewidth=1.0, capsize=2.5, capthick=0.8,
                            zorder=5)

        # Full-reservoir reference line: horizontal at full_frame_mse mean
        ax_mse.axhline(full_frame_mse.mean(), color=BLUE, lw=1.2,
                       ls="--", alpha=0.5,
                       label=f"Full reservoir mean ({full_frame_mse.mean():.3f})")

        # x-ticks: label each violin with its k value (ascending left to right)
        ax_mse.set_xticks(range(len(ks)))
        ax_mse.set_xticklabels([str(k) for k in reversed(ks)], fontsize=7)
        ax_mse.set_xlabel("Number of active markers")
        ax_mse.set_ylabel(r"Frame-level MSE  $\|\mathbf{y}(t) - \mathbf{y}^*\|^2$")
        ax_mse.set_ylim(0, None)
        ax_mse.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax_mse.legend(loc="upper right", frameon=True,
                      framealpha=0.9, edgecolor="#DDDDDD")

        # Match x-axis spine style with panel a
        ax_mse.tick_params(axis="x", length=0)
        ax_mse.spines["bottom"].set_linewidth(0.4)

        fig.suptitle(
            "Sensor count ablation\n"
            "Trained on all markers — tested with k active markers",
            fontweight="bold", fontsize=9, y=0.99)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved -> {out_path}")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    rng     = random.Random(RANDOM_SEED)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sensor Ablation Study -- Marker Count vs Accuracy + Frame MSE")
    print(f"{'='*60}")
    print(f"  Time window : {T_START}-{T_END} s")
    print(f"  MAX_COMBOS  : {MAX_COMBOS}")
    print(f"  Random seed : {RANDOM_SEED}\n")

    # 1. Train on all markers
    Wout, mu, std, n_markers = fit_ols_full(TRAIN_SAMPLES)

    # 2. Pre-load all test trials
    print("\n  Pre-loading test data...")
    test_data = load_test_data(TEST_SAMPLES)
    print(f"  {len(test_data)} test trials loaded.")

    # 3. Sweep k from N down to 1
    print("\n  Running ablation sweep...")
    results = run_ablation_sweep(Wout, mu, std, n_markers, test_data, rng)

    # 4. Print summary
    full_acc      = results[n_markers][0]["acc"]
    full_frame_mse = results[n_markers][0]["frame_mse"]
    k1            = results.get(1, [])
    print(f"\n{'--'*25}")
    print(f"  Full reservoir ({n_markers} markers)")
    print(f"    Accuracy         : {full_acc*100:.1f}%")
    print(f"    Frame MSE mean   : {full_frame_mse.mean():.4f}")
    print(f"    Frame MSE median : {np.median(full_frame_mse):.4f}")
    if k1:
        k1_accs  = [r["acc"]               for r in k1]
        k1_fmses = [r["frame_mse"].mean()  for r in k1]
        print(f"  At k=1 (across all single-marker subsets):")
        print(f"    Acc       — mean={np.mean(k1_accs)*100:.1f}%  "
              f"best={np.max(k1_accs)*100:.1f}%  "
              f"worst={np.min(k1_accs)*100:.1f}%")
        print(f"    Frame MSE — mean={np.mean(k1_fmses):.4f}  "
              f"best={np.min(k1_fmses):.4f}  "
              f"worst={np.max(k1_fmses):.4f}")
    print(f"{'--'*25}\n")

    # 5. Plot
    print("  Generating figure...")
    plot_ablation_figure(results, n_markers, out_dir / "ablation_curve.png")

    # 6. Save numeric results
    np.save(str(out_dir / "ablation_results.npy"),
            {"results": results, "n_markers": n_markers,
             "full_acc": full_acc,
             "full_frame_mse_mean": full_frame_mse.mean()},
            allow_pickle=True)

    print(f"\n  Done.  Output -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()