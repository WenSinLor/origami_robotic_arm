"""
Statistical Significance: Soft vs Stiff Configuration Comparison
=================================================================
Compares diagonal distances (predicted-to-correct-target) from two
configurations using three complementary methods appropriate for
small sample sizes:

  1. Permutation test     — p-value with no normality assumption
  2. Bootstrap 95% CI     — confidence interval on the mean difference
  3. Mann-Whitney U test  — non-parametric rank-based test
  4. Cohen's d            — effect size (how large is the difference?)

The diagonal distance = Euclidean distance from the raw predicted (x,y)
to the correct target. Lower = more confident correct prediction.

This script runs LOO-CV on BOTH configs internally, extracts diagonal
distances, then runs all statistical tests and produces:

  stats_significance.png  — permutation null distribution + observed diff
  stats_bootstrap.png     — bootstrap CI on mean difference
  stats_summary.png       — combined summary panel

Usage:  python loo_significance.py
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — set both configuration paths here
# ══════════════════════════════════════════════════════════════════════════════

ROOT = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

CONFIGS = {
    "Stiff (20g)":  f"{ROOT}/stiff_state_20g",   # ← update path as needed
    "Mix (20g)":  f"{ROOT}/mix_state_20g_right",   # ← update path as needed
}

COOR_DIRS = ["coor_0", "coor_1", "coor_2", "coor_3"]

ALL_SAMPLE_FILES = [
    "trajectories_sample_0.h5",
    "trajectories_sample_1.h5",
    "trajectories_sample_2.h5",
    "trajectories_sample_3.h5",
    "trajectories_sample_4.h5",
    "trajectories_sample_5.h5",
    "trajectories_sample_6.h5",
    "trajectories_sample_7.h5",
    "trajectories_sample_8.h5",
    "trajectories_sample_9.h5",
    "trajectories_sample_10.h5",
    "trajectories_sample_11.h5",
]

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

N_PERMUTATIONS = 10000
N_BOOTSTRAP    = 10000
ALPHA          = 0.05        # significance threshold

OUTPUT_DIR = f"{ROOT}/mix_state_20g_right"

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

NATURE_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_h5(path, baseline_frames=10):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]
    for n in range(pos.shape[1]):
        for ax in range(2):
            col = pos[:, n, ax]
            mask = np.isnan(col)
            if mask.all():
                col[:] = 0.0
            elif mask.any():
                idx = np.where(~mask)[0]
                col[:idx[0]]  = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i+1] - idx[i] > 1:
                        col[idx[i]+1:idx[i+1]] = col[idx[i]]
    baseline = pos[:min(baseline_frames, pos.shape[0])].mean(axis=0, keepdims=True)
    return pos - baseline, time


def extract_features(disp, time, exclude_markers):
    i0 = int(np.searchsorted(time, T_START, side="left"))
    i1 = int(np.searchsorted(time, T_END, side="right")) if T_END else len(time)
    X  = disp[i0:i1].reshape(i1 - i0, -1)
    if exclude_markers:
        N    = X.shape[1] // 2
        keep = [c for n in range(N) if n not in exclude_markers
                for c in (2*n, 2*n+1)]
        X = X[:, keep]
    return X


def add_bias(X):
    return np.hstack([np.ones((len(X), 1)), X])


def fit_ols(base, coor_dirs, class_labels, class_targets, train_files):
    Xb, Yb = [], []
    for label, cdir, tgt in zip(class_labels, coor_dirs, class_targets):
        for fname in train_files:
            p = base / cdir / fname
            if not p.exists(): continue
            disp, time = load_h5(p)
            X = extract_features(disp, time, EXCLUDE_MARKERS)
            Xb.append(X)
            Yb.append(np.tile(tgt, (len(X), 1)))
    if not Xb: return None
    X_all = np.vstack(Xb)
    mu = X_all.mean(0); std = X_all.std(0); std[std < 1e-8] = 1.0
    Wout, _, _, _ = np.linalg.lstsq(add_bias((X_all - mu) / std),
                                     np.vstack(Yb), rcond=None)
    return Wout, mu, std


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACT DIAGONAL DISTANCES VIA LOO-CV
# ══════════════════════════════════════════════════════════════════════════════

def get_diagonal_distances(base_dir, coor_dirs, class_labels,
                            class_names, class_targets, valid_files):
    """
    Run LOO-CV and return only the diagonal distances:
    distance from predicted (x,y) to the CORRECT target, per prediction.
    Returns array of shape (N_folds * C,).
    """
    base = Path(base_dir)
    diag_dists = []

    for held_out in valid_files:
        train_files = [f for f in valid_files if f != held_out]
        result = fit_ols(base, coor_dirs, class_labels, class_targets, train_files)
        if result is None: continue
        Wout, mu, std = result

        for ci, (label, cdir) in enumerate(zip(class_labels, coor_dirs)):
            p = base / cdir / held_out
            if not p.exists(): continue
            disp, time = load_h5(p)
            X = extract_features(disp, time, EXCLUDE_MARKERS)
            pred_xy = (add_bias((X - mu) / std) @ Wout).mean(axis=0)
            dist_correct = float(np.linalg.norm(class_targets[ci] - pred_xy))
            diag_dists.append(dist_correct)

    return np.array(diag_dists)


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Pooled Cohen's d: positive = a > b (a is worse/larger)."""
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)


def permutation_test(a, b, n_perm=N_PERMUTATIONS, rng=None):
    """
    Two-sample permutation test on the difference of means.
    H0: the two groups are drawn from the same distribution.
    Returns (observed_diff, null_distribution, p_value).
    observed_diff = mean(a) - mean(b)  (positive = a larger)
    p_value = P(|null_diff| >= |observed_diff|)  [two-tailed]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    observed = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    null = np.array([
        rng.permutation(combined)[:na].mean() -
        rng.permutation(combined)[na:].mean()
        for _ in range(n_perm)
    ])
    p_val = float(np.mean(np.abs(null) >= np.abs(observed)))
    return observed, null, p_val


def bootstrap_ci(a, b, n_boot=N_BOOTSTRAP, alpha=ALPHA, rng=None):
    """
    Bootstrap 95% CI on mean(a) - mean(b).
    Returns (observed_diff, lower, upper, boot_diffs).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    observed = a.mean() - b.mean()
    boot = np.array([
        rng.choice(a, len(a), replace=True).mean() -
        rng.choice(b, len(b), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return observed, lo, hi, boot


def mann_whitney(a, b):
    """Mann-Whitney U test (two-tailed)."""
    stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(stat), float(p)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(config_names, diag_a, diag_b,
                 perm_obs, perm_null, perm_p,
                 boot_obs, boot_lo, boot_hi, boot_dists,
                 mw_stat, mw_p,
                 d, out_path):
    """
    4-panel summary figure:
      Top-left   : violin/strip of diagonal distances per config
      Top-right  : permutation null distribution + observed difference
      Bottom-left: bootstrap distribution of mean difference + CI
      Bottom-right: text summary of all test results
    """
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.45})

        col_a, col_b = PALETTE[0], PALETTE[1]
        sig_color    = "#009E73" if perm_p < ALPHA else "#D55E00"

        # ── Panel 1: violin + strip ──────────────────────────────────────────
        ax = axes[0, 0]
        for xi, (data, col, name) in enumerate(
                zip([diag_a, diag_b], [col_a, col_b], config_names)):
            parts = ax.violinplot(data, positions=[xi], widths=0.5,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col); pc.set_alpha(0.5)
            parts["cmedians"].set_color(col); parts["cmedians"].set_lw(1.5)
            jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(data))
            ax.scatter(np.full(len(data), xi) + jitter, data,
                       color=col, s=12, alpha=0.75, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(config_names)
        ax.set_ylabel("Distance to correct target")
        ax.set_title("Diagonal distances per config")
        for xi, data in enumerate([diag_a, diag_b]):
            ax.text(xi, data.max() + 0.02,
                    f"μ={data.mean():.3f}\nσ={data.std():.3f}",
                    ha="center", va="bottom", fontsize=5.5, color="#333333")

        # ── Panel 2: permutation null distribution ───────────────────────────
        ax = axes[0, 1]
        ax.hist(perm_null, bins=60, color="#AAAAAA", alpha=0.7,
                density=True, zorder=2, label="Null distribution")
        ax.axvline(perm_obs, color=sig_color, lw=1.5, zorder=5,
                   label=f"Observed Δ={perm_obs:+.3f}")
        ax.axvline(-abs(perm_obs), color=sig_color, lw=1.5,
                   ls="--", zorder=5, alpha=0.6)
        ax.set_xlabel(f"Δ mean  ({config_names[0]} − {config_names[1]})")
        ax.set_ylabel("Density")
        ax.set_title(f"Permutation test  (p={perm_p:.4f})")
        ax.legend(fontsize=5.5, frameon=True, framealpha=0.9)

        # ── Panel 3: bootstrap CI ────────────────────────────────────────────
        ax = axes[1, 0]
        ax.hist(boot_dists, bins=60, color="#AAAAAA", alpha=0.7,
                density=True, zorder=2, label="Bootstrap Δ")
        ax.axvline(boot_obs, color=sig_color, lw=1.5, zorder=5,
                   label=f"Observed Δ={boot_obs:+.3f}")
        ax.axvspan(boot_lo, boot_hi, alpha=0.18, color=sig_color,
                   label=f"95% CI [{boot_lo:+.3f}, {boot_hi:+.3f}]")
        ax.axvline(0, color="#333333", lw=0.8, ls=":", zorder=3)
        ax.set_xlabel(f"Δ mean  ({config_names[0]} − {config_names[1]})")
        ax.set_ylabel("Density")
        ax.set_title("Bootstrap 95% CI on mean difference")
        ax.legend(fontsize=5.5, frameon=True, framealpha=0.9)

        # ── Panel 4: text summary ────────────────────────────────────────────
        ax = axes[1, 1]
        ax.axis("off")

        def sig_str(p):
            if p < 0.001: return "p < 0.001 ***"
            if p < 0.01:  return "p < 0.01  **"
            if p < 0.05:  return "p < 0.05  *"
            return f"p = {p:.3f}  n.s."

        d_interp = ("negligible" if abs(d) < 0.2 else
                    "small"      if abs(d) < 0.5 else
                    "medium"     if abs(d) < 0.8 else "large")

        ci_sig = "excludes zero → significant" if not (boot_lo <= 0 <= boot_hi) \
                 else "includes zero → not significant"

        lines = [
            ("Statistical Summary", True),
            ("", False),
            (f"Sample sizes:", False),
            (f"  {config_names[0]}: n={len(diag_a)}", False),
            (f"  {config_names[1]}: n={len(diag_b)}", False),
            ("", False),
            (f"Permutation test ({N_PERMUTATIONS:,} perms):", True),
            (f"  Observed Δ = {perm_obs:+.4f}", False),
            (f"  {sig_str(perm_p)}", False),
            ("", False),
            (f"Bootstrap CI ({N_BOOTSTRAP:,} resamples):", True),
            (f"  95% CI [{boot_lo:+.3f}, {boot_hi:+.3f}]", False),
            (f"  {ci_sig}", False),
            ("", False),
            (f"Mann-Whitney U test:", True),
            (f"  U = {mw_stat:.1f}", False),
            (f"  {sig_str(mw_p)}", False),
            ("", False),
            (f"Effect size (Cohen's d):", True),
            (f"  d = {d:+.3f}  ({d_interp})", False),
            ("", False),
            (f"Interpretation:", True),
            (f"  Positive Δ = {config_names[0]} has", False),
            (f"  LARGER distances (worse)", False),
        ]

        y = 0.97
        for text, bold in lines:
            ax.text(0.05, y, text,
                    transform=ax.transAxes,
                    fontsize=6.5 if bold else 6,
                    fontweight="bold" if bold else "normal",
                    color="#111111" if bold else "#333333",
                    va="top")
            y -= 0.055 if bold else 0.048

        fig.suptitle(
            f"Statistical significance: {config_names[0]} vs {config_names[1]}\n"
            f"(diagonal distance = distance from predicted point to correct target)",
            fontsize=8, fontweight="bold")
        fig.tight_layout()
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

    config_names = list(CONFIGS.keys())
    diag_dists   = {}

    for name, base_dir in CONFIGS.items():
        valid_files = [f for f in ALL_SAMPLE_FILES
                       if all((Path(base_dir) / d / f).exists()
                               for d in coor_dirs)]
        if len(valid_files) < 2:
            print(f"[ERROR] {name}: need ≥2 valid files (found {len(valid_files)}).")
            return
        print(f"\n  [{name}] — {len(valid_files)} folds...")
        diag_dists[name] = get_diagonal_distances(
            base_dir, coor_dirs, class_labels,
            class_names, class_targets, valid_files)
        print(f"  [{name}] mean={diag_dists[name].mean():.4f}  "
              f"std={diag_dists[name].std():.4f}  n={len(diag_dists[name])}")

    a = diag_dists[config_names[0]]
    b = diag_dists[config_names[1]]

    print(f"\n  Running statistical tests...")

    perm_obs, perm_null, perm_p = permutation_test(a, b)
    boot_obs, boot_lo, boot_hi, boot_dists = bootstrap_ci(a, b)
    mw_stat, mw_p = mann_whitney(a, b)
    d = cohens_d(a, b)

    print(f"\n{'═'*55}")
    print(f"  Results: {config_names[0]} vs {config_names[1]}")
    print(f"{'═'*55}")
    print(f"  Observed Δ mean  : {perm_obs:+.4f}")
    print(f"  Permutation p    : {perm_p:.4f}")
    print(f"  Bootstrap 95% CI : [{boot_lo:+.3f}, {boot_hi:+.3f}]")
    print(f"  Mann-Whitney p   : {mw_p:.4f}")
    print(f"  Cohen's d        : {d:+.3f}")

    plot_summary(config_names, a, b,
                 perm_obs, perm_null, perm_p,
                 boot_obs, boot_lo, boot_hi, boot_dists,
                 mw_stat, mw_p, d,
                 out_dir / "stats_significance.png")

    print(f"\n  Done.  Output → {out_dir.resolve()}")
    print(f"    stats_significance.png")


if __name__ == "__main__":
    main()