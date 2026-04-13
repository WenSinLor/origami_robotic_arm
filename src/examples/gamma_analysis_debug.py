"""
Story 1 — AMP_WINDOW_END sensitivity analysis
==============================================

Purpose
-------
Test whether the Gamma_shape monotonic mass trend (empty < 20g < 40g < 100g)
is robust to the choice of AMP_WINDOW_END, or whether the result depends
critically on this hyperparameter.

This script sweeps AMP_WINDOW_END over a dense grid from 0.2s to 2.8s and,
for each window value, computes:

  1. The condition means of Gamma_shape for all four mass levels.
  2. Whether the monotonic ordering of means holds exactly
     (empty <= 20g <= 40g <= 100g).
  3. The Kendall tau rank correlation across all (mass_code, Gamma_shape)
     pairs.
  4. The between-condition effect size (Cohen's d, 20g vs 100g pairwise).

It then produces two outputs:

  story1_window_sensitivity.png
    - Top panel: condition means of Gamma_shape vs AMP_WINDOW_END,
      one line per condition. Shows how stable the means are.
    - Bottom panel: Kendall tau vs AMP_WINDOW_END, with a horizontal
      reference at tau = 0. Annotates the fraction of windows where
      monotonicity holds.

  story1_window_sensitivity.csv
    - One row per window value with means, tau, and monotonicity flag.

How to use the output to justify your chosen AMP_WINDOW_END
-----------------------------------------------------------
SCENARIO A — Tau stays positive and means stay ordered across
             a wide range of windows:
    Your result is robust. Report this plot as a supplementary
    sensitivity check and state: "The monotonic trend holds across
    AMP_WINDOW_END in [lo, hi] s, encompassing [fraction]% of the
    tested range." You can then choose AMP_WINDOW_END on principled
    physical grounds (e.g. first half-period of the dominant mode).

SCENARIO B — Tau is positive only for a narrow sub-range:
    You have a real sensitivity problem. Switch to the
    amplitude-free formulation: use the full-window RMS of the
    empty-box mean trajectory as the normaliser (constant across
    trials, computed once from the empty baseline). This removes
    AMP_WINDOW_END entirely.  See the function
    `compute_global_norm_baseline()` provided below.

SCENARIO C — Tau fluctuates around zero regardless of window:
    The shape metric is not reliable. Fall back to the raw
    deviation metric as the primary Story 1 observable and
    note in the paper that amplitude normalisation does not
    qualitatively change the conclusion.

Physical justification for early-window normalisation
------------------------------------------------------
The arm's free-vibration response to an impulsive torque reaches its
peak displacement during the first half-period of the dominant mode.
For a 3 s ringdown the dominant frequency is typically 1-2 Hz, so the
first half-period is 0.25-0.5 s. The RMS amplitude over that window
is the cleanest proxy for the initial energy injected by the impulse.
Choosing AMP_WINDOW_END in [0.5, 1.0] s therefore has a physical basis:
it captures the initial peak response before damping has substantially
reduced the signal.
"""

from pathlib import Path
import csv

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — copy your own paths / sample selections here
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "empty": f"{ROOT_DIR}/soft_state_noload",
    "20g":   f"{ROOT_DIR}/soft_state_20g",
    "40g":   f"{ROOT_DIR}/soft_state_40g",
    "100g":  f"{ROOT_DIR}/soft_state_100g",
}

INCLUDE_SAMPLES = {
    "empty": {
        "coor_0": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "20g": {
        "coor_0": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_1": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_2": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_3": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "40g": {
        "coor_0": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_1": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_2": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_3": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    "100g": {
        "coor_0": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_1": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_2": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_3": [0, 1, 2, 3, 4, 5, 6, 7],
    },
}

T_START        = 0.0
T_END          = 3.0
BASELINE_FRAMES = 1
EXCLUDE_MARKERS = []
EPS            = 1e-12

# Dense sweep grid — 0.2 s to 2.8 s in 0.1 s steps
WINDOW_GRID = np.round(np.arange(0.2, 2.85, 0.1), 2)

# Your original value — will be highlighted on the plot
ORIGINAL_WINDOW = 1.0

OUTPUT_DIR = f"{ROOT_DIR}/soft_state_noload"

NATURE_RC = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
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

COLORS = {
    "empty": "#7A7A7A",
    "20g":   "#56B4E9",
    "40g":   "#009E73",
    "100g":  "#D55E00",
}
COND_ORDER = ["empty", "20g", "40g", "100g"]
MASS_CODES = {"empty": 0, "20g": 20, "40g": 40, "100g": 100}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — identical to story1_analysis.py
# ══════════════════════════════════════════════════════════════════════════════

def load_h5(path: Path, baseline_frames: int = 1):
    with h5py.File(str(path), "r") as f:
        pos = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]

    for n in range(pos.shape[1]):
        for ax in range(pos.shape[2]):
            col = pos[:, n, ax]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                col[:] = 0.0
            elif nan_mask.any():
                idx = np.where(~nan_mask)[0]
                col[:idx[0]] = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i + 1] - idx[i] > 1:
                        col[idx[i] + 1:idx[i + 1]] = col[idx[i]]

    baseline = pos[:min(baseline_frames, pos.shape[0])].mean(axis=0, keepdims=True)
    disp = pos - baseline
    return disp, time


def extract_features(disp, time, t_start, t_end, exclude_markers):
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end, side="right")) if t_end is not None else len(time)
    X  = disp[i0:i1].reshape(i1 - i0, -1)
    ts = time[i0:i1] - time[i0]
    if exclude_markers:
        n_axes  = disp.shape[2]
        n_nodes = X.shape[1] // n_axes
        keep    = [c for n in range(n_nodes) if n not in exclude_markers
                   for c in range(n_axes * n, n_axes * n + n_axes)]
        X = X[:, keep]
    return X, ts


def load_all_trials():
    all_trials = {}
    for cond in COND_ORDER:
        trials = []
        base_dir = Path(DIRS[cond])
        for coor, sample_ids in INCLUDE_SAMPLES[cond].items():
            for sid in sample_ids:
                p = base_dir / coor / f"trajectories_sample_{sid}.h5"
                if not p.exists():
                    print(f"[WARN] Missing: {p}")
                    continue
                disp, time = load_h5(p, baseline_frames=BASELINE_FRAMES)
                X, ts = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
                trials.append({"X": X, "ts": ts, "cond": cond})
        all_trials[cond] = trials
        print(f"  {cond}: {len(trials)} trials")
    return all_trials


# ══════════════════════════════════════════════════════════════════════════════
# METRIC COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_trial_amplitude(X, ts, amp_window_end):
    """RMS of displacement norm over early window."""
    i_end = max(int(np.searchsorted(ts, amp_window_end, side="right")), 1)
    return float(np.sqrt(np.mean(np.sum(X[:i_end] ** 2, axis=1))))


def gamma_shape_for_window(all_trials, amp_window_end):
    """
    Returns dict: condition -> array of Gamma_shape values,
    computed using the given amp_window_end.
    """
    # Step 1: build normalised empty baseline
    empty_norm_trials = []
    for tr in all_trials["empty"]:
        amp = compute_trial_amplitude(tr["X"], tr["ts"], amp_window_end)
        empty_norm_trials.append(tr["X"] / max(amp, EPS))
    empty_mean_norm = np.stack(empty_norm_trials, axis=0).mean(axis=0)

    # Step 2: compute Gamma_shape for every condition
    result = {}
    for cond in COND_ORDER:
        vals = []
        for tr in all_trials[cond]:
            amp = compute_trial_amplitude(tr["X"], tr["ts"], amp_window_end)
            Xn  = tr["X"] / max(amp, EPS)
            delta = Xn - empty_mean_norm
            vals.append(float(np.mean(np.sum(delta ** 2, axis=1))))
        result[cond] = np.array(vals, dtype=float)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ALTERNATIVE: amplitude-free normaliser (removes AMP_WINDOW_END entirely)
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_norm_baseline(all_trials):
    """
    Normaliser that does NOT depend on AMP_WINDOW_END.

    Uses the full-window RMS of the empty-box mean trajectory as the
    global scale. This is a single number, computed once, that does not
    vary from trial to trial and is not sensitive to any window parameter.

    Returns:
        empty_mean_norm : [T, D] — empty baseline normalised by its own RMS
        global_scale    : float  — the single scale factor used
    """
    raw_stack  = np.stack([tr["X"] for tr in all_trials["empty"]], axis=0)
    empty_mean = raw_stack.mean(axis=0)   # [T, D]
    global_scale = float(np.sqrt(np.mean(np.sum(empty_mean ** 2, axis=1))) + EPS)
    return empty_mean / global_scale, global_scale


def gamma_shape_global_norm(all_trials):
    """
    Compute Gamma_shape using the amplitude-free (window-independent) normaliser.
    Each trial is normalised by the same global_scale derived from the empty baseline.
    """
    empty_mean_norm, global_scale = compute_global_norm_baseline(all_trials)
    result = {}
    for cond in COND_ORDER:
        vals = []
        for tr in all_trials[cond]:
            Xn    = tr["X"] / global_scale
            delta = Xn - empty_mean_norm
            vals.append(float(np.mean(np.sum(delta ** 2, axis=1))))
        result[cond] = np.array(vals, dtype=float)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def kendall_tau(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[j] - x[i])
            sy = np.sign(y[j] - y[i])
            p  = sx * sy
            if p > 0: conc += 1
            elif p < 0: disc += 1
    denom = n * (n - 1) / 2
    return (conc - disc) / denom if denom else np.nan


def is_monotone(means_dict):
    """True if empty <= 20g <= 40g <= 100g (means)."""
    vals = [means_dict[c] for c in COND_ORDER]
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def cohens_d(a, b):
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return (np.mean(b) - np.mean(a)) / pooled_sd if pooled_sd > EPS else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_sweep(all_trials):
    rows = []
    means_by_cond = {c: [] for c in COND_ORDER}
    taus   = []
    monot  = []
    d_vals = []

    for w in WINDOW_GRID:
        gs  = gamma_shape_for_window(all_trials, w)
        mns = {c: float(gs[c].mean()) for c in COND_ORDER}

        # Kendall tau over all (mass, gamma_shape) pairs
        all_x, all_y = [], []
        for c in COND_ORDER:
            all_x.extend([MASS_CODES[c]] * len(gs[c]))
            all_y.extend(gs[c].tolist())
        tau = kendall_tau(all_x, all_y)

        mono = is_monotone(mns)
        d    = cohens_d(gs["20g"], gs["100g"])

        for c in COND_ORDER:
            means_by_cond[c].append(mns[c])
        taus.append(tau)
        monot.append(int(mono))
        d_vals.append(d)

        rows.append({
            "amp_window_end":  w,
            "mean_empty":      mns["empty"],
            "mean_20g":        mns["20g"],
            "mean_40g":        mns["40g"],
            "mean_100g":       mns["100g"],
            "kendall_tau":     tau,
            "monotone":        int(mono),
            "cohens_d_20_100": d,
        })

    return rows, means_by_cond, np.array(taus), np.array(monot), np.array(d_vals)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(means_by_cond, taus, monot, out_path):
    frac_mono = monot.mean() * 100

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(
            2, 1, figsize=(5.5, 4.8),
            gridspec_kw={"hspace": 0.45, "height_ratios": [1.6, 1]}
        )

        ax_means, ax_tau = axes

        # ── Top: condition means vs window ──
        for cond in COND_ORDER:
            ax_means.plot(
                WINDOW_GRID, means_by_cond[cond],
                color=COLORS[cond], lw=1.4, label=cond
            )

        ax_means.axvline(
            ORIGINAL_WINDOW, color="#444444", lw=0.9,
            linestyle="--", label=f"Original ({ORIGINAL_WINDOW} s)"
        )
        ax_means.set_xlabel("AMP_WINDOW_END (s)")
        ax_means.set_ylabel(r"Mean $\Gamma_{\mathrm{shape}}$")
        ax_means.set_title(
            r"Condition means of $\Gamma_{\mathrm{shape}}$ vs amplitude window",
            fontweight="bold"
        )
        ax_means.legend(frameon=True, framealpha=0.9, edgecolor="#DDDDDD", ncol=2)

        # ── Bottom: Kendall tau vs window ──
        ax_tau.plot(WINDOW_GRID, taus, color="#333333", lw=1.3)
        ax_tau.axhline(0, color="#AAAAAA", lw=0.7, linestyle=":")
        ax_tau.axvline(
            ORIGINAL_WINDOW, color="#444444", lw=0.9, linestyle="--"
        )
        ax_tau.set_xlabel("AMP_WINDOW_END (s)")
        ax_tau.set_ylabel("Kendall τ")
        ax_tau.set_title(
            f"Kendall τ vs window  —  monotone in {frac_mono:.0f}% of windows",
            fontweight="bold"
        )
        ax_tau.set_ylim(-0.05, 0.55)

        # Shade region where monotonicity holds
        in_mono = False
        lo = None
        for i, (w, m) in enumerate(zip(WINDOW_GRID, monot)):
            if m and not in_mono:
                lo = w
                in_mono = True
            elif not m and in_mono:
                ax_tau.axvspan(lo, WINDOW_GRID[i - 1], color="#009E73", alpha=0.12)
                in_mono = False
        if in_mono:
            ax_tau.axvspan(lo, WINDOW_GRID[-1], color="#009E73", alpha=0.12)

        fig.suptitle(
            "Story 1 — sensitivity of Γ_shape to AMP_WINDOW_END",
            fontsize=8.5, fontweight="bold", y=1.02
        )
        fig.savefig(str(out_path))
        plt.close(fig)
        print(f"[INFO] Monotone ordering holds for {frac_mono:.0f}% of tested windows")


def plot_global_norm_comparison(gamma_shape_window, gamma_shape_global, out_path):
    """
    Side-by-side strip plots: window-based vs global-norm Gamma_shape.
    Lets you see at a glance whether switching removes sensitivity.
    """
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), gridspec_kw={"wspace": 0.45})
        rng = np.random.default_rng(42)

        for ax, gs, title in [
            (axes[0], gamma_shape_window, f"Window-based (w={ORIGINAL_WINDOW} s)"),
            (axes[1], gamma_shape_global,  "Global-norm (window-free)"),
        ]:
            x_pos = np.arange(len(COND_ORDER))
            for i, cond in enumerate(COND_ORDER):
                vals = gs[cond]
                jit  = rng.uniform(-0.1, 0.1, len(vals))
                ax.scatter(
                    x_pos[i] + jit, vals,
                    s=18, color=COLORS[cond], alpha=0.85,
                    edgecolors="white", linewidths=0.35, zorder=3
                )
                m = vals.mean()
                ax.hlines(m, i - 0.18, i + 0.18, colors="black", lw=1.2, zorder=4)

            means = [gs[c].mean() for c in COND_ORDER]
            ax.plot(x_pos, means, color="#666666", lw=1.0, alpha=0.8, zorder=2)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(["Empty", "20 g", "40 g", "100 g"])
            ax.set_ylabel(r"$\Gamma_{\mathrm{shape}}$ (a.u.)")
            ax.set_title(title, fontweight="bold")

        fig.suptitle(
            "Window-based vs window-free normalisation",
            fontsize=8.5, fontweight="bold", y=1.03
        )
        fig.savefig(str(out_path))
        plt.close(fig)


def save_csv(rows, out_path):
    fieldnames = [
        "amp_window_end", "mean_empty", "mean_20g", "mean_40g", "mean_100g",
        "kendall_tau", "monotone", "cohens_d_20_100"
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{v:.8f}" if isinstance(v, float) else v for k, v in r.items()})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 72)
    print("AMP_WINDOW_END sensitivity sweep")
    print("═" * 72)
    print(f"  Window grid: {WINDOW_GRID[0]} s → {WINDOW_GRID[-1]} s "
          f"in 0.1 s steps  ({len(WINDOW_GRID)} values)")
    print(f"  Original AMP_WINDOW_END: {ORIGINAL_WINDOW} s")
    print()

    print("Loading all trials...")
    all_trials = load_all_trials()

    # ── Main sweep ──
    print("\nRunning sweep...")
    rows, means_by_cond, taus, monot, d_vals = run_sweep(all_trials)

    # ── Window-free baseline ──
    print("Computing global-norm (window-free) baseline...")
    gs_global = gamma_shape_global_norm(all_trials)
    gs_window = gamma_shape_for_window(all_trials, ORIGINAL_WINDOW)

    # ── Outputs ──
    plot_sensitivity(
        means_by_cond, taus, monot,
        out_dir / "story1_window_sensitivity.png"
    )
    plot_global_norm_comparison(
        gs_window, gs_global,
        out_dir / "story1_global_norm_comparison.png"
    )
    save_csv(rows, out_dir / "story1_window_sensitivity.csv")

    # ── Console summary ──
    tau_at_original = float(taus[np.argmin(np.abs(WINDOW_GRID - ORIGINAL_WINDOW))])
    mono_at_original = bool(monot[np.argmin(np.abs(WINDOW_GRID - ORIGINAL_WINDOW))])
    frac_mono = monot.mean() * 100
    stable_range = WINDOW_GRID[monot.astype(bool)]

    print("\n" + "─" * 60)
    print("RESULTS SUMMARY")
    print("─" * 60)
    print(f"  Tau at original window ({ORIGINAL_WINDOW} s): {tau_at_original:.4f}")
    print(f"  Monotone at original window:           {mono_at_original}")
    print(f"  Fraction of windows with monotone trend: {frac_mono:.1f}%")
    if len(stable_range) > 0:
        print(f"  Monotone range: [{stable_range[0]:.1f}, {stable_range[-1]:.1f}] s")
    else:
        print("  Monotone range: none — consider switching to global-norm")

    # Global-norm tau for comparison
    all_x_g, all_y_g = [], []
    for c in COND_ORDER:
        all_x_g.extend([MASS_CODES[c]] * len(gs_global[c]))
        all_y_g.extend(gs_global[c].tolist())
    tau_global = kendall_tau(all_x_g, all_y_g)
    mono_global = is_monotone({c: float(gs_global[c].mean()) for c in COND_ORDER})
    print(f"\n  Global-norm tau:    {tau_global:.4f}")
    print(f"  Global-norm monotone: {mono_global}")
    print("─" * 60)

    print("\nSaved outputs:")
    print(f"  {out_dir / 'story1_window_sensitivity.png'}")
    print(f"  {out_dir / 'story1_global_norm_comparison.png'}")
    print(f"  {out_dir / 'story1_window_sensitivity.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()