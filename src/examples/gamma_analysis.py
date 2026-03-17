"""
Story 1 — Revised mechanism test
================================

This script computes two Story 1 metrics relative to the empty-box baseline:

1. Raw deviation metric:
       Gamma_raw
   Measures total deviation from the empty-box mean trajectory.

2. Amplitude-normalized shape metric:
       Gamma_shape
   Measures deviation from the normalized empty-box mean trajectory after
   removing gross trial-wise amplitude scaling.

The second metric is the recommended primary Story 1 metric because it is
less confounded by load-dependent amplitude changes.

Outputs
-------
- story1_raw_vs_shape.png
- story1_shape_curves.png
- story1_summary.csv
"""

from pathlib import Path
import csv

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "empty": f"{ROOT_DIR}/soft_state_noload",   # change if needed
    "20g":   f"{ROOT_DIR}/soft_state_20g",
    "40g":   f"{ROOT_DIR}/soft_state_40g",
    "100g":  f"{ROOT_DIR}/soft_state_100g",
}

# Manual sample selection
# empty only uses coor_0 as bookkeeping storage
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

T_START = 0.0
T_END = 3.0
BASELINE_FRAMES = 1
EXCLUDE_MARKERS = []

# Early window used to define amplitude scale
AMP_WINDOW_END = 1.0
EPS = 1e-12

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


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
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

    X = disp[i0:i1].reshape(i1 - i0, -1)
    ts = time[i0:i1] - time[i0]

    if exclude_markers:
        n_axes = disp.shape[2]
        n_nodes = X.shape[1] // n_axes
        keep = [c for n in range(n_nodes) if n not in exclude_markers
                for c in range(n_axes * n, n_axes * n + n_axes)]
        X = X[:, keep]

    return X, ts


def bootstrap_mean_ci(values, n_boot=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])

    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[b] = sample.mean()

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def kendall_tau_simple(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            sx = np.sign(x[j] - x[i])
            sy = np.sign(y[j] - y[i])
            prod = sx * sy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    denom = n * (n - 1) / 2
    if denom == 0:
        return np.nan
    return (concordant - discordant) / denom


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_condition_trials(dir_key: str):
    trials = []
    base_dir = Path(DIRS[dir_key])

    for coor, sample_ids in INCLUDE_SAMPLES[dir_key].items():
        for sid in sample_ids:
            p = base_dir / coor / f"trajectories_sample_{sid}.h5"
            if not p.exists():
                print(f"[WARN] Missing file: {p}")
                continue

            disp, time = load_h5(p, baseline_frames=BASELINE_FRAMES)
            X, ts = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)

            trials.append({
                "condition": dir_key,
                "coor": coor,
                "sample_id": sid,
                "sample_name": f"trajectories_sample_{sid}",
                "X": X,
                "ts": ts,
            })

    if not trials:
        raise RuntimeError(f"No trials loaded for '{dir_key}'")
    return trials


def check_compatible_trials(trials_by_condition):
    T_ref = None
    D_ref = None
    ts_ref = None

    for cond, trials in trials_by_condition.items():
        for tr in trials:
            T, D = tr["X"].shape
            if T_ref is None:
                T_ref, D_ref, ts_ref = T, D, tr["ts"]
            if T != T_ref or D != D_ref:
                raise RuntimeError(
                    f"Incompatible shape: expected ({T_ref}, {D_ref}), "
                    f"got ({T}, {D}) in {cond}/{tr['coor']}/{tr['sample_name']}"
                )
    return T_ref, D_ref, ts_ref


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_trial_amplitude(X, ts, amp_window_end):
    """RMS amplitude over early-time window."""
    i_end = int(np.searchsorted(ts, amp_window_end, side="right"))
    i_end = max(i_end, 1)
    return float(np.sqrt(np.mean(np.sum(X[:i_end] ** 2, axis=1))))


def normalize_trial(X, amp):
    return X / max(amp, EPS)


def compute_empty_baselines(empty_trials, ts_ref):
    """
    Returns:
    - empty_mean_raw: [T, D]
    - empty_mean_norm: [T, D]
    """
    raw_stack = np.stack([tr["X"] for tr in empty_trials], axis=0)

    norm_trials = []
    for tr in empty_trials:
        amp = compute_trial_amplitude(tr["X"], tr["ts"], AMP_WINDOW_END)
        norm_trials.append(normalize_trial(tr["X"], amp))
    norm_stack = np.stack(norm_trials, axis=0)

    return raw_stack.mean(axis=0), norm_stack.mean(axis=0)


def compute_metrics_for_trials(trials, empty_mean_raw, empty_mean_norm):
    """
    For each trial compute:
    - Gamma_raw
    - Gamma_shape
    - raw deviation curve
    - normalized deviation curve
    """
    gamma_raw_vals = []
    gamma_shape_vals = []
    raw_curves = []
    norm_curves = []
    meta = []

    for tr in trials:
        X = tr["X"]
        ts = tr["ts"]

        # raw
        delta_raw = X - empty_mean_raw
        sq_norm_raw_t = np.sum(delta_raw ** 2, axis=1)
        gamma_raw = float(np.mean(sq_norm_raw_t))

        # normalized shape
        amp = compute_trial_amplitude(X, ts, AMP_WINDOW_END)
        Xn = normalize_trial(X, amp)
        delta_norm = Xn - empty_mean_norm
        sq_norm_norm_t = np.sum(delta_norm ** 2, axis=1)
        gamma_shape = float(np.mean(sq_norm_norm_t))

        gamma_raw_vals.append(gamma_raw)
        gamma_shape_vals.append(gamma_shape)
        raw_curves.append(np.sqrt(sq_norm_raw_t))
        norm_curves.append(np.sqrt(sq_norm_norm_t))

        meta.append({
            "condition": tr["condition"],
            "coor": tr["coor"],
            "sample_id": tr["sample_id"],
            "sample_name": tr["sample_name"],
            "amp_scale": amp,
        })

    return (
        np.array(gamma_raw_vals, dtype=float),
        np.array(gamma_shape_vals, dtype=float),
        raw_curves,
        norm_curves,
        meta,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_panel(ax, metric_dict, ylabel, title):
    cond_order = ["empty", "20g", "40g", "100g"]
    x_positions = np.arange(len(cond_order))
    rng = np.random.default_rng(0)

    means = []
    all_x = []
    all_y = []

    for i, cond in enumerate(cond_order):
        vals = np.asarray(metric_dict[cond], dtype=float)
        means.append(vals.mean())

        mass_code = {"empty": 0, "20g": 20, "40g": 40, "100g": 100}[cond]
        all_x.extend([mass_code] * len(vals))
        all_y.extend(vals.tolist())

        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            np.full(len(vals), x_positions[i], dtype=float) + jitter,
            vals,
            s=18,
            color=COLORS[cond],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

        mean_v = vals.mean()
        lo, hi = bootstrap_mean_ci(vals, n_boot=3000, alpha=0.05, seed=i)

        ax.hlines(mean_v, i - 0.18, i + 0.18, colors="black", lw=1.2, zorder=4)
        ax.vlines(i, lo, hi, colors="black", lw=0.9, zorder=4)
        ax.hlines([lo, hi], i - 0.07, i + 0.07, colors="black", lw=0.9, zorder=4)

    ax.plot(x_positions, means, color="#666666", lw=1.1, alpha=0.9, zorder=2)
    tau = kendall_tau_simple(np.array(all_x), np.array(all_y))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Empty", "20 g", "40 g", "100 g"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.text(
        0.98, 0.04,
        f"Kendall tau = {tau:.3f}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=6.2,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", lw=0.5, alpha=0.95)
    )


def plot_raw_vs_shape(gamma_raw_dict, gamma_shape_dict, out_path):
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.45})

        plot_metric_panel(
            axes[0],
            gamma_raw_dict,
            ylabel=r"Raw deviation $\Gamma_{\mathrm{raw}}$ (a.u.)",
            title="Raw deviation from empty baseline",
        )

        plot_metric_panel(
            axes[1],
            gamma_shape_dict,
            ylabel=r"Normalized shape deviation $\Gamma_{\mathrm{shape}}$ (a.u.)",
            title="Amplitude-normalized deviation",
        )

        fig.suptitle(
            "Story 1 — payload changes the ringdown relative to the empty-box baseline",
            fontsize=8.5,
            fontweight="bold",
            y=1.03,
        )
        fig.savefig(str(out_path))
        plt.close(fig)


def plot_shape_curves(norm_curve_dict, ts, out_path):
    cond_order = ["empty", "20g", "40g", "100g"]

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))

        for cond in cond_order:
            curves = np.stack(norm_curve_dict[cond], axis=0)
            mu = curves.mean(axis=0)
            sd = curves.std(axis=0)

            ax.plot(ts, mu, color=COLORS[cond], lw=1.3, label=cond)
            ax.fill_between(ts, mu - sd, mu + sd, color=COLORS[cond], alpha=0.18, linewidth=0)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"$\|\tilde{x}(t)-\bar{\tilde{x}}_{\mathrm{empty}}(t)\|$")
        ax.set_title("Amplitude-normalized deviation over time", fontweight="bold")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#DDDDDD")

        fig.savefig(str(out_path))
        plt.close(fig)


def save_summary_csv(gamma_raw_dict, gamma_shape_dict, meta_dict, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "coor", "sample_id", "sample_name",
            "amp_scale", "gamma_raw", "gamma_shape"
        ])

        for cond in ["empty", "20g", "40g", "100g"]:
            for meta, g_raw, g_shape in zip(meta_dict[cond], gamma_raw_dict[cond], gamma_shape_dict[cond]):
                writer.writerow([
                    cond,
                    meta["coor"],
                    meta["sample_id"],
                    meta["sample_name"],
                    f"{meta['amp_scale']:.10f}",
                    f"{g_raw:.10f}",
                    f"{g_shape:.10f}",
                ])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 76)
    print("Story 1 — revised mechanism test (raw + amplitude-normalized metrics)")
    print("═" * 76)

    trials_by_condition = {}
    for cond in ["empty", "20g", "40g", "100g"]:
        print(f"\nLoading condition: {cond}")
        trials = load_condition_trials(cond)
        trials_by_condition[cond] = trials
        print(f"  Loaded {len(trials)} trials")
        for coor, ids in INCLUDE_SAMPLES[cond].items():
            print(f"    {coor}: {ids}")

    T_ref, D_ref, ts_ref = check_compatible_trials(trials_by_condition)
    print(f"\nCommon shape across selected trials: T={T_ref}, D={D_ref}")

    empty_mean_raw, empty_mean_norm = compute_empty_baselines(trials_by_condition["empty"], ts_ref)

    gamma_raw_dict = {}
    gamma_shape_dict = {}
    raw_curve_dict = {}
    norm_curve_dict = {}
    meta_dict = {}

    for cond, trials in trials_by_condition.items():
        gamma_raw, gamma_shape, raw_curves, norm_curves, meta = compute_metrics_for_trials(
            trials, empty_mean_raw, empty_mean_norm
        )

        gamma_raw_dict[cond] = gamma_raw
        gamma_shape_dict[cond] = gamma_shape
        raw_curve_dict[cond] = raw_curves
        norm_curve_dict[cond] = norm_curves
        meta_dict[cond] = meta

        print(f"\nCondition: {cond}")
        print(f"  n_trials = {len(gamma_raw)}")
        print(f"  mean Gamma_raw   = {gamma_raw.mean():.6f}")
        print(f"  mean Gamma_shape = {gamma_shape.mean():.6f}")

    plot_raw_vs_shape(gamma_raw_dict, gamma_shape_dict, out_dir / "story1_raw_vs_shape.png")
    plot_shape_curves(norm_curve_dict, ts_ref, out_dir / "story1_shape_curves.png")
    save_summary_csv(gamma_raw_dict, gamma_shape_dict, meta_dict, out_dir / "story1_summary.csv")

    print("\nSaved outputs:")
    print(f"  {out_dir / 'story1_raw_vs_shape.png'}")
    print(f"  {out_dir / 'story1_shape_curves.png'}")
    print(f"  {out_dir / 'story1_summary.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()