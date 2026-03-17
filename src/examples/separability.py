"""
Trajectory separability check inspired by Step 6b
=================================================

This script visualizes whether trajectory classes are separated in the
sense of the Step 6b theory:

    Delta_ab > rho_a + rho_b

where
- Delta_ab = distance between class mean trajectories
- rho_a    = within-class radius of class a
- rho_b    = within-class radius of class b

IMPORTANT
---------
This is an empirical finite-sample diagnostic, not a theorem.
It helps visualize whether your measured trajectory classes look
separable in the trajectory-space sense.

Outputs
-------
- separability_pca.png
- separability_mean_distance.png
- separability_margin.png
- separability_within_class_hist.png
- separability_summary.csv
"""

from pathlib import Path
import csv

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "100g": f"{ROOT_DIR}/soft_state_100g",
    # "20g":  f"{ROOT_DIR}/soft_state_20g",
    # "40g":  f"{ROOT_DIR}/soft_state_40g",
    # "near": f"{ROOT_DIR}/soft_state_100g_near",
}

# Example: check angular separability for one condition
INCLUDE_SAMPLES = {
    "100g": {
        "coor_0": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_1": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_2": [0, 1, 2, 3, 4, 5, 6, 7],
        "coor_3": [0, 1, 2, 3, 4, 5, 6, 7],
    },
}

# Time window
T_START = 0.0
T_END = 3.0
BASELINE_FRAMES = 1

# Optional marker exclusion, same style as your other scripts
EXCLUDE_MARKERS = []

# Radius choice:
# "max"     -> rho_z = maximum within-class distance
# "q95"     -> rho_z = 95th percentile within-class distance
# "mean2sd" -> rho_z = mean + 2*std
RADIUS_MODE = "q95"

# Distance normalization:
# if True, divide all trajectory vectors by their own norm before comparison
# This makes the test more shape-based than amplitude-based
NORMALIZE_EACH_TRAJECTORY = False

OUTPUT_DIR = f"{ROOT_DIR}/soft_state_100g/separability_check"

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
    "coor_0": "#0072B2",
    "coor_1": "#E69F00",
    "coor_2": "#009E73",
    "coor_3": "#CC79A7",
}

EPS = 1e-12


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


def flatten_trial(X):
    v = X.reshape(-1).astype(float)
    if NORMALIZE_EACH_TRAJECTORY:
        nrm = np.linalg.norm(v)
        if nrm > EPS:
            v = v / nrm
    return v


def euclidean(a, b):
    return float(np.linalg.norm(a - b))


def compute_radius(dists, mode="q95"):
    dists = np.asarray(dists, dtype=float)
    if len(dists) == 0:
        return np.nan
    if mode == "max":
        return float(np.max(dists))
    if mode == "q95":
        return float(np.quantile(dists, 0.95))
    if mode == "mean2sd":
        return float(np.mean(dists) + 2.0 * np.std(dists))
    raise ValueError(f"Unknown RADIUS_MODE: {mode}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_trials():
    """
    Returns list of dicts with:
    - condition
    - class_label  (here class = coor_x)
    - sample_id
    - X
    - ts
    - vec  (flattened trajectory)
    """
    trials = []

    for cond, coor_map in INCLUDE_SAMPLES.items():
        base_dir = Path(DIRS[cond])
        for coor, sample_ids in coor_map.items():
            for sid in sample_ids:
                p = base_dir / coor / f"trajectories_sample_{sid}.h5"
                if not p.exists():
                    print(f"[WARN] Missing file: {p}")
                    continue

                disp, time = load_h5(p, baseline_frames=BASELINE_FRAMES)
                X, ts = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
                vec = flatten_trial(X)

                trials.append({
                    "condition": cond,
                    "class_label": coor,
                    "sample_id": sid,
                    "path": str(p),
                    "X": X,
                    "ts": ts,
                    "vec": vec,
                })

    if not trials:
        raise RuntimeError("No trials loaded.")
    return trials


def check_compatible_trials(trials):
    T_ref = None
    D_ref = None
    for tr in trials:
        T, D = tr["X"].shape
        if T_ref is None:
            T_ref, D_ref = T, D
        if T != T_ref or D != D_ref:
            raise RuntimeError(
                f"Incompatible trial shape: expected ({T_ref}, {D_ref}), got ({T}, {D}) "
                f"for {tr['class_label']}, sample {tr['sample_id']}"
            )
    return T_ref, D_ref


# ══════════════════════════════════════════════════════════════════════════════
# SEPARABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def group_by_class(trials):
    classes = {}
    for tr in trials:
        classes.setdefault(tr["class_label"], []).append(tr)
    return classes


def compute_class_statistics(class_trials):
    """
    For each class:
    - mean vector mu_z
    - within-class distances to mean
    - radius rho_z
    """
    class_stats = {}
    for label, trs in class_trials.items():
        V = np.stack([tr["vec"] for tr in trs], axis=0)
        mu = V.mean(axis=0)

        dists = np.array([euclidean(v, mu) for v in V], dtype=float)
        rho = compute_radius(dists, mode=RADIUS_MODE)

        class_stats[label] = {
            "mu": mu,
            "V": V,
            "within_dists": dists,
            "rho": rho,
            "n": len(trs),
        }
    return class_stats


def compute_pairwise_matrices(class_stats, class_order):
    K = len(class_order)
    Delta = np.zeros((K, K), dtype=float)
    Margin = np.zeros((K, K), dtype=float)
    Rule = np.zeros((K, K), dtype=float)

    for i, a in enumerate(class_order):
        mu_a = class_stats[a]["mu"]
        rho_a = class_stats[a]["rho"]
        for j, b in enumerate(class_order):
            mu_b = class_stats[b]["mu"]
            rho_b = class_stats[b]["rho"]

            d = euclidean(mu_a, mu_b)
            Delta[i, j] = d
            Margin[i, j] = d - (rho_a + rho_b)
            Rule[i, j] = 1.0 if d > (rho_a + rho_b) else 0.0

    return Delta, Margin, Rule


def summarize_pairs(class_stats, class_order):
    rows = []
    for i, a in enumerate(class_order):
        for j, b in enumerate(class_order):
            if j <= i:
                continue
            d = euclidean(class_stats[a]["mu"], class_stats[b]["mu"])
            rho_a = class_stats[a]["rho"]
            rho_b = class_stats[b]["rho"]
            margin = d - (rho_a + rho_b)
            rows.append({
                "class_a": a,
                "class_b": b,
                "Delta_ab": d,
                "rho_a": rho_a,
                "rho_b": rho_b,
                "rho_sum": rho_a + rho_b,
                "margin": margin,
                "separable_by_rule": int(d > rho_a + rho_b),
            })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca(trials, class_order, out_path):
    V = np.stack([tr["vec"] for tr in trials], axis=0)
    labels = [tr["class_label"] for tr in trials]

    pca = PCA(n_components=2)
    Z = pca.fit_transform(V)

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.0))

        for cls in class_order:
            idx = [i for i, lab in enumerate(labels) if lab == cls]
            ax.scatter(
                Z[idx, 0], Z[idx, 1],
                s=20, alpha=0.85,
                color=COLORS.get(cls, None),
                edgecolors="white", linewidths=0.35,
                label=cls
            )

            # class centroid in PCA coordinates
            zc = Z[idx].mean(axis=0)
            ax.scatter(
                zc[0], zc[1],
                s=60, marker="X",
                color=COLORS.get(cls, None),
                edgecolors="black", linewidths=0.5, zorder=4
            )

        ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
        ax.set_title("Trajectory-cloud PCA")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#DDDDDD")
        fig.savefig(str(out_path))
        plt.close(fig)


def plot_matrix(M, class_order, title, cmap, fmt, out_path, center_zero=False):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.0))

        if center_zero:
            vmax = np.max(np.abs(M))
            vmin = -vmax
        else:
            vmin = None
            vmax = None

        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(class_order)))
        ax.set_yticks(np.arange(len(class_order)))
        ax.set_xticklabels(class_order)
        ax.set_yticklabels(class_order)
        ax.set_title(title)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, format(M[i, j], fmt),
                        ha="center", va="center", fontsize=6,
                        color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)
        fig.savefig(str(out_path))
        plt.close(fig)


def plot_within_class_hist(class_stats, class_order, out_path):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(4.0, 2.9))

        for cls in class_order:
            d = class_stats[cls]["within_dists"]
            rho = class_stats[cls]["rho"]
            ax.hist(
                d, bins=10, alpha=0.35, density=False,
                color=COLORS.get(cls, None), label=f"{cls}"
            )
            ax.axvline(rho, color=COLORS.get(cls, None), lw=1.1, ls="--")

        ax.set_xlabel("Distance to class mean")
        ax.set_ylabel("Count")
        ax.set_title(f"Within-class distances (radius mode: {RADIUS_MODE})")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#DDDDDD")
        fig.savefig(str(out_path))
        plt.close(fig)


def save_summary_csv(class_stats, pair_rows, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["[Class summary]"])
        writer.writerow(["class_label", "n_trials", "rho"])
        for cls, st in class_stats.items():
            writer.writerow([cls, st["n"], f"{st['rho']:.10f}"])

        writer.writerow([])
        writer.writerow(["[Pairwise summary]"])
        writer.writerow([
            "class_a", "class_b", "Delta_ab",
            "rho_a", "rho_b", "rho_sum",
            "margin", "separable_by_rule"
        ])
        for row in pair_rows:
            writer.writerow([
                row["class_a"],
                row["class_b"],
                f"{row['Delta_ab']:.10f}",
                f"{row['rho_a']:.10f}",
                f"{row['rho_b']:.10f}",
                f"{row['rho_sum']:.10f}",
                f"{row['margin']:.10f}",
                row["separable_by_rule"],
            ])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 76)
    print("Trajectory separability check inspired by Step 6b")
    print("═" * 76)

    trials = load_trials()
    T_ref, D_ref = check_compatible_trials(trials)
    print(f"\nLoaded {len(trials)} trials with common shape: T={T_ref}, D={D_ref}")

    class_trials = group_by_class(trials)
    class_order = sorted(class_trials.keys())
    print(f"Classes: {class_order}")

    class_stats = compute_class_statistics(class_trials)
    Delta, Margin, Rule = compute_pairwise_matrices(class_stats, class_order)
    pair_rows = summarize_pairs(class_stats, class_order)

    print("\nClass radii:")
    for cls in class_order:
        print(f"  {cls}: rho = {class_stats[cls]['rho']:.6f}  (n={class_stats[cls]['n']})")

    print("\nPairwise margins Delta_ab - (rho_a + rho_b):")
    for row in pair_rows:
        print(
            f"  {row['class_a']} vs {row['class_b']}: "
            f"Delta={row['Delta_ab']:.6f}, "
            f"rho_sum={row['rho_sum']:.6f}, "
            f"margin={row['margin']:.6f}, "
            f"separable={bool(row['separable_by_rule'])}"
        )

    plot_pca(trials, class_order, out_dir / "separability_pca.png")
    plot_matrix(
        Delta, class_order,
        title="Between-class mean distance $\\Delta_{ab}$",
        cmap="Blues", fmt=".2f",
        out_path=out_dir / "separability_mean_distance.png",
        center_zero=False
    )
    plot_matrix(
        Margin, class_order,
        title="Margin $\\Delta_{ab}-(\\rho_a+\\rho_b)$",
        cmap="RdBu_r", fmt=".2f",
        out_path=out_dir / "separability_margin.png",
        center_zero=True
    )
    plot_within_class_hist(class_stats, class_order, out_dir / "separability_within_class_hist.png")
    save_summary_csv(class_stats, pair_rows, out_dir / "separability_summary.csv")

    print("\nSaved outputs:")
    print(f"  {out_dir / 'separability_pca.png'}")
    print(f"  {out_dir / 'separability_mean_distance.png'}")
    print(f"  {out_dir / 'separability_margin.png'}")
    print(f"  {out_dir / 'separability_within_class_hist.png'}")
    print(f"  {out_dir / 'separability_summary.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()