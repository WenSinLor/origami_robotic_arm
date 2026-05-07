"""
COM Demo — Dynamic-Summary 2D Ridge Regressor
=============================================
This version rewrites the frame-level COM demo into a trial-level
dynamic-summary regressor.

Why:
- keeps physical meaning by using time-aware summaries of the dynamics
- avoids the frame-level / trial-level mismatch
- predicts one 2D point per trial directly
- decodes by nearest offset target

Target geometry:
    -3 -> [-3, 0]
    -2 -> [-2, 0]
    +2 -> [+2, 0]
    +3 -> [+3, 0]
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, classification_report
from scipy.io import savemat


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "base":      f"{ROOT_DIR}/soft_state_100g_bending_sensor",
    "near":      f"{ROOT_DIR}/soft_state_100g_near_bending_sensor",
    "nearest":   f"{ROOT_DIR}/soft_state_100g_nearest_bending_sensor",
    "bar_base":  f"{ROOT_DIR}/com_demo_60g",
    "bar_near":  f"{ROOT_DIR}/com_demo_60g_near",
}

# Sign convention:
# coor_0 = positive side
# coor_2 = negative side
OFFSET_BY_DIR_AND_CLASS = {
    "base": {
        "coor_0": +3,
        "coor_2": -3,
    },
    "bar_base": {
        "coor_0": +3,
        "coor_2": -3,
    },
    "near": {
        "coor_0": +2,
        "coor_2": -2,
    },
    "bar_near": {
        "coor_0": +2,
        "coor_2": -2,
    },
}

ACTIVE_CLASSES = ["coor_0", "coor_2"]
COOR_DIRS = ["coor_0", "coor_2"]

OFFSET_TARGETS = {
    -3: np.array([-3.0, 0.0]),
    -2: np.array([-2.0, 0.0]),
    +2: np.array([+2.0, 0.0]),
    +3: np.array([+3.0, 0.0]),
}
VALID_OFFSETS = np.array([-3, -2, 2, 3], dtype=int)

TRAIN_SAMPLES = {
    "coor_0": [
        ("bar_base", "trajectories_sample_0.h5"),
        ("bar_base", "trajectories_sample_1.h5"),
        ("bar_base", "trajectories_sample_2.h5"),
        ("bar_base", "trajectories_sample_3.h5"),
        ("bar_base", "trajectories_sample_4.h5"),
        ("bar_base", "trajectories_sample_5.h5"),
        ("bar_base", "trajectories_sample_6.h5"),
        ("bar_base", "trajectories_sample_7.h5"),
        ("bar_base", "trajectories_sample_8.h5"),
        ("bar_base", "trajectories_sample_9.h5"),
        ("bar_base", "trajectories_sample_10.h5"),
        ("bar_base", "trajectories_sample_11.h5"),
        # ("bar_base", "trajectories_sample_12.h5"),
        # ("bar_base", "trajectories_sample_13.h5"),
    ],
    "coor_2": [
        ("bar_base", "trajectories_sample_0.h5"),
        ("bar_base", "trajectories_sample_1.h5"),
        ("bar_base", "trajectories_sample_2.h5"),
        ("bar_base", "trajectories_sample_3.h5"),
        ("bar_base", "trajectories_sample_4.h5"),
        ("bar_base", "trajectories_sample_5.h5"),
        ("bar_base", "trajectories_sample_6.h5"),
        ("bar_base", "trajectories_sample_7.h5"),
        ("bar_base", "trajectories_sample_8.h5"),
        ("bar_base", "trajectories_sample_9.h5"),
        ("bar_base", "trajectories_sample_10.h5"),
        ("bar_base", "trajectories_sample_11.h5"),
        # ("bar_base", "trajectories_sample_12.h5"),
        # ("bar_base", "trajectories_sample_13.h5"),
    ],
}

TEST_SAMPLES = {
    "coor_0": [
        # ("bar_base", "trajectories_sample_8.h5"),
        # ("bar_base", "trajectories_sample_9.h5"),
        # ("bar_base", "trajectories_sample_10.h5"),
        # ("bar_base", "trajectories_sample_11.h5"),
        # ("bar_base", "trajectories_sample_12.h5"),
        # ("bar_base", "trajectories_sample_13.h5"),
        # ("bar_base", "trajectories_sample_14.h5"),
        # ("bar_base", "trajectories_sample_15.h5"),
        # ("bar_base", "trajectories_sample_16.h5"),
        # ("bar_base", "trajectories_sample_17.h5"),
        # ("bar_base", "trajectories_sample_18.h5"),
        # ("bar_base", "trajectories_sample_19.h5"),

        # ("bar_outer", "trajectories_sample_0.h5"),
        # ("bar_outer", "trajectories_sample_1.h5"),
        # ("bar_outer", "trajectories_sample_2.h5"),
        # ("bar_outer", "trajectories_sample_3.h5"),
        # ("bar_outer", "trajectories_sample_4.h5"),
        # ("bar_outer", "trajectories_sample_5.h5"),
        # ("bar_outer", "trajectories_sample_6.h5"),
        # ("bar_outer", "trajectories_sample_7.h5"),
        # ("bar_outer", "trajectories_sample_8.h5"),
        # ("bar_outer", "trajectories_sample_9.h5"),
        # ("bar_outer", "trajectories_sample_10.h5"),
        # ("bar_outer", "trajectories_sample_11.h5"),
        # ("bar_outer", "trajectories_sample_12.h5"),
        # ("bar_outer", "trajectories_sample_13.h5"),
        # ("bar_outer", "trajectories_sample_14.h5"),
        # ("bar_outer", "trajectories_sample_15.h5"),
        # ("bar_outer", "trajectories_sample_16.h5"),
        # ("bar_outer", "trajectories_sample_17.h5"),
        # ("bar_outer", "trajectories_sample_18.h5"),
        # ("bar_outer", "trajectories_sample_19.h5"),

        ("bar_near", "trajectories_sample_0.h5"),
        ("bar_near", "trajectories_sample_1.h5"),
        ("bar_near", "trajectories_sample_2.h5"),
        ("bar_near", "trajectories_sample_3.h5"),
        ("bar_near", "trajectories_sample_4.h5"),
        ("bar_near", "trajectories_sample_5.h5"),
        ("bar_near", "trajectories_sample_6.h5"),
        ("bar_near", "trajectories_sample_7.h5"),
        ("bar_near", "trajectories_sample_8.h5"),
        ("bar_near", "trajectories_sample_9.h5"),
        ("bar_near", "trajectories_sample_10.h5"),
        ("bar_near", "trajectories_sample_11.h5"),
        ("bar_near", "trajectories_sample_12.h5"),
        ("bar_near", "trajectories_sample_13.h5"),
        ("bar_near", "trajectories_sample_14.h5"),
        ("bar_near", "trajectories_sample_15.h5"),
        ("bar_near", "trajectories_sample_16.h5"),
        ("bar_near", "trajectories_sample_17.h5"),
        ("bar_near", "trajectories_sample_18.h5"),
        ("bar_near", "trajectories_sample_19.h5"),
    ],
    "coor_2": [
        # ("bar_base", "trajectories_sample_12.h5"),
        # ("bar_base", "trajectories_sample_13.h5"),
        # ("bar_base", "trajectories_sample_14.h5"),
        # ("bar_base", "trajectories_sample_15.h5"),
        # ("bar_base", "trajectories_sample_16.h5"),
        # ("bar_base", "trajectories_sample_17.h5"),
        # ("bar_base", "trajectories_sample_18.h5"),
        # ("bar_base", "trajectories_sample_19.h5"),

        # ("bar_outer", "trajectories_sample_0.h5"),
        # ("bar_outer", "trajectories_sample_1.h5"),
        # ("bar_outer", "trajectories_sample_2.h5"),
        # ("bar_outer", "trajectories_sample_3.h5"),
        # ("bar_outer", "trajectories_sample_4.h5"),
        # ("bar_outer", "trajectories_sample_5.h5"),
        # ("bar_outer", "trajectories_sample_6.h5"),
        # ("bar_outer", "trajectories_sample_7.h5"),
        # ("bar_outer", "trajectories_sample_8.h5"),
        # ("bar_outer", "trajectories_sample_9.h5"),
        # ("bar_outer", "trajectories_sample_10.h5"),
        # ("bar_outer", "trajectories_sample_11.h5"),
        # ("bar_outer", "trajectories_sample_12.h5"),
        # ("bar_outer", "trajectories_sample_13.h5"),
        # ("bar_outer", "trajectories_sample_14.h5"),
        # ("bar_outer", "trajectories_sample_15.h5"),
        # ("bar_outer", "trajectories_sample_16.h5"),
        # ("bar_outer", "trajectories_sample_17.h5"),
        # ("bar_outer", "trajectories_sample_18.h5"),
        # ("bar_outer", "trajectories_sample_19.h5"),

        ("bar_near", "trajectories_sample_0.h5"),
        ("bar_near", "trajectories_sample_1.h5"),
        ("bar_near", "trajectories_sample_2.h5"),
        ("bar_near", "trajectories_sample_3.h5"),
        ("bar_near", "trajectories_sample_4.h5"),
        ("bar_near", "trajectories_sample_5.h5"),
        ("bar_near", "trajectories_sample_6.h5"),
        ("bar_near", "trajectories_sample_7.h5"),
        ("bar_near", "trajectories_sample_8.h5"),
        ("bar_near", "trajectories_sample_9.h5"),
        ("bar_near", "trajectories_sample_10.h5"),
        ("bar_near", "trajectories_sample_11.h5"),
        ("bar_near", "trajectories_sample_12.h5"),
        ("bar_near", "trajectories_sample_13.h5"),
        ("bar_near", "trajectories_sample_14.h5"),
        ("bar_near", "trajectories_sample_15.h5"),
        ("bar_near", "trajectories_sample_16.h5"),
        ("bar_near", "trajectories_sample_17.h5"),
        ("bar_near", "trajectories_sample_18.h5"),
        ("bar_near", "trajectories_sample_19.h5"),
    ],
}

T_START = 0.0
T_END   = 3.0

EXCLUDE_MARKERS = []
NORMALIZE_ENERGY = False

# Dynamic-summary settings
EARLY_FRAC = 0.35
LATE_FRAC  = 0.35

# Readout settings
RIDGE_ALPHA = 1.0
USE_PCA = False
PCA_VARIANCE = 0.95
USE_COSINE_DECODE = False

OUTPUT_DIR = f"{ROOT_DIR}/com_demo_60g_near"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_label(dir_key, fname):
    return f"{dir_key}/{fname.replace('trajectories_','').replace('.h5','')}"

def resolve(dir_key, coor_dir, fname):
    return Path(DIRS[dir_key]) / coor_dir / fname

def get_offset(dir_key, coor_name):
    return int(OFFSET_BY_DIR_AND_CLASS[dir_key][coor_name])

def get_target_xy(dir_key, coor_name):
    off = get_offset(dir_key, coor_name)
    return OFFSET_TARGETS[int(off)].astype(float)

def nearest_offset_euclidean(p_xy):
    keys = list(OFFSET_TARGETS.keys())
    tgts = np.vstack([OFFSET_TARGETS[k] for k in keys])
    idx = np.argmin(np.linalg.norm(tgts - p_xy[None, :], axis=1))
    return int(keys[idx])

def nearest_offset_cosine(p_xy):
    keys = list(OFFSET_TARGETS.keys())
    tgts = np.vstack([OFFSET_TARGETS[k] for k in keys])

    p = np.asarray(p_xy, dtype=float)
    pn = np.linalg.norm(p)
    if pn < 1e-12:
        return int(keys[0])

    sims = []
    for t in tgts:
        tn = np.linalg.norm(t)
        sims.append(np.dot(p, t) / (pn * tn))
    return int(keys[np.argmax(sims)])

def decode_offset(p_xy):
    if USE_COSINE_DECODE:
        return nearest_offset_cosine(p_xy)
    return nearest_offset_euclidean(p_xy)

def summarise_samples(samples_dict):
    for cname, slist in samples_dict.items():
        counts = {}
        for dk, _ in slist:
            counts[dk] = counts.get(dk, 0) + 1
        parts = [f"{counts[k]}×{k}" for k in sorted(counts.keys())]
        if not parts:
            parts = ["0"]
        print(f"    {cname:8s}: {' + '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_h5(path: Path, baseline_frames: int = 1):
    with h5py.File(str(path), "r") as f:
        pos = f["time_series/nodes/positions"][:]
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
                col[:idx[0]] = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i + 1] - idx[i] > 1:
                        col[idx[i] + 1:idx[i + 1]] = col[idx[i]]

    baseline = pos[:min(baseline_frames, pos.shape[0])].mean(axis=0, keepdims=True)
    return pos - baseline, time

def extract_features(disp, time, t_start, t_end, exclude_markers):
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end, side="right")) if t_end is not None else len(time)

    X = disp[i0:i1].reshape(i1 - i0, -1).astype(np.float64)
    ts = time[i0:i1] - time[i0]

    if exclude_markers:
        coord_dim = disp.shape[2]
        n_markers = disp.shape[1]
        keep = [
            coord_dim * n + d
            for n in range(n_markers)
            if n not in exclude_markers
            for d in range(coord_dim)
        ]
        X = X[:, keep]

    return X, ts


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC SUMMARY FEATURE
# ══════════════════════════════════════════════════════════════════════════════

def build_dynamic_summary_feature(X):
    """
    X shape: (T, D)

    Summary blocks:
    - whole mean
    - whole std
    - early mean
    - late mean
    - early std
    - late std
    - late - early mean
    - rms
    """
    T, D = X.shape

    n_early = max(1, int(np.floor(EARLY_FRAC * T)))
    n_late = max(1, int(np.floor(LATE_FRAC * T)))

    X_early = X[:n_early]
    X_late = X[-n_late:]

    full_mean = X.mean(axis=0)
    full_std = X.std(axis=0)

    early_mean = X_early.mean(axis=0)
    late_mean = X_late.mean(axis=0)

    early_std = X_early.std(axis=0)
    late_std = X_late.std(axis=0)

    delta_mean = late_mean - early_mean
    rms = np.sqrt(np.mean(X**2, axis=0))

    feat = np.concatenate([
        full_mean,
        full_std,
        early_mean,
        late_mean,
        early_std,
        late_std,
        delta_mean,
        rms,
    ], axis=0)

    if NORMALIZE_ENERGY:
        denom = np.linalg.norm(feat)
        if denom > 1e-12:
            feat = feat / denom

    return feat.astype(np.float64)

def extract_dynamic_summary_feature(disp, time, t_start, t_end, exclude_markers):
    X, ts = extract_features(disp, time, t_start, t_end, exclude_markers)
    feat = build_dynamic_summary_feature(X)
    return feat, ts


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC-SUMMARY RIDGE REGRESSOR
# ══════════════════════════════════════════════════════════════════════════════

def fit_dynamic_summary_offset(train_samples_dict, ridge_alpha=1.0,
                               use_pca=False, pca_variance=0.95):
    X_rows, Y_rows = [], []
    N = D = None
    summary_dim = None

    print("=" * 72)
    print("  Building dynamic-summary training matrix")
    print("=" * 72)

    for coor_name in ACTIVE_CLASSES:
        sample_list = train_samples_dict.get(coor_name, [])
        if not sample_list:
            print(f"  [WARN] No training samples defined for {coor_name} — skipped")
            continue

        for dir_key, fname in sample_list:
            p = resolve(dir_key, coor_name, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} — skipped")
                continue

            disp, time = load_h5(p)
            X_raw, _ = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            feat, ts = extract_dynamic_summary_feature(
                disp, time, T_START, T_END, EXCLUDE_MARKERS
            )
            tgt = get_target_xy(dir_key, coor_name)

            if D is None:
                coord_dim = disp.shape[2]
                D = X_raw.shape[1]
                N = D // coord_dim
                summary_dim = feat.shape[0]

            X_rows.append(feat)
            Y_rows.append(tgt)

            print(
                f"  + [{dir_key}] {coor_name}/{fname}  "
                f"offset={get_offset(dir_key, coor_name):+d}  "
                f"raw_dim={X_raw.shape[1]}  summary_dim={feat.shape[0]}"
            )

    if not X_rows:
        raise RuntimeError("No training data found. Check TRAIN_SAMPLES / DIRS.")

    X_train = np.vstack(X_rows)
    Y_train = np.vstack(Y_rows)

    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    Xn = (X_train - mu) / std

    pca = None
    Z = Xn
    if use_pca:
        pca_full = PCA().fit(Xn)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        k = min(
            int(np.searchsorted(cumvar, pca_variance)) + 1,
            Xn.shape[1],
            max(1, Xn.shape[0] - 1)
        )
        pca = PCA(n_components=k).fit(Xn)
        Z = pca.transform(Xn)
        print(f"  PCA used         : {k} components ({pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(Z, Y_train)

    print(f"\n  Raw state dim      : {D}")
    print(f"  Summary feature dim: {summary_dim}")
    print(f"  Train matrix       : {Z.shape} -> {Y_train.shape}")
    print(f"  Ridge alpha        : {ridge_alpha}")
    print(f"  Normalize energy   : {NORMALIZE_ENERGY}")
    print(f"  Use PCA            : {use_pca}")

    return model, pca, mu, std, N, D, summary_dim

def run_dynamic_summary_offset(samples_dict, model, pca, mu, std, label=""):
    results = []
    y_true, y_pred = [], []

    print(f"\n{'=' * 72}")
    print(f"  Inference — {label}")
    print(f"{'=' * 72}")

    for coor_name in ACTIVE_CLASSES:
        sample_list = samples_dict.get(coor_name, [])
        if not sample_list:
            print(f"  [WARN] No samples defined for {coor_name} — skipped")
            continue

        for dir_key, fname in sample_list:
            p = resolve(dir_key, coor_name, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} — skipped")
                continue

            disp, time = load_h5(p)
            feat, ts = extract_dynamic_summary_feature(
                disp, time, T_START, T_END, EXCLUDE_MARKERS
            )

            x = feat[None, :]
            xn = (x - mu) / std
            z = pca.transform(xn) if pca is not None else xn
            mean_xy = model.predict(z)[0]

            true_offset = get_offset(dir_key, coor_name)
            pred_offset = decode_offset(mean_xy)
            conf = float(np.linalg.norm(mean_xy))

            tick = "✓" if pred_offset == true_offset else "✗"
            slabel = sample_label(dir_key, fname)

            print(
                f"  {tick}  {coor_name}/{slabel}  "
                f"true={true_offset:+d}  pred={pred_offset:+d}  "
                f"mean=({mean_xy[0]:+.3f}, {mean_xy[1]:+.3f})  |mean|={conf:.3f}"
            )

            results.append(dict(
                true_offset=true_offset,
                coor_name=coor_name,
                sample=slabel,
                dir_key=dir_key,
                ts=ts,
                mean_xy=mean_xy,
                pred_offset=pred_offset,
                target_xy=OFFSET_TARGETS[int(true_offset)].astype(float),
                conf=conf,
            ))

            y_true.append(true_offset)
            y_pred.append(pred_offset)

    return results, np.array(y_true), np.array(y_pred)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
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

DIR_MARKERS = {
    "base": "o",
    "near": "s",
    "bar_base": "o",
    "bar_near": "s",
    "bar_outer": "^",
}

DIR_BG = {
    "base": "#EEF4FB",
    "near": "#FFF8EE",
    "bar_base": "#EEF4FB",
    "bar_near": "#FFF8EE",
    "bar_outer": "#F5EEFF",
}

def plot_trial_points_offset(results, train_acc, test_acc, out_path):
    offset_colors = {
        -4: "#17becf",
        -3: "#1f77b4",
        -2: "#2ca02c",
        +2: "#d62728",
        +3: "#9467bd",
        +4: "#ff7f0e",
    }

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.3))

        for off, tgt in OFFSET_TARGETS.items():
            ax.scatter(tgt[0], tgt[1], marker="*", s=160,
                       color=offset_colors[off], edgecolor="#333333",
                       linewidths=0.6, zorder=5)
            ax.text(tgt[0], tgt[1] + 0.18, f"{off:+d}",
                    ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color=offset_colors[off])

        for r in results:
            x, y = r["mean_xy"]
            col = offset_colors[r["true_offset"]]
            marker = DIR_MARKERS.get(r["dir_key"], "o")
            if r["pred_offset"] == r["true_offset"]:
                ax.scatter(x, y, s=30, marker=marker, color=col, alpha=0.8, zorder=4)
            else:
                ax.scatter(x, y, s=36, marker="x", color=col, linewidths=1.0, zorder=6)

        ax.axhline(0, color="#BBBBBB", lw=0.7)
        ax.axvline(0, color="#BBBBBB", lw=0.7)
        ax.set_xlim(-4.6, 4.6)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("Readout x")
        ax.set_ylabel("Readout y")
        ax.set_title(
            f"Dynamic-summary COM readout\ntrain acc={train_acc*100:.0f}%  test acc={test_acc*100:.0f}%"
        )

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")

def plot_confusion_matrix_offset(cm_raw, class_order, train_acc, test_acc, out_path):
    vals = class_order
    C = len(vals)
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_raw / row_sum, 0.0)

    cmap_count = LinearSegmentedColormap.from_list(
        "nat_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    )
    cmap_recall = LinearSegmentedColormap.from_list(
        "nat_green", ["#F7FCF5", "#C7E9C0", "#74C476", "#238B45", "#00441B"]
    )

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4),
                                 gridspec_kw={"wspace": 0.65})
        panels = [
            (axes[0], cm_raw.astype(float), cmap_count, None, ".0f", "Count"),
            (axes[1], cm_norm, cmap_recall, 1.0, ".2f", "Recall"),
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
            ax.set_xticklabels([f"{v:+d}" for v in vals], rotation=30, ha="right")
            ax.set_yticks(range(C))
            ax.set_yticklabels([f"{v:+d}" for v in vals])
            ax.set_xlabel("Predicted offset")
            ax.set_ylabel("True offset")

            thresh = vm * 0.55
            for i in range(C):
                for j in range(C):
                    v = data[i, j]
                    ax.text(j, i, f"{v:{fmt}}",
                            ha="center", va="center",
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
            f"Offset confusion matrix  (train acc = {train_acc*100:.0f}%,  "
            f"test acc = {test_acc*100:.0f}%)",
            fontsize=7.5, fontweight="bold", y=1.03
        )
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")

def plot_pca_offset(model_pca, mu, std, train_samples_dict, test_samples_dict, out_path):
    if model_pca is None:
        print("  Skipped PCA plot because USE_PCA=False")
        return

    train_proj = {c: [] for c in ACTIVE_CLASSES}
    test_proj = {c: [] for c in ACTIVE_CLASSES}

    for cname in ACTIVE_CLASSES:
        for dir_key, fname in train_samples_dict.get(cname, []):
            p = resolve(dir_key, cname, fname)
            if not p.exists():
                continue
            disp, time = load_h5(p)
            feat, _ = extract_dynamic_summary_feature(
                disp, time, T_START, T_END, EXCLUDE_MARKERS
            )
            train_proj[cname].append(model_pca.transform(((feat[None, :] - mu) / std)))

        for dir_key, fname in test_samples_dict.get(cname, []):
            p = resolve(dir_key, cname, fname)
            if not p.exists():
                continue
            disp, time = load_h5(p)
            feat, _ = extract_dynamic_summary_feature(
                disp, time, T_START, T_END, EXCLUDE_MARKERS
            )
            test_proj[cname].append(model_pca.transform(((feat[None, :] - mu) / std)))

    for cname in ACTIVE_CLASSES:
        train_proj[cname] = np.vstack(train_proj[cname]) if train_proj[cname] else np.empty((0, model_pca.n_components_))
        test_proj[cname] = np.vstack(test_proj[cname]) if test_proj[cname] else np.empty((0, model_pca.n_components_))

    colors = {"coor_0": "#d62728", "coor_2": "#1f77b4"}

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))

        for cname in ACTIVE_CLASSES:
            Xp = train_proj[cname]
            Xt = test_proj[cname]
            col = colors[cname]

            if len(Xp) > 0:
                ax.scatter(Xp[:, 0], Xp[:, 1] if model_pca.n_components_ > 1 else np.zeros(len(Xp)),
                           c=col, s=14, alpha=0.8, label=f"{cname} train")
            if len(Xt) > 0:
                ax.scatter(Xt[:, 0], Xt[:, 1] if model_pca.n_components_ > 1 else np.zeros(len(Xt)),
                           facecolors="none", edgecolors=col, s=18, alpha=0.7, label=f"{cname} test")

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Dynamic-summary PCA space")
        ax.legend(fontsize=5.5, loc="best")
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*72}")
    print("  COM Demo — Dynamic-Summary 2D Ridge Regressor")
    print(f"{'═'*72}")
    print("  Train samples per class:")
    summarise_samples(TRAIN_SAMPLES)
    print("  Test samples per class:")
    summarise_samples(TEST_SAMPLES)
    print(f"  Time window       : {T_START}–{T_END} s")
    print(f"  Ridge alpha       : {RIDGE_ALPHA}")
    print(f"  Normalize energy  : {NORMALIZE_ENERGY}")
    print(f"  Use PCA           : {USE_PCA}")
    print(f"  PCA variance      : {PCA_VARIANCE}")
    print(f"  Early fraction    : {EARLY_FRAC}")
    print(f"  Late fraction     : {LATE_FRAC}")
    print(f"  Cosine decode     : {USE_COSINE_DECODE}\n")

    model, pca, mu, std, N, D, summary_dim = fit_dynamic_summary_offset(
        TRAIN_SAMPLES,
        ridge_alpha=RIDGE_ALPHA,
        use_pca=USE_PCA,
        pca_variance=PCA_VARIANCE
    )

    weight_path = out_dir / "dynamic_summary_ridge_weights.mat"

    save_dict = {
        "coef": model.coef_,
        "intercept": model.intercept_,
        "mu": mu,
        "std": std,
        "ridge_alpha": RIDGE_ALPHA,
        "T_START": T_START,
        "T_END": T_END,
        "EARLY_FRAC": EARLY_FRAC,
        "LATE_FRAC": LATE_FRAC,
        "summary_dim": summary_dim,
        "raw_dim": D,
        "n_markers": N,
        "normalize_energy": int(NORMALIZE_ENERGY),
        "use_pca": int(USE_PCA),
        "valid_offsets": VALID_OFFSETS,
        "offset_targets_keys": np.array(list(OFFSET_TARGETS.keys()), dtype=np.int32),
        "offset_targets_values": np.vstack(list(OFFSET_TARGETS.values())),
    }

    if pca is not None:
        save_dict["pca_components"] = pca.components_
        save_dict["pca_mean"] = pca.mean_
    else:
        save_dict["pca_components"] = np.empty((0, 0))
        save_dict["pca_mean"] = np.empty((0,))

    savemat(weight_path, save_dict)
    print(f"  Saved weights → {weight_path}")

    _, tr_true, tr_pred = run_dynamic_summary_offset(
        TRAIN_SAMPLES, model, pca, mu, std, label="Training set"
    )
    train_acc = float(np.mean(tr_true == tr_pred)) if len(tr_true) else 0.0
    print(f"\n  Train accuracy : {train_acc*100:.1f}%")

    results, te_true, te_pred = run_dynamic_summary_offset(
        TEST_SAMPLES, model, pca, mu, std, label="Test set"
    )
    test_acc = float(np.mean(te_true == te_pred)) if len(te_true) else 0.0
    print(f"\n  Test accuracy  : {test_acc*100:.1f}%")

    if len(te_true):
        labels_sorted = sorted(np.unique(np.concatenate([tr_true, te_true, te_pred])))
        print("\n  Per-class report (test):")
        print(classification_report(
            te_true, te_pred,
            labels=labels_sorted,
            target_names=[f"{v:+d}" for v in labels_sorted],
            digits=3, zero_division=0
        ))

    print()
    if len(te_true):
        labels_sorted = sorted(np.unique(np.concatenate([tr_true, te_true, te_pred])))
        cm = confusion_matrix(te_true, te_pred, labels=labels_sorted)
        plot_confusion_matrix_offset(
            cm, labels_sorted, train_acc, test_acc,
            out_dir / "dynamic_summary_confusion_matrix.svg"
        )

    plot_trial_points_offset(
        results, train_acc, test_acc,
        out_dir / "dynamic_summary_readout_points.png"
    )

    plot_pca_offset(
        pca, mu, std, TRAIN_SAMPLES, TEST_SAMPLES,
        out_dir / "dynamic_summary_pca.png"
    )

    print(f"\n  Raw state dim        : {D}")
    print(f"  Summary feature dim  : {summary_dim}")
    print(f"  Done. Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()