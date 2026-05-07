"""
COM Bridge Demo — 2D Offset Readout with Nearest-Target Decoding
================================================================
Train a 2D Ridge readout for discrete signed offsets:
    -3 -> [-3, 0]
    -2 -> [-2, 0]
    -1 -> [-1, 0]
    +1 -> [ 1, 0]
    +2 -> [ 2, 0]
    +3 -> [ 3, 0]

This mimics the old 2D classifier style:
- predict a 2D output
- decode by nearest target prototype

Note:
Since all targets lie on the x-axis, this is mathematically equivalent to
a 1D problem, but it preserves the 2D readout / nearest-target structure.
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


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "base":    f"{ROOT_DIR}/soft_state_100g_bending_sensor",
    "near":    f"{ROOT_DIR}/soft_state_100g_near_bending_sensor",
    "nearest": f"{ROOT_DIR}/soft_state_100g_nearest_bending_sensor",
}

OFFSET_BY_DIR_AND_CLASS = {
    "base": {
        "coor_0": +3,
        "coor_2": -3,
    },
    "near": {
        "coor_0": +2,
        "coor_2": -2,
    },
    "nearest": {
        "coor_0": +1,
        "coor_2": -1,
    },
}

# 2D targets
OFFSET_TARGETS = {
    -3: np.array([-3.0, 0.0]),
    -2: np.array([-2.0, 0.0]),
    -1: np.array([-1.0, 0.0]),
    +1: np.array([+1.0, 0.0]),
    +2: np.array([+2.0, 0.0]),
    +3: np.array([+3.0, 0.0]),
}

VALID_OFFSETS = np.array([-3, -2, -1, 1, 2, 3], dtype=int)

ACTIVE_CLASSES = ["coor_0", "coor_2"]

TRAIN_SAMPLES = {
    "coor_0": [
        ("base",    "trajectories_sample_0.h5"),
        ("base",    "trajectories_sample_1.h5"),
        ("base",    "trajectories_sample_2.h5"),
        ("base",    "trajectories_sample_3.h5"),
        ("base",    "trajectories_sample_4.h5"),
        ("base",    "trajectories_sample_5.h5"),
        ("base",    "trajectories_sample_6.h5"),
        ("near",    "trajectories_sample_0.h5"),
        ("near",    "trajectories_sample_1.h5"),
        ("near",    "trajectories_sample_2.h5"),
        ("near",    "trajectories_sample_3.h5"),
        ("near",    "trajectories_sample_4.h5"),
        ("near",    "trajectories_sample_5.h5"),
        ("near",    "trajectories_sample_6.h5"),
        ("nearest", "trajectories_sample_0.h5"),
        ("nearest", "trajectories_sample_1.h5"),
        ("nearest", "trajectories_sample_2.h5"),
        ("nearest", "trajectories_sample_3.h5"),
        ("nearest", "trajectories_sample_4.h5"),
        ("nearest", "trajectories_sample_5.h5"),
        ("nearest", "trajectories_sample_6.h5"),
    ],
    "coor_2": [
        ("base",    "trajectories_sample_0.h5"),
        ("base",    "trajectories_sample_1.h5"),
        ("base",    "trajectories_sample_2.h5"),
        ("base",    "trajectories_sample_3.h5"),
        ("base",    "trajectories_sample_4.h5"),
        ("base",    "trajectories_sample_5.h5"),
        ("base",    "trajectories_sample_6.h5"),
        ("near",    "trajectories_sample_0.h5"),
        ("near",    "trajectories_sample_1.h5"),
        ("near",    "trajectories_sample_2.h5"),
        ("near",    "trajectories_sample_3.h5"),
        ("near",    "trajectories_sample_4.h5"),
        ("near",    "trajectories_sample_5.h5"),
        ("near",    "trajectories_sample_6.h5"),
        ("nearest", "trajectories_sample_0.h5"),
        ("nearest", "trajectories_sample_1.h5"),
        ("nearest", "trajectories_sample_2.h5"),
        ("nearest", "trajectories_sample_3.h5"),
        ("nearest", "trajectories_sample_4.h5"),
        ("nearest", "trajectories_sample_5.h5"),
        ("nearest", "trajectories_sample_6.h5"),
    ],
}

TEST_SAMPLES = {
    "coor_0": [
        ("base",    "trajectories_sample_14.h5"),
        ("base",    "trajectories_sample_15.h5"),
        ("base",    "trajectories_sample_16.h5"),
        ("base",    "trajectories_sample_17.h5"),
        ("base",    "trajectories_sample_18.h5"),
        ("base",    "trajectories_sample_19.h5"),
        ("near",    "trajectories_sample_14.h5"),
        ("near",    "trajectories_sample_15.h5"),
        ("near",    "trajectories_sample_16.h5"),
        ("near",    "trajectories_sample_17.h5"),
        ("near",    "trajectories_sample_18.h5"),
        ("near",    "trajectories_sample_19.h5"),
        ("nearest", "trajectories_sample_14.h5"),
        ("nearest", "trajectories_sample_15.h5"),
        ("nearest", "trajectories_sample_16.h5"),
        ("nearest", "trajectories_sample_17.h5"),
        ("nearest", "trajectories_sample_18.h5"),
        ("nearest", "trajectories_sample_19.h5"),
    ],
    "coor_2": [
        ("base",    "trajectories_sample_14.h5"),
        ("base",    "trajectories_sample_15.h5"),
        ("base",    "trajectories_sample_16.h5"),
        ("base",    "trajectories_sample_17.h5"),
        ("base",    "trajectories_sample_18.h5"),
        ("base",    "trajectories_sample_19.h5"),
        ("near",    "trajectories_sample_14.h5"),
        ("near",    "trajectories_sample_15.h5"),
        ("near",    "trajectories_sample_16.h5"),
        ("near",    "trajectories_sample_17.h5"),
        ("near",    "trajectories_sample_18.h5"),
        ("near",    "trajectories_sample_19.h5"),
        ("nearest", "trajectories_sample_14.h5"),
        ("nearest", "trajectories_sample_15.h5"),
        ("nearest", "trajectories_sample_16.h5"),
        ("nearest", "trajectories_sample_17.h5"),
        ("nearest", "trajectories_sample_18.h5"),
        ("nearest", "trajectories_sample_19.h5"),
    ],
}

T_START = 0.0
T_END   = 3.0
RIDGE_ALPHA = 1.0
EXCLUDE_MARKERS = []
OUTPUT_DIR = f"{ROOT_DIR}/com_bridge_demo_output_2d"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_label(dir_key, fname):
    return f"{dir_key}/{fname.replace('trajectories_', '').replace('.h5', '')}"

def resolve(dir_key, coor_dir, fname):
    return Path(DIRS[dir_key]) / coor_dir / fname

def get_offset(dir_key, coor_name):
    if dir_key not in OFFSET_BY_DIR_AND_CLASS:
        raise KeyError(f"Missing dir_key='{dir_key}' in OFFSET_BY_DIR_AND_CLASS")
    if coor_name not in OFFSET_BY_DIR_AND_CLASS[dir_key]:
        raise KeyError(f"Missing coor_name='{coor_name}' for dir_key='{dir_key}' in OFFSET_BY_DIR_AND_CLASS")
    return int(OFFSET_BY_DIR_AND_CLASS[dir_key][coor_name])

def nearest_offset_from_xy(pred_xy):
    keys = list(OFFSET_TARGETS.keys())
    tgts = np.vstack([OFFSET_TARGETS[k] for k in keys])
    idx = np.argmin(np.linalg.norm(tgts - pred_xy[None, :], axis=1))
    return int(keys[idx])

def summarise_samples(samples_dict):
    for cname, slist in samples_dict.items():
        counts = {}
        for dk, _ in slist:
            counts[dk] = counts.get(dk, 0) + 1
        parts = [f"{counts[k]}×{k}" for k in sorted(counts.keys())]
        print(f"    {cname:8s}: {' + '.join(parts)}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
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
                col[:idx[0]] = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i + 1] - idx[i] > 1:
                        col[idx[i] + 1:idx[i + 1]] = col[idx[i]]

    baseline = pos[:min(baseline_frames, pos.shape[0])].mean(axis=0, keepdims=True)
    return pos - baseline, time

def extract_frame_features(disp, time, t_start, t_end, exclude_markers):
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end, side="right")) if t_end is not None else len(time)

    X = disp[i0:i1].reshape(i1 - i0, -1)
    ts = time[i0:i1] - time[i0]

    if exclude_markers:
        coord_dim = disp.shape[2]
        n_markers = disp.shape[1]
        keep = [coord_dim * n + d
                for n in range(n_markers)
                if n not in exclude_markers
                for d in range(coord_dim)]
        X = X[:, keep]

    return X, ts

def extract_trial_feature(disp, time, t_start, t_end, exclude_markers, normalize_energy=False):
    X, ts = extract_frame_features(disp, time, t_start, t_end, exclude_markers)
    feat = X.reshape(-1)

    if normalize_energy:
        denom = np.linalg.norm(feat)
        if denom > 1e-12:
            feat = feat / denom

    return feat, ts


# ══════════════════════════════════════════════════════════════════════════════
# DATASET BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_trial_dataset(samples_dict):
    rows = []
    y = []
    meta = []

    print("=" * 72)
    print("  Building trial-level dataset")
    print("=" * 72)

    for coor_name, sample_list in samples_dict.items():
        if coor_name not in ACTIVE_CLASSES:
            continue

        for dir_key, fname in sample_list:
            p = resolve(dir_key, coor_name, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} — skipped")
                continue

            disp, time = load_h5(p)
            feat, ts = extract_trial_feature(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            target = get_offset(dir_key, coor_name)

            rows.append(feat)
            y.append(target)
            meta.append({
                "coor_name": coor_name,
                "dir_key": dir_key,
                "fname": fname,
                "sample": sample_label(dir_key, fname),
                "target": target,
                "ts": ts,
            })

            print(f"  + [{dir_key}] {coor_name}/{fname}  target={target:+d}  dim={feat.shape[0]}")

    if not rows:
        raise RuntimeError("No usable samples found. Check CONFIG.")

    X = np.vstack(rows)
    y = np.array(y, dtype=float)
    return X, y, meta


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_regressor_2d(X_train, y_train, pca_variance=0.95):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    Xn = (X_train - mu) / std

    pca_full = PCA().fit(Xn)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = min(int(np.searchsorted(cumvar, pca_variance)) + 1,
            Xn.shape[1], max(1, Xn.shape[0] - 1))
    pca = PCA(n_components=k).fit(Xn)

    Y_train = np.vstack([OFFSET_TARGETS[int(v)] for v in y_train])

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xn, Y_train)

    print(f"\n  Trial feature dim : {X_train.shape[1]}")
    print(f"  Train matrix      : {Xn.shape}")
    print(f"  PCA               : {k} components ({pca.explained_variance_ratio_.sum() * 100:.1f}% variance)")
    print(f"  Ridge alpha       : {RIDGE_ALPHA}")
    print("  Output            : 2D offset target")

    return model, pca, mu, std


def run_regression_2d(X, y, meta, model, mu, std, label=""):
    print(f"\n{'=' * 72}")
    print(f"  Inference — {label}")
    print(f"{'=' * 72}")

    Xn = (X - mu) / std
    Y_cont = model.predict(Xn)

    y_pred = []
    results = []

    for i, m in enumerate(meta):
        pred_xy = Y_cont[i]
        pred_offset = nearest_offset_from_xy(pred_xy)
        y_pred.append(pred_offset)

        ok = "✓" if int(pred_offset) == int(y[i]) else "✗"
        print(
            f"  {ok}  {m['coor_name']}/{m['sample']}  "
            f"true={int(y[i]):+d}  pred_xy=({pred_xy[0]:+.3f}, {pred_xy[1]:+.3f})  pred={int(pred_offset):+d}"
        )

        results.append({
            "coor_name": m["coor_name"],
            "dir_key": m["dir_key"],
            "sample": m["sample"],
            "fname": m["fname"],
            "target": int(y[i]),
            "pred_x": float(pred_xy[0]),
            "pred_y": float(pred_xy[1]),
            "pred": int(pred_offset),
        })

    return results, Y_cont, np.array(y_pred, dtype=int)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    exact_acc = float(np.mean(y_true == y_pred))
    within1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    sign_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    return {
        "exact_acc": exact_acc,
        "within1": within1,
        "sign_acc": sign_acc,
        "mae": mae,
        "rmse": rmse,
    }


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

def plot_pred_xy(results, out_path):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))

        for off, tgt in OFFSET_TARGETS.items():
            ax.scatter(tgt[0], tgt[1], marker="*", s=120, color="black")
            ax.text(tgt[0], tgt[1] + 0.08, f"{off:+d}", ha="center", va="bottom", fontsize=6)

        xs = [r["pred_x"] for r in results]
        ys = [r["pred_y"] for r in results]
        ax.scatter(xs, ys, s=20, alpha=0.75)

        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Predicted x")
        ax.set_ylabel("Predicted y")
        ax.set_title("Predicted 2D readout")
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")

def plot_offset_confusion(y_true, y_pred, valid_offsets, out_path):
    vals = list(valid_offsets)
    idx_map = {v: i for i, v in enumerate(vals)}

    cm = np.zeros((len(vals), len(vals)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[idx_map[int(yt)], idx_map[int(yp)]] += 1

    overall_correct = int(np.trace(cm))
    overall_total = int(cm.sum())
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    cmap_count = LinearSegmentedColormap.from_list(
        "nat_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    )

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(3.8, 3.2))

        vm = max(cm.max(), 1e-6)
        im = ax.imshow(cm.astype(float), cmap=cmap_count, vmin=0, vmax=vm,
                       interpolation="nearest", aspect="equal")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.85)
        cb.set_label("Count", labelpad=4, fontsize=6)
        cb.ax.tick_params(labelsize=5.5, width=0.5, length=2, pad=2)
        cb.outline.set_linewidth(0.4)

        ax.set_xticks(range(len(vals)))
        ax.set_yticks(range(len(vals)))
        ax.set_xticklabels([f"{v:+d}" for v in vals], rotation=30, ha="right")
        ax.set_yticklabels([f"{v:+d}" for v in vals])
        ax.set_xlabel("Predicted offset")
        ax.set_ylabel("True offset")
        ax.set_title("Count")

        thresh = vm * 0.55
        for i in range(len(vals)):
            for j in range(len(vals)):
                v = cm[i, j]
                ax.text(
                    j, i, f"{int(v)}",
                    ha="center", va="center",
                    fontsize=5.5, fontweight="bold",
                    color="white" if v > thresh else "#333333"
                )

        for k in range(len(vals) + 1):
            ax.axhline(k - 0.5, color="white", lw=0.4)
            ax.axvline(k - 0.5, color="white", lw=0.4)

        ax.set_xlim(-0.5, len(vals) - 0.5)
        ax.set_ylim(len(vals) - 0.5, -0.5)

        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

        fig.suptitle(
            f"Offset confusion matrix  |  overall accuracy = "
            f"{overall_acc*100:.1f}%  ({overall_correct} / {overall_total} correct)",
            fontsize=7.5, fontweight="bold", y=1.03
        )

        fig.savefig(str(out_path))
        plt.close(fig)

    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 72}")
    print("  COM Bridge Demo — 2D Offset Readout")
    print(f"{'═' * 72}")
    print("  Active classes:")
    for cname in ACTIVE_CLASSES:
        print(f"    {cname}")
    print("\n  Train samples per class:")
    summarise_samples({k: v for k, v in TRAIN_SAMPLES.items() if k in ACTIVE_CLASSES})
    print("  Test samples per class:")
    summarise_samples({k: v for k, v in TEST_SAMPLES.items() if k in ACTIVE_CLASSES})
    print(f"  Time window : {T_START}–{T_END} s")
    print(f"  Valid offsets: {VALID_OFFSETS.tolist()}")

    X_train, y_train, meta_train = build_trial_dataset(TRAIN_SAMPLES)
    X_test, y_test, meta_test = build_trial_dataset(TEST_SAMPLES)

    model, pca, mu, std = fit_regressor_2d(X_train, y_train)

    train_results, Y_train_cont, y_train_pred = run_regression_2d(
        X_train, y_train, meta_train, model, mu, std, label="Training set"
    )
    train_metrics = compute_metrics(y_train, y_train_pred)

    print("\n  Train metrics:")
    print(f"    Exact cell accuracy : {train_metrics['exact_acc'] * 100:.1f}%")
    print(f"    Within-1-cell acc   : {train_metrics['within1'] * 100:.1f}%")
    print(f"    Sign accuracy       : {train_metrics['sign_acc'] * 100:.1f}%")
    print(f"    MAE (cells)         : {train_metrics['mae']:.3f}")
    print(f"    RMSE (cells)        : {train_metrics['rmse']:.3f}")

    test_results, Y_test_cont, y_test_pred = run_regression_2d(
        X_test, y_test, meta_test, model, mu, std, label="Test set"
    )
    test_metrics = compute_metrics(y_test, y_test_pred)

    print("\n  Test metrics:")
    print(f"    Exact cell accuracy : {test_metrics['exact_acc'] * 100:.1f}%")
    print(f"    Within-1-cell acc   : {test_metrics['within1'] * 100:.1f}%")
    print(f"    Sign accuracy       : {test_metrics['sign_acc'] * 100:.1f}%")
    print(f"    MAE (cells)         : {test_metrics['mae']:.3f}")
    print(f"    RMSE (cells)        : {test_metrics['rmse']:.3f}")

    print()
    plot_pred_xy(test_results, out_dir / "pred_xy_test.png")
    plot_offset_confusion(y_test, y_test_pred, VALID_OFFSETS, out_dir / "offset_confusion_test.png")

    print(f"\n  Done. Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()