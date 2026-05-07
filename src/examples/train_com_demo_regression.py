"""
COM Demo — Dynamic-Summary Scalar Ridge Regressor
=================================================
This version uses the same dynamic-summary feature vector as the current code,
but predicts a scalar offset directly instead of a 2D target.

Prediction flow:
    trial -> dynamic-summary feature -> scalar offset -> round to nearest valid class

This allows interpolation to unseen offsets such as -1, 0, +1.
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
    "nearest": {
        "coor_0": +1,
        "coor_2": -1,
    },
    "bar_near": {
        "coor_0": +2,
        "coor_2": -2,
    },
}

ACTIVE_CLASSES = ["coor_0", "coor_2"]

# Train only on seen offsets. Round to all allowed offsets.
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
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
        ("base", "trajectories_sample_12.h5"),
        ("base", "trajectories_sample_13.h5"),
        ("base", "trajectories_sample_14.h5"),
        ("base", "trajectories_sample_15.h5"),
        ("base", "trajectories_sample_16.h5"),
        ("base", "trajectories_sample_17.h5"),
        ("base", "trajectories_sample_18.h5"),
        ("base", "trajectories_sample_19.h5"),
        ("nearest", "trajectories_sample_0.h5"),
        ("nearest", "trajectories_sample_1.h5"),
        ("nearest", "trajectories_sample_2.h5"),
        ("nearest", "trajectories_sample_3.h5"),
        ("nearest", "trajectories_sample_4.h5"),
        ("nearest", "trajectories_sample_5.h5"),
        ("nearest", "trajectories_sample_6.h5"),
        ("nearest", "trajectories_sample_7.h5"),
        ("nearest", "trajectories_sample_8.h5"),
        ("nearest", "trajectories_sample_9.h5"),
        ("nearest", "trajectories_sample_10.h5"),
        ("nearest", "trajectories_sample_11.h5"),
        ("nearest", "trajectories_sample_12.h5"),
        ("nearest", "trajectories_sample_13.h5"),
        ("nearest", "trajectories_sample_14.h5"),
        ("nearest", "trajectories_sample_15.h5"),
        ("nearest", "trajectories_sample_16.h5"),
        ("nearest", "trajectories_sample_17.h5"),
        ("nearest", "trajectories_sample_18.h5"),
        ("nearest", "trajectories_sample_19.h5"),
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
        ("base", "trajectories_sample_8.h5"),
        ("base", "trajectories_sample_9.h5"),
        ("base", "trajectories_sample_10.h5"),
        ("base", "trajectories_sample_11.h5"),
        ("base", "trajectories_sample_12.h5"),
        ("base", "trajectories_sample_13.h5"),
        ("base", "trajectories_sample_14.h5"),
        ("base", "trajectories_sample_15.h5"),
        ("base", "trajectories_sample_16.h5"),
        ("base", "trajectories_sample_17.h5"),
        ("base", "trajectories_sample_18.h5"),
        ("base", "trajectories_sample_19.h5"),
        ("nearest", "trajectories_sample_0.h5"),
        ("nearest", "trajectories_sample_1.h5"),
        ("nearest", "trajectories_sample_2.h5"),
        ("nearest", "trajectories_sample_3.h5"),
        ("nearest", "trajectories_sample_4.h5"),
        ("nearest", "trajectories_sample_5.h5"),
        ("nearest", "trajectories_sample_6.h5"),
        ("nearest", "trajectories_sample_7.h5"),
        ("nearest", "trajectories_sample_8.h5"),
        ("nearest", "trajectories_sample_9.h5"),
        ("nearest", "trajectories_sample_10.h5"),
        ("nearest", "trajectories_sample_11.h5"),
        ("nearest", "trajectories_sample_12.h5"),
        ("nearest", "trajectories_sample_13.h5"),
        ("nearest", "trajectories_sample_14.h5"),
        ("nearest", "trajectories_sample_15.h5"),
        ("nearest", "trajectories_sample_16.h5"),
        ("nearest", "trajectories_sample_17.h5"),
        ("nearest", "trajectories_sample_18.h5"),
        ("nearest", "trajectories_sample_19.h5"),
    ],
}

TEST_SAMPLES = {
    "coor_0": [
        ("near", "trajectories_sample_0.h5"),
        ("near", "trajectories_sample_1.h5"),
        ("near", "trajectories_sample_2.h5"),
        ("near", "trajectories_sample_3.h5"),
        ("near", "trajectories_sample_4.h5"),
        ("near", "trajectories_sample_5.h5"),
        ("near", "trajectories_sample_6.h5"),
        ("near", "trajectories_sample_7.h5"),
        ("near", "trajectories_sample_8.h5"),
        ("near", "trajectories_sample_9.h5"),
        ("near", "trajectories_sample_10.h5"),
        ("near", "trajectories_sample_11.h5"),
        ("near", "trajectories_sample_12.h5"),
        ("near", "trajectories_sample_13.h5"),
        ("near", "trajectories_sample_14.h5"),
        ("near", "trajectories_sample_15.h5"),
        ("near", "trajectories_sample_16.h5"),
        ("near", "trajectories_sample_17.h5"),
        ("near", "trajectories_sample_18.h5"),
        ("near", "trajectories_sample_19.h5"),
    ],
    "coor_2": [
        ("near", "trajectories_sample_0.h5"),
        ("near", "trajectories_sample_1.h5"),
        ("near", "trajectories_sample_2.h5"),
        ("near", "trajectories_sample_3.h5"),
        ("near", "trajectories_sample_4.h5"),
        ("near", "trajectories_sample_5.h5"),
        ("near", "trajectories_sample_6.h5"),
        ("near", "trajectories_sample_7.h5"),
        ("near", "trajectories_sample_8.h5"),
        ("near", "trajectories_sample_9.h5"),
        ("near", "trajectories_sample_10.h5"),
        ("near", "trajectories_sample_11.h5"),
        ("near", "trajectories_sample_12.h5"),
        ("near", "trajectories_sample_13.h5"),
        ("near", "trajectories_sample_14.h5"),
        ("near", "trajectories_sample_15.h5"),
        ("near", "trajectories_sample_16.h5"),
        ("near", "trajectories_sample_17.h5"),
        ("near", "trajectories_sample_18.h5"),
        ("near", "trajectories_sample_19.h5"),
    ],
}

# allowed output bins after regression
ROUND_CLASSES = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)

T_START = 0.0
T_END   = 3.0

EXCLUDE_MARKERS = []
NORMALIZE_ENERGY = False

EARLY_FRAC = 0.35
LATE_FRAC  = 0.35

RIDGE_ALPHA = 1.0
USE_PCA = False
PCA_VARIANCE = 0.95

OUTPUT_DIR = f"{ROOT_DIR}/com_demo_100g"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_label(dir_key, fname):
    return f"{dir_key}/{fname.replace('trajectories_','').replace('.h5','')}"

def resolve(dir_key, coor_dir, fname):
    return Path(DIRS[dir_key]) / coor_dir / fname

def get_offset(dir_key, coor_name):
    return float(OFFSET_BY_DIR_AND_CLASS[dir_key][coor_name])

def round_to_offset_class(x):
    return float(ROUND_CLASSES[np.argmin(np.abs(ROUND_CLASSES - x))])

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
# SCALAR RIDGE REGRESSOR
# ══════════════════════════════════════════════════════════════════════════════

def fit_scalar_regressor(train_samples_dict, ridge_alpha=1.0,
                         use_pca=False, pca_variance=0.95):
    X_rows, y_rows = [], []
    raw_dim = None
    summary_dim = None

    print("=" * 72)
    print("  Building dynamic-summary scalar-regression matrix")
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
            feat, _ = extract_dynamic_summary_feature(disp, time, T_START, T_END, EXCLUDE_MARKERS)
            target = get_offset(dir_key, coor_name)

            if raw_dim is None:
                raw_dim = X_raw.shape[1]
                summary_dim = feat.shape[0]

            X_rows.append(feat)
            y_rows.append(target)

            print(
                f"  + [{dir_key}] {coor_name}/{fname}  "
                f"target={target:+.1f}  raw_dim={X_raw.shape[1]}  summary_dim={feat.shape[0]}"
            )

    if not X_rows:
        raise RuntimeError("No training data found.")

    X_train = np.vstack(X_rows)
    y_train = np.asarray(y_rows, dtype=np.float64)

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
    model.fit(Z, y_train)

    print(f"\n  Raw state dim      : {raw_dim}")
    print(f"  Summary feature dim: {summary_dim}")
    print(f"  Train matrix       : {Z.shape} -> {y_train.shape}")
    print(f"  Ridge alpha        : {ridge_alpha}")

    return model, pca, mu, std, raw_dim, summary_dim

def run_scalar_regressor(samples_dict, model, pca, mu, std, label=""):
    rows = []
    y_true = []
    y_pred_cont = []
    y_pred_round = []

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
            feat, ts = extract_dynamic_summary_feature(disp, time, T_START, T_END, EXCLUDE_MARKERS)

            x = feat[None, :]
            xn = (x - mu) / std
            z = pca.transform(xn) if pca is not None else xn

            pred_cont = float(model.predict(z)[0])
            pred_round = round_to_offset_class(pred_cont)
            true_offset = get_offset(dir_key, coor_name)

            ok = "✓" if pred_round == true_offset else "✗"
            slabel = sample_label(dir_key, fname)

            print(
                f"  {ok}  {coor_name}/{slabel}  "
                f"true={true_offset:+.1f}  pred_cont={pred_cont:+.3f}  pred_round={pred_round:+.1f}"
            )

            rows.append(dict(
                true_offset=true_offset,
                pred_cont=pred_cont,
                pred_round=pred_round,
                coor_name=coor_name,
                sample=slabel,
                dir_key=dir_key,
                ts=ts,
            ))

            y_true.append(true_offset)
            y_pred_cont.append(pred_cont)
            y_pred_round.append(pred_round)

    return rows, np.asarray(y_true), np.asarray(y_pred_cont), np.asarray(y_pred_round)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_regression_scatter(rows, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    xs = [r["true_offset"] for r in rows]
    ys = [r["pred_cont"] for r in rows]

    ax.scatter(xs, ys, s=30, alpha=0.8)

    lo = min(xs + ys) - 0.5
    hi = max(xs + ys) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0)

    ax.set_xlabel("True offset")
    ax.set_ylabel("Predicted continuous offset")
    ax.set_title("Dynamic-summary scalar regression")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*72}")
    print("  COM Demo — Dynamic-Summary Scalar Ridge Regressor")
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
    print(f"  Late fraction     : {LATE_FRAC}\n")

    model, pca, mu, std, raw_dim, summary_dim = fit_scalar_regressor(
        TRAIN_SAMPLES,
        ridge_alpha=RIDGE_ALPHA,
        use_pca=USE_PCA,
        pca_variance=PCA_VARIANCE
    )

    _, ytr_true, ytr_cont, ytr_round = run_scalar_regressor(
        TRAIN_SAMPLES, model, pca, mu, std, label="Training set"
    )
    train_mae = mean_absolute_error(ytr_true, ytr_cont)
    train_rmse = np.sqrt(mean_squared_error(ytr_true, ytr_cont))
    train_r2 = r2_score(ytr_true, ytr_cont)
    train_round_acc = float(np.mean(ytr_round == ytr_true))

    print(f"\n  Train MAE        : {train_mae:.4f}")
    print(f"  Train RMSE       : {train_rmse:.4f}")
    print(f"  Train R^2        : {train_r2:.4f}")
    print(f"  Train round-acc  : {train_round_acc*100:.1f}%")

    rows, yte_true, yte_cont, yte_round = run_scalar_regressor(
        TEST_SAMPLES, model, pca, mu, std, label="Test set"
    )
    test_mae = mean_absolute_error(yte_true, yte_cont)
    test_rmse = np.sqrt(mean_squared_error(yte_true, yte_cont))
    test_r2 = r2_score(yte_true, yte_cont)
    test_round_acc = float(np.mean(yte_round == yte_true))

    print(f"\n  Test MAE         : {test_mae:.4f}")
    print(f"  Test RMSE        : {test_rmse:.4f}")
    print(f"  Test R^2         : {test_r2:.4f}")
    print(f"  Test round-acc   : {test_round_acc*100:.1f}%")

    plot_regression_scatter(rows, out_dir / "scalar_regression_scatter.png")

    print(f"\n  Raw state dim        : {raw_dim}")
    print(f"  Summary feature dim  : {summary_dim}")
    print(f"  Done. Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()