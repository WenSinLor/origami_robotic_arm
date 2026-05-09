"""
COM Demo - Bar-Outer +4 Actual Demo
===================================

Standalone dynamic-summary COM readout for the actual bar-outer demo.

Training:
    bar_base/coor_0 -> +3
    bar_base/coor_2 -> -3

Testing:
    bar_outer/coor_0 -> +4

Important:
    - This script intentionally does not include a -4 target.
    - It does not modify or depend on the other COM demo scripts.
    - The feature vector is the dynamic-summary feature only.
"""

from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, confusion_matrix
from scipy.io import savemat


# ============================================================================
# CONFIG
# ============================================================================

ROOT_DIR = "/home/wensin/Documents/origami_robotic_arm/data"

DIRS = {
    "bar_base": f"{ROOT_DIR}/com_demo_60g",
    "bar_outer": f"{ROOT_DIR}/com_demo_100g_outer",
}

OFFSET_BY_DIR_AND_CLASS = {
    "bar_base": {
        "coor_0": +3,
        "coor_2": -3,
    },
    "bar_outer": {
        "coor_0": +4,
    },
}

ACTIVE_CLASSES = ["coor_0", "coor_2"]
TEST_CLASSES = ["coor_0"]

OFFSET_TARGETS = {
    -3: np.array([-3.0, 0.0]),
    +3: np.array([+3.0, 0.0]),
    +4: np.array([+4.0, 0.0]),
}
VALID_OFFSETS = np.array([-3, 3, 4], dtype=int)

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
        ("bar_base", "trajectories_sample_12.h5"),
        ("bar_base", "trajectories_sample_13.h5"),
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
        ("bar_base", "trajectories_sample_12.h5"),
        ("bar_base", "trajectories_sample_13.h5"),
    ],
}

TEST_SAMPLES = {
    "coor_0": [
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
        ("bar_outer", "trajectories_sample_10.h5"),
        ("bar_outer", "trajectories_sample_11.h5"),
        ("bar_outer", "trajectories_sample_12.h5"),
        ("bar_outer", "trajectories_sample_13.h5"),
        ("bar_outer", "trajectories_sample_14.h5"),
        ("bar_outer", "trajectories_sample_15.h5"),
        ("bar_outer", "trajectories_sample_16.h5"),
        ("bar_outer", "trajectories_sample_17.h5"),
        ("bar_outer", "trajectories_sample_18.h5"),
        ("bar_outer", "trajectories_sample_19.h5"),
    ],
}

T_START = 0.0
T_END = 3.0
EXCLUDE_MARKERS = []
NORMALIZE_ENERGY = False
EARLY_FRAC = 0.35
LATE_FRAC = 0.35
RIDGE_ALPHA = 1.0
STREAM_MIN_FRAMES = 5
STREAM_STEP_FRAMES = 1
STREAM_TARGET_OFFSET = 4.0
STREAM_TOLERANCE_ABS = 0.5

OUTPUT_DIR = f"{ROOT_DIR}/com_demo_100g_outer/bar_outer_dynamic_summary"


# ============================================================================
# HELPERS
# ============================================================================

def sample_label(dir_key, fname):
    return f"{dir_key}/{fname.replace('trajectories_', '').replace('.h5', '')}"


def resolve(dir_key, coor_dir, fname):
    return Path(DIRS[dir_key]) / coor_dir / fname


def get_offset(dir_key, coor_name):
    return int(OFFSET_BY_DIR_AND_CLASS[dir_key][coor_name])


def get_target_xy(dir_key, coor_name):
    return OFFSET_TARGETS[get_offset(dir_key, coor_name)].astype(float)


def decode_offset(p_xy):
    keys = list(OFFSET_TARGETS.keys())
    tgts = np.vstack([OFFSET_TARGETS[k] for k in keys])
    idx = int(np.argmin(np.linalg.norm(tgts - p_xy[None, :], axis=1)))
    return int(keys[idx])


def summarise_samples(samples_dict):
    for cname, slist in samples_dict.items():
        counts = {}
        for dk, _ in slist:
            counts[dk] = counts.get(dk, 0) + 1
        parts = [f"{counts[k]}x{k}" for k in sorted(counts)]
        print(f"    {cname:8s}: {' + '.join(parts) if parts else '0'}")


# ============================================================================
# DATA AND FEATURES
# ============================================================================

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


def build_dynamic_summary_feature(X):
    T, _ = X.shape
    n_early = max(1, int(np.floor(EARLY_FRAC * T)))
    n_late = max(1, int(np.floor(LATE_FRAC * T)))

    X_early = X[:n_early]
    X_late = X[-n_late:]

    feat = np.concatenate([
        X.mean(axis=0),
        X.std(axis=0),
        X_early.mean(axis=0),
        X_late.mean(axis=0),
        X_early.std(axis=0),
        X_late.std(axis=0),
        X_late.mean(axis=0) - X_early.mean(axis=0),
        np.sqrt(np.mean(X**2, axis=0)),
    ])

    if NORMALIZE_ENERGY:
        denom = np.linalg.norm(feat)
        if denom > 1e-12:
            feat = feat / denom

    return feat.astype(np.float64)


def extract_dynamic_summary_feature(disp, time, t_start, t_end, exclude_markers):
    X, ts = extract_features(disp, time, t_start, t_end, exclude_markers)
    return build_dynamic_summary_feature(X), ts, X.shape[1]


# ============================================================================
# MODEL
# ============================================================================

def fit_dynamic_summary_offset(train_samples_dict):
    X_rows, Y_rows = [], []
    raw_dim = summary_dim = None

    print("=" * 72)
    print("  Building dynamic-summary training matrix for bar-base +/-3")
    print("=" * 72)

    for coor_name in ACTIVE_CLASSES:
        for dir_key, fname in train_samples_dict.get(coor_name, []):
            p = resolve(dir_key, coor_name, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} -- skipped")
                continue

            disp, time = load_h5(p)
            feat, _, raw_d = extract_dynamic_summary_feature(
                disp, time, T_START, T_END, EXCLUDE_MARKERS
            )
            tgt = get_target_xy(dir_key, coor_name)

            if raw_dim is None:
                raw_dim = raw_d
                summary_dim = feat.shape[0]

            X_rows.append(feat)
            Y_rows.append(tgt)
            print(
                f"  + [{dir_key}] {coor_name}/{fname}  "
                f"target={get_offset(dir_key, coor_name):+d}  "
                f"summary_dim={feat.shape[0]}"
            )

    if not X_rows:
        raise RuntimeError("No training data found. Check TRAIN_SAMPLES / DIRS.")

    X_train = np.vstack(X_rows)
    Y_train = np.vstack(Y_rows)

    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    Xn = (X_train - mu) / std

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(Xn, Y_train)

    print(f"\n  Raw state dim      : {raw_dim}")
    print(f"  Summary feature dim: {summary_dim}")
    print(f"  Train matrix       : {Xn.shape} -> {Y_train.shape}")
    print(f"  Ridge alpha        : {RIDGE_ALPHA}")

    return model, mu, std, raw_dim, summary_dim


def run_dynamic_summary_offset(samples_dict, model, mu, std, label="", classes=None):
    classes = classes or ACTIVE_CLASSES
    results = []
    y_true, y_pred = [], []

    print(f"\n{'=' * 72}")
    print(f"  Inference -- {label}")
    print(f"{'=' * 72}")

    for coor_name in classes:
        for dir_key, fname in samples_dict.get(coor_name, []):
            p = resolve(dir_key, coor_name, fname)
            if not p.exists():
                print(f"  [WARN] Missing: {p} -- skipped")
                continue

            disp, time = load_h5(p)
            feat, ts, _ = extract_dynamic_summary_feature(
                disp, time, T_START, T_END, EXCLUDE_MARKERS
            )

            pred_xy = model.predict((feat[None, :] - mu) / std)[0]
            true_offset = get_offset(dir_key, coor_name)
            pred_offset = decode_offset(pred_xy)
            tick = "OK" if pred_offset == true_offset else "!!"
            slabel = sample_label(dir_key, fname)

            print(
                f"  {tick}  {coor_name}/{slabel}  "
                f"true={true_offset:+d}  pred={pred_offset:+d}  "
                f"readout=({pred_xy[0]:+.3f}, {pred_xy[1]:+.3f})"
            )

            results.append(dict(
                true_offset=true_offset,
                pred_offset=pred_offset,
                coor_name=coor_name,
                sample=slabel,
                dir_key=dir_key,
                ts=ts,
                mean_xy=pred_xy,
                target_xy=OFFSET_TARGETS[int(true_offset)].astype(float),
            ))
            y_true.append(true_offset)
            y_pred.append(pred_offset)

    return results, np.array(y_true), np.array(y_pred)


def predict_streaming_readout(disp, time, model, mu, std):
    X, ts = extract_features(disp, time, T_START, T_END, EXCLUDE_MARKERS)
    n_frames = len(X)
    if n_frames < STREAM_MIN_FRAMES:
        return None

    times, xy_rows, decoded = [], [], []
    for stop in range(STREAM_MIN_FRAMES, n_frames + 1, STREAM_STEP_FRAMES):
        feat = build_dynamic_summary_feature(X[:stop])
        pred_xy = model.predict((feat[None, :] - mu) / std)[0]
        times.append(ts[stop - 1])
        xy_rows.append(pred_xy)
        decoded.append(decode_offset(pred_xy))

    return {
        "time": np.asarray(times, dtype=float),
        "xy": np.vstack(xy_rows),
        "decoded": np.asarray(decoded, dtype=int),
    }


def collect_streaming_readouts(samples_dict, model, mu, std, classes=None):
    classes = classes or ACTIVE_CLASSES
    streams = []

    for coor_name in classes:
        for dir_key, fname in samples_dict.get(coor_name, []):
            p = resolve(dir_key, coor_name, fname)
            if not p.exists():
                continue

            disp, time = load_h5(p)
            stream = predict_streaming_readout(disp, time, model, mu, std)
            if stream is None:
                continue

            stream.update(dict(
                true_offset=get_offset(dir_key, coor_name),
                coor_name=coor_name,
                sample=sample_label(dir_key, fname),
                dir_key=dir_key,
            ))
            streams.append(stream)

    return streams


# ============================================================================
# FIGURES
# ============================================================================

VT_MAROON = "#861F41"
VT_ORANGE = "#E5751F"
VT_STONE = "#75787B"
VT_DARK_STONE = "#54585A"
VT_LIGHT_STONE = "#D7D2CB"
VT_PALE_MAROON = "#F2E8ED"
VT_PALE_ORANGE = "#FBE9DC"
VT_GOLD = "#B3A369"

PLOT_FONT_SIZES = {
    "base": 9,
    "axis_label": 9,
    "tick": 8,
    "legend": 7.5,
    "title": 10,
    "panel_title": 9,
    "suptitle": 10,
    "annotation": 7,
    "matrix_cell": 8,
    "class_label": 8,
    "colorbar_label": 8,
    "colorbar_tick": 7,
}

NATURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": PLOT_FONT_SIZES["base"],
    "axes.titlesize": PLOT_FONT_SIZES["panel_title"],
    "axes.labelsize": PLOT_FONT_SIZES["axis_label"],
    "xtick.labelsize": PLOT_FONT_SIZES["tick"],
    "ytick.labelsize": PLOT_FONT_SIZES["tick"],
    "legend.fontsize": PLOT_FONT_SIZES["legend"],
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

OFFSET_COLORS = {
    -3: VT_MAROON,
    +3: VT_DARK_STONE,
    +4: VT_ORANGE,
}

DIR_MARKERS = {
    "bar_base": "o",
    "bar_outer": "^",
}


def plot_trial_points_offset(results, train_acc, test_acc, out_path):
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(4.1, 3.3))

        for off, tgt in OFFSET_TARGETS.items():
            ax.scatter(tgt[0], tgt[1], marker="*", s=165,
                       color=OFFSET_COLORS[off], edgecolor=VT_DARK_STONE,
                       linewidths=0.6, zorder=5)
            ax.text(tgt[0], tgt[1] + 0.18, f"{off:+d}",
                    ha="center", va="bottom",
                    fontsize=PLOT_FONT_SIZES["class_label"],
                    fontweight="bold", color=OFFSET_COLORS[off])

        for r in results:
            x, y = r["mean_xy"]
            col = OFFSET_COLORS[r["true_offset"]]
            marker = DIR_MARKERS.get(r["dir_key"], "o")
            if r["pred_offset"] == r["true_offset"]:
                ax.scatter(x, y, s=32, marker=marker, color=col, alpha=0.85, zorder=4)
            else:
                ax.scatter(x, y, s=38, marker="x", color=col, linewidths=1.0, zorder=6)

        ax.axhline(0, color=VT_LIGHT_STONE, lw=0.7)
        ax.axvline(0, color=VT_LIGHT_STONE, lw=0.7)
        ax.set_xlim(-3.8, 4.6)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("Predicted offset", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.set_ylabel("Residual readout", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        ax.set_title(
            f"Bar-outer +4 dynamic-summary readout\n"
            f"train acc={train_acc*100:.0f}%  test acc={test_acc*100:.0f}%",
            fontsize=PLOT_FONT_SIZES["title"], fontweight="bold",
            color=VT_MAROON,
        )

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved -> {out_path}")


def _stack_stream_values(streams, key="x"):
    if not streams:
        return None, None

    t0 = max(float(s["time"][0]) for s in streams)
    t1 = min(float(s["time"][-1]) for s in streams)
    n = min(len(s["time"]) for s in streams)
    if n <= 1 or t1 <= t0:
        return None, None

    t_common = np.linspace(t0, t1, n)
    rows = []
    for s in streams:
        if key == "x":
            values = s["xy"][:, 0]
        elif key == "y":
            values = s["xy"][:, 1]
        else:
            raise ValueError(f"Unknown stream key: {key}")
        rows.append(np.interp(t_common, s["time"], values))

    return t_common, np.vstack(rows)


def compute_time_to_tolerance_band(t_common, x_mean):
    lower = STREAM_TARGET_OFFSET - STREAM_TOLERANCE_ABS
    upper = STREAM_TARGET_OFFSET + STREAM_TOLERANCE_ABS
    in_band = (x_mean >= lower) & (x_mean <= upper)
    hit_idx = np.where(in_band)[0]

    if len(hit_idx) == 0:
        return {
            "target": STREAM_TARGET_OFFSET,
            "tolerance_abs": STREAM_TOLERANCE_ABS,
            "lower": lower,
            "upper": upper,
            "time_to_enter": np.nan,
            "predicted_offset_at_entry": np.nan,
            "entered": False,
        }

    idx = int(hit_idx[0])
    return {
        "target": STREAM_TARGET_OFFSET,
        "tolerance_abs": STREAM_TOLERANCE_ABS,
        "lower": lower,
        "upper": upper,
        "time_to_enter": float(t_common[idx]),
        "predicted_offset_at_entry": float(x_mean[idx]),
        "entered": True,
    }


def save_streaming_metrics(metrics, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Streaming readout timing metrics\n")
        f.write("================================\n")
        f.write(f"target_offset: {metrics['target']:.6g}\n")
        f.write(f"tolerance_abs: {metrics['tolerance_abs']:.6g}\n")
        f.write(f"band_lower: {metrics['lower']:.6g}\n")
        f.write(f"band_upper: {metrics['upper']:.6g}\n")
        f.write(f"entered_band: {int(metrics['entered'])}\n")
        if metrics["entered"]:
            f.write(f"time_to_enter_s: {metrics['time_to_enter']:.6g}\n")
            f.write(
                f"predicted_offset_at_entry: "
                f"{metrics['predicted_offset_at_entry']:.6g}\n"
            )
        else:
            f.write("time_to_enter_s: nan\n")
            f.write("predicted_offset_at_entry: nan\n")
    print(f"  Saved -> {out_path}")


def plot_streaming_offset(streams, out_path):
    t_common, x_rows = _stack_stream_values(streams, key="x")
    if t_common is None:
        print("  Skipped streaming offset plot because no stream data were available")
        return None

    x_mean = x_rows.mean(axis=0)
    x_std = x_rows.std(axis=0)
    metrics = compute_time_to_tolerance_band(t_common, x_mean)

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(5.4, 3.1))

        for row in x_rows:
            ax.plot(t_common, row, color=VT_ORANGE, lw=0.75, alpha=0.25)

        ax.fill_between(
            t_common, x_mean - x_std, x_mean + x_std,
            color=VT_PALE_ORANGE, alpha=0.95, linewidth=0.0,
            label="+/-1 SD"
        )
        ax.plot(t_common, x_mean, color=VT_MAROON, lw=1.7,
                label="mean predicted offset")

        ax.axhspan(metrics["lower"], metrics["upper"],
                   color=VT_LIGHT_STONE, alpha=0.72)
        ax.axhline(metrics["lower"], color=VT_ORANGE, lw=1.0, ls=":")
        ax.axhline(metrics["upper"], color=VT_ORANGE, lw=1.0, ls=":")
        ax.axhline(STREAM_TARGET_OFFSET, color=VT_ORANGE, lw=1.1,
                   ls="-", label="+4 target offset")
        ax.text(
            t_common[-1], metrics["upper"],
            f"  +/-0.5 band [{metrics['lower']:.1f}, {metrics['upper']:.1f}]",
            ha="right", va="bottom",
            fontsize=PLOT_FONT_SIZES["annotation"],
            color="black",
        )
        ax.axhline(3.0, color=VT_STONE, lw=0.85, ls=(0, (5, 3)),
                   label="+3 trained offset")
        ax.axhline(-3.0, color=VT_STONE, lw=0.9, ls=":", label="-3 trained")
        if metrics["entered"]:
            ax.axvline(metrics["time_to_enter"], color=VT_DARK_STONE,
                       lw=1.0, ls="-.", label="time to +/-0.5 band")
            ax.text(metrics["time_to_enter"], metrics["upper"],
                    f"  {metrics['time_to_enter']:.2f} s",
                    ha="left", va="bottom",
                    fontsize=PLOT_FONT_SIZES["annotation"],
                    color="black")
        ax.set_xlabel("Elapsed time (s)", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.set_ylabel("Predicted offset", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.set_title("COM demo streaming at +4 predicted offset",
                     fontsize=PLOT_FONT_SIZES["title"], fontweight="bold",
                     color=VT_MAROON)
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(facecolor=VT_LIGHT_STONE, edgecolor=VT_ORANGE,
                             linewidth=1.0, linestyle=":",
                             label="+/-0.5 target band"))
        labels.append("+/-0.5 target band")
        ax.legend(handles=handles, labels=labels,
                  fontsize=PLOT_FONT_SIZES["legend"], loc="best",
                  frameon=True, framealpha=0.92, edgecolor=VT_LIGHT_STONE)

        inset = ax.inset_axes([0.26, 0.1, 0.34, 0.32])
        for row in x_rows:
            inset.plot(t_common, row, color=VT_ORANGE, lw=0.55, alpha=0.18)
        inset.fill_between(
            t_common, x_mean - x_std, x_mean + x_std,
            color=VT_PALE_ORANGE, alpha=0.85, linewidth=0.0,
        )
        inset.plot(t_common, x_mean, color=VT_MAROON, lw=1.2)
        inset.axhspan(metrics["lower"], metrics["upper"],
                      color=VT_LIGHT_STONE, alpha=0.78)
        inset.axhline(metrics["lower"], color=VT_ORANGE, lw=0.75, ls=":")
        inset.axhline(metrics["upper"], color=VT_ORANGE, lw=0.75, ls=":")
        inset.axhline(STREAM_TARGET_OFFSET, color=VT_ORANGE, lw=0.9)
        if metrics["entered"]:
            inset.axvline(metrics["time_to_enter"], color=VT_DARK_STONE,
                          lw=0.8, ls="-.")
        inset.set_ylim(metrics["lower"] - 0.15, metrics["upper"] + 0.15)
        inset.set_xlim(t_common[0], t_common[-1])
        inset.set_title("target zoom", fontsize=PLOT_FONT_SIZES["annotation"],
                        color=VT_DARK_STONE)
        inset.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["annotation"],
                          width=0.5, length=2, pad=1)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved -> {out_path}")
    return metrics


def plot_streaming_readout_path(streams, out_path):
    if not streams:
        print("  Skipped streaming path plot because no stream data were available")
        return

    min_len = min(len(s["xy"]) for s in streams)
    if min_len <= 1:
        print("  Skipped streaming path plot because streams were too short")
        return

    xy_stack = np.stack([s["xy"][:min_len] for s in streams], axis=0)
    xy_mean = xy_stack.mean(axis=0)

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(4.6, 3.5))

        for s in streams:
            xy = s["xy"]
            ax.plot(xy[:, 0], xy[:, 1], color=VT_ORANGE, lw=0.75, alpha=0.22)

        ax.plot(xy_mean[:, 0], xy_mean[:, 1], color=VT_MAROON, lw=1.8,
                label="mean streaming path")
        ax.scatter(xy_mean[0, 0], xy_mean[0, 1], marker="o", s=34,
                   color=VT_STONE, edgecolor=VT_DARK_STONE, linewidths=0.5,
                   zorder=5, label="start")
        ax.scatter(xy_mean[-1, 0], xy_mean[-1, 1], marker="s", s=40,
                   color=VT_ORANGE, edgecolor=VT_DARK_STONE, linewidths=0.5,
                   zorder=6, label="final")

        for off, tgt in OFFSET_TARGETS.items():
            ax.scatter(tgt[0], tgt[1], marker="*", s=170,
                       color=OFFSET_COLORS[off], edgecolor=VT_DARK_STONE,
                       linewidths=0.6, zorder=7)
            ax.text(tgt[0], tgt[1] + 0.18, f"{off:+d}",
                    ha="center", va="bottom",
                    fontsize=PLOT_FONT_SIZES["class_label"],
                    fontweight="bold", color=OFFSET_COLORS[off])

        ax.axhline(0, color=VT_LIGHT_STONE, lw=0.7)
        ax.axvline(0, color=VT_LIGHT_STONE, lw=0.7)
        ax.set_xlim(-3.8, 4.6)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xlabel("Predicted offset", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.set_ylabel("Residual readout", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.set_title("Streaming readout path toward +4",
                     fontsize=PLOT_FONT_SIZES["title"], fontweight="bold",
                     color=VT_MAROON)
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        ax.legend(fontsize=PLOT_FONT_SIZES["legend"], loc="best",
                  frameon=True, framealpha=0.92, edgecolor=VT_LIGHT_STONE)

        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved -> {out_path}")


def plot_confusion_matrix_offset(cm_raw, class_order, train_acc, test_acc, out_path):
    vals = class_order
    row_sum = cm_raw.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_raw / row_sum, 0.0)

    cmap_count = LinearSegmentedColormap.from_list(
        "vt_maroon", ["#FFFFFF", VT_PALE_MAROON, "#C89AAE", VT_MAROON]
    )
    cmap_recall = LinearSegmentedColormap.from_list(
        "vt_orange", ["#FFFFFF", VT_PALE_ORANGE, "#F1B889", VT_ORANGE]
    )

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4),
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
            cb.set_label(cbar_label, labelpad=4,
                         fontsize=PLOT_FONT_SIZES["colorbar_label"])
            cb.ax.tick_params(labelsize=PLOT_FONT_SIZES["colorbar_tick"],
                              width=0.5, length=2, pad=2)
            cb.outline.set_linewidth(0.4)

            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels([f"{v:+d}" for v in vals], rotation=30, ha="right")
            ax.set_yticks(range(len(vals)))
            ax.set_yticklabels([f"{v:+d}" for v in vals])
            ax.set_xlabel("Predicted offset", fontsize=PLOT_FONT_SIZES["axis_label"])
            ax.set_ylabel("True offset", fontsize=PLOT_FONT_SIZES["axis_label"])
            ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])

            thresh = vm * 0.55
            for i in range(len(vals)):
                for j in range(len(vals)):
                    v = data[i, j]
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=PLOT_FONT_SIZES["matrix_cell"],
                            fontweight="bold",
                            color="white" if v > thresh else VT_DARK_STONE)

            for k in range(len(vals) + 1):
                ax.axhline(k - 0.5, color="white", lw=0.4)
                ax.axvline(k - 0.5, color="white", lw=0.4)

            ax.set_xlim(-0.5, len(vals) - 0.5)
            ax.set_ylim(len(vals) - 0.5, -0.5)

        axes[0].set_title("Count", fontsize=PLOT_FONT_SIZES["panel_title"])
        axes[1].set_title("Recall", fontsize=PLOT_FONT_SIZES["panel_title"])
        fig.suptitle(
            f"Bar-outer +4 confusion matrix  "
            f"(train acc = {train_acc*100:.0f}%, test acc = {test_acc*100:.0f}%)",
            fontsize=PLOT_FONT_SIZES["suptitle"], fontweight="bold",
            color=VT_MAROON, y=1.03,
        )
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved -> {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}")
    print("  COM Demo -- Bar-Outer +4 Dynamic-Summary Readout")
    print(f"{'=' * 72}")
    print("  Train samples:")
    summarise_samples(TRAIN_SAMPLES)
    print("  Test samples:")
    summarise_samples(TEST_SAMPLES)
    print(f"  Allowed targets   : {list(OFFSET_TARGETS.keys())}")
    print(f"  No -4 target      : {-4 not in OFFSET_TARGETS}")
    print(f"  Time window       : {T_START}-{T_END} s")
    print(f"  Ridge alpha       : {RIDGE_ALPHA}")
    print(f"  Normalize energy  : {NORMALIZE_ENERGY}\n")

    model, mu, std, raw_dim, summary_dim = fit_dynamic_summary_offset(TRAIN_SAMPLES)

    weight_path = out_dir / "bar_outer_dynamic_summary_ridge_weights.mat"
    savemat(weight_path, {
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
        "raw_dim": raw_dim,
        "normalize_energy": int(NORMALIZE_ENERGY),
        "valid_offsets": VALID_OFFSETS,
        "offset_targets_keys": np.array(list(OFFSET_TARGETS.keys()), dtype=np.int32),
        "offset_targets_values": np.vstack(list(OFFSET_TARGETS.values())),
    })
    print(f"  Saved weights -> {weight_path}")

    _, tr_true, tr_pred = run_dynamic_summary_offset(
        TRAIN_SAMPLES, model, mu, std, label="Training set", classes=ACTIVE_CLASSES
    )
    train_acc = float(np.mean(tr_true == tr_pred)) if len(tr_true) else 0.0
    print(f"\n  Train accuracy : {train_acc*100:.1f}%")

    results, te_true, te_pred = run_dynamic_summary_offset(
        TEST_SAMPLES, model, mu, std, label="Bar-outer +4 actual demo", classes=TEST_CLASSES
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
            digits=3,
            zero_division=0,
        ))

        cm = confusion_matrix(te_true, te_pred, labels=labels_sorted)
        plot_confusion_matrix_offset(
            cm, labels_sorted, train_acc, test_acc,
            out_dir / "bar_outer_confusion_matrix.pdf",
        )

    plot_trial_points_offset(
        results, train_acc, test_acc,
        out_dir / "bar_outer_readout_points.pdf",
    )

    streams = collect_streaming_readouts(
        TEST_SAMPLES, model, mu, std, classes=TEST_CLASSES
    )
    streaming_metrics = plot_streaming_offset(
        streams,
        out_dir / "bar_outer_streaming_readout.pdf",
    )
    if streaming_metrics is not None:
        if streaming_metrics["entered"]:
            print(
                f"  Time to enter +/-0.5 band around +4: "
                f"{streaming_metrics['time_to_enter']:.3f} s "
                f"(predicted offset="
                f"{streaming_metrics['predicted_offset_at_entry']:.3f}, "
                f"band=[{streaming_metrics['lower']:.3f}, "
                f"{streaming_metrics['upper']:.3f}])"
            )
        else:
            print(
                f"  Mean predicted offset did not enter the +/-0.5 band around +4 "
                f"(band=[{streaming_metrics['lower']:.3f}, "
                f"{streaming_metrics['upper']:.3f}])"
            )
        save_streaming_metrics(
            streaming_metrics,
            out_dir / "bar_outer_streaming_metrics.txt",
        )
    plot_streaming_readout_path(
        streams,
        out_dir / "bar_outer_streaming_readout_path.pdf",
    )

    print(f"\n  Raw state dim        : {raw_dim}")
    print(f"  Summary feature dim  : {summary_dim}")
    print(f"  Done. Outputs -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
