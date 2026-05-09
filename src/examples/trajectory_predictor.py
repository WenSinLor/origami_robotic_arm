"""
Marker Trajectory Predictor — Multi-File Training + Autoregressive Test
========================================================================
Training  : multiple files, sliding history window pairs (same as original)
Test      : autoregressive — once a frame is predicted, it feeds back in
            as input for the next step. No real frames used after the seed.

Seed = first HISTORY_WINDOW real frames of the test file.

  Step 1: real[0..14]               → predict f̂15
  Step 2: real[1..14] + f̂15        → predict f̂16
  Step 3: real[2..14] + f̂15 + f̂16 → predict f̂17
  ...after HISTORY_WINDOW steps, window contains zero real frames.

Usage:  python trajectory_predictor.py
"""

from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"
ROOT_DIR = "/home/wensin/Documents/origami_robotic_arm/data"

DIRS = {
    "base": f"{ROOT_DIR}/soft_state_100g",
    # "near": f"{ROOT_DIR}/soft_state_100g_near",
}

COOR_DIR = "coor_0"

TARGET_MARKER_IDS = [4, 9, 13, 18]

TRAIN_SAMPLES = [
    ("base", "trajectories_sample_0.h5"),
    ("base", "trajectories_sample_1.h5"),
    ("base", "trajectories_sample_2.h5"),
    ("base", "trajectories_sample_3.h5"),
    # ("base", "trajectories_sample_4.h5"),
    # ("base", "trajectories_sample_5.h5"),
    # ("base", "trajectories_sample_6.h5"),
    # ("base", "trajectories_sample_7.h5"),
]

TEST_SAMPLES = [
    ("base", "trajectories_sample_7.h5"),
]

T_START        = 0.0
T_END          = 3.0    # set to None to use full recording
HISTORY_WINDOW = 15     # seed frames; also the input feature width
RIDGE_ALPHA    = 1e-3

OUTPUT_DIR = f"{ROOT_DIR}/soft_state_100g"

VT_MAROON = "#861F41"
VT_ORANGE = "#E5751F"
VT_STONE = "#75787B"
VT_DARK_STONE = "#54585A"
VT_LIGHT_STONE = "#D7D2CB"
VT_PALE_MAROON = "#F2E8ED"
VT_GOLD = "#B3A369"
VT_TEAL = "#508590"

PLOT_FONT_SIZES = {
    "base": 9,
    "axis_label": 9,
    "tick": 8,
    "legend": 7.5,
    "title": 10,
    "panel_title": 9,
    "suptitle": 10,
    "annotation": 7,
    "marker_label": 8,
    "legend_title": 7,
}

NATURE_RC = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          PLOT_FONT_SIZES["base"],
    "axes.titlesize":     PLOT_FONT_SIZES["panel_title"],
    "axes.labelsize":     PLOT_FONT_SIZES["axis_label"],
    "xtick.labelsize":    PLOT_FONT_SIZES["tick"],
    "ytick.labelsize":    PLOT_FONT_SIZES["tick"],
    "legend.fontsize":    PLOT_FONT_SIZES["legend"],
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

PALETTE = [VT_MAROON, VT_ORANGE, VT_TEAL, VT_GOLD,
           VT_STONE, "#C64600", VT_DARK_STONE, "#2C2A29"]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA I/O
# ══════════════════════════════════════════════════════════════════════════════

def resolve(dir_key, fname):
    return Path(DIRS[dir_key]) / COOR_DIR / fname


def load_h5(path):
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
                col[:idx[0]] = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i+1] - idx[i] > 1:
                        col[idx[i]+1:idx[i+1]] = col[idx[i]]
    baseline = pos[:1].mean(axis=0, keepdims=True)
    return pos - baseline, pos, time


def slice_markers(disp, time, marker_ids):
    i0 = int(np.searchsorted(time, T_START, side="left"))
    i1 = (int(np.searchsorted(time, T_END, side="right"))
          if T_END is not None else len(time))
    cols = [c for m in marker_ids
            for c in (disp[i0:i1, m, 0], disp[i0:i1, m, 1])]
    return np.stack(cols, axis=1), time[i0:i1] - time[i0]


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL — train once on all training files
# ══════════════════════════════════════════════════════════════════════════════

def fit_model(marker_ids):
    """
    Build sliding window pairs from all training files and fit Ridge once.
    Input : (HISTORY_WINDOW * M*2,)  — H stacked frames flattened
    Output: (M*2,)                   — next frame
    """
    Xw_all, Yw_all = [], []
    for dir_key, fname in TRAIN_SAMPLES:
        p = resolve(dir_key, fname)
        if not p.exists():
            print(f"  [WARN] Missing {p} — skipped"); continue
        disp, _, time = load_h5(p)
        X, _ = slice_markers(disp, time, marker_ids)
        n = len(X) - HISTORY_WINDOW
        if n <= 0: continue
        Xw_all.append(
            np.stack([X[t:t+HISTORY_WINDOW].ravel() for t in range(n)]))
        Yw_all.append(X[HISTORY_WINDOW:])
        print(f"  + {dir_key}/{fname}  T={len(X)}  pairs={n}")

    Xw = np.vstack(Xw_all)
    Yw = np.vstack(Yw_all)
    mu  = Xw.mean(0)
    std = Xw.std(0);  std[std < 1e-8] = 1.0
    model = Ridge(alpha=RIDGE_ALPHA).fit((Xw - mu) / std, Yw)
    print(f"  Model fitted  {Xw.shape} → {Yw.shape}")
    return model, mu, std


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOREGRESSIVE PREDICTION — seed with real frames, then self-feed
# ══════════════════════════════════════════════════════════════════════════════

def predict_autoregressive(model, mu, std, X):
    """
    Seed with first HISTORY_WINDOW real frames, then roll forward
    using predicted frames as input. No real frames used after the seed.

    Returns y_pred (T - HISTORY_WINDOW, M*2)
            y_true (T - HISTORY_WINDOW, M*2)
    """
    n_predict = len(X) - HISTORY_WINDOW
    buf = X[:HISTORY_WINDOW].copy()   # (HISTORY_WINDOW, M*2) — real seed
    preds = []

    for _ in range(n_predict):
        window = buf.ravel()[np.newaxis]
        y_hat  = model.predict((window - mu) / std)[0]
        preds.append(y_hat)
        # Slide: drop oldest frame, append prediction
        buf = np.vstack([buf[1:], y_hat])

    return np.array(preds), X[HISTORY_WINDOW:]


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_pred, y_true, marker_ids):
    stats = {}
    for i, m in enumerate(marker_ids):
        ix, iy = 2*i, 2*i+1
        dx = y_pred[:, ix] - y_true[:, ix]
        dy = y_pred[:, iy] - y_true[:, iy]
        e  = np.sqrt(dx**2 + dy**2)
        ss_res = float(np.sum(dx**2 + dy**2))
        ss_tot = float(np.sum(
            (y_true[:, ix] - y_true[:, ix].mean())**2 +
            (y_true[:, iy] - y_true[:, iy].mean())**2))
        stats[m] = dict(
            r2_2d = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            rms   = float(np.sqrt(np.mean(e**2))),
            mean  = float(e.mean()),
            peak  = float(e.max()),
        )
        print(f"  Marker {m}: R²(2D)={stats[m]['r2_2d']:.4f}  "
              f"RMSE={stats[m]['rms']:.3f}px  "
              f"mean={stats[m]['mean']:.3f}px  "
              f"peak={stats[m]['peak']:.3f}px")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_absolute(label, pos_abs, time, y_pred, y_true,
                  marker_ids, stats, out_path):
    i0 = int(np.searchsorted(time, T_START, side="left"))
    T_pred = len(y_pred)
    abs_start = i0 + HISTORY_WINDOW

    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(6.0, 7.0))

        for i, m in enumerate(marker_ids):
            col  = PALETTE[i % len(PALETTE)]
            ix, iy = 2*i, 2*i+1

            # Seed portion (faint dotted)
            seed_x = pos_abs[i0:i0 + HISTORY_WINDOW, m, 0]
            seed_y = pos_abs[i0:i0 + HISTORY_WINDOW, m, 1]
            ax.plot(seed_x, seed_y,
                    color=col, lw=0.8, ls=":", alpha=0.32, zorder=2)

            # True test portion (grey dotted)
            true_x = pos_abs[abs_start:abs_start + T_pred, m, 0]
            true_y = pos_abs[abs_start:abs_start + T_pred, m, 1]
            ax.plot(true_x, true_y,
                    color=VT_STONE, lw=1.1, ls=":", alpha=0.9, zorder=3)

            # Predicted (solid colour)
            split_abs_x = float(pos_abs[abs_start, m, 0])
            split_abs_y = float(pos_abs[abs_start, m, 1])
            pred_x = split_abs_x + (y_pred[:, ix] - y_pred[0, ix])
            pred_y = split_abs_y + (y_pred[:, iy] - y_pred[0, iy])
            ax.plot(pred_x, pred_y,
                    color=col, lw=1.4, alpha=0.9, zorder=4,
                    label=(f"Marker {m}  "
                           f"R²={stats[m]['r2_2d']:.4f}  "
                           f"RMSE={stats[m]['rms']:.2f}px"))

            # Start / end on true test portion
            ax.scatter(true_x[0],  true_y[0],  marker="o", s=30,
                       color=col, zorder=7, edgecolors=VT_DARK_STONE, lw=0.6)
            ax.scatter(true_x[-1], true_y[-1], marker="s", s=30,
                       color=col, zorder=7, edgecolors=VT_DARK_STONE, lw=0.6)
            ax.annotate("start", xy=(true_x[0], true_y[0]),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=PLOT_FONT_SIZES["annotation"],
                        color=VT_DARK_STONE, ha="center", va="bottom")
            ax.annotate("end", xy=(true_x[-1], true_y[-1]),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=PLOT_FONT_SIZES["annotation"],
                        color=VT_DARK_STONE, ha="center", va="bottom")

            # Marker label
            ax.text(float(np.nanmean(true_x)), float(np.nanmean(true_y)),
                    str(m), fontsize=PLOT_FONT_SIZES["marker_label"],
                    fontweight="bold", color=col,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              alpha=0.75, ec=col, lw=0.6), zorder=6)

        ax.invert_yaxis()
        ax.set_xlabel("x (px)", fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.set_ylabel("y (px)  [video frame coords]",
                      fontsize=PLOT_FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="lower left", fontsize=PLOT_FONT_SIZES["legend"],
                  frameon=True, framealpha=0.94, edgecolor=VT_LIGHT_STONE,
                  title=f"faint=seed ({HISTORY_WINDOW} fr)  "
                        f"grey·=true  solid=predicted",
                  title_fontsize=PLOT_FONT_SIZES["legend_title"])
        fig.suptitle(f"Trajectory prediction — {label}",
                     fontsize=PLOT_FONT_SIZES["suptitle"], fontweight="bold",
                     color=VT_MAROON)
        fig.tight_layout()
        fig.savefig(str(out_path))
        plt.close(fig)
    print(f"  Saved → {out_path}")



# ══════════════════════════════════════════════════════════════════════════════
#  PHASE PLOT — predicted vs true displacement over time
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase(label, y_pred, y_true, ts, marker_ids, stats, out_path):
    """
    For each marker: plot x(t) and y(t) predicted vs true side by side.
    Reveals phase shift, amplitude error, and drift over time.
    """
    M = len(marker_ids)
    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(M, 2, figsize=(7.0, 2.0 * M),
                                  gridspec_kw={"wspace": 0.4, "hspace": 0.55})
        if M == 1:
            axes = axes[np.newaxis, :]

        for i, m in enumerate(marker_ids):
            col  = PALETTE[i % len(PALETTE)]
            ix, iy = 2*i, 2*i+1

            for j, (dim, label_dim) in enumerate([("x", ix), ("y", iy)]):
                ax = axes[i, j]
                ax.plot(ts, y_true[:, label_dim],
                        color=VT_STONE, lw=1.0, label="true", zorder=3)
                ax.plot(ts, y_pred[:, label_dim],
                        color=col, lw=1.0, label="pred", alpha=0.9, zorder=4)
                ax.set_xlabel("time (s)", fontsize=PLOT_FONT_SIZES["axis_label"])
                ax.set_ylabel(f"{dim} displacement (px)",
                              fontsize=PLOT_FONT_SIZES["axis_label"])
                ax.set_title(f"Marker {m} — {dim}(t)",
                             fontsize=PLOT_FONT_SIZES["panel_title"],
                             color=VT_DARK_STONE)
                ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
                if i == 0 and j == 0:
                    ax.legend(fontsize=PLOT_FONT_SIZES["legend"],
                              frameon=True, framealpha=0.92,
                              edgecolor=VT_LIGHT_STONE)

        fig.suptitle(f"Phase alignment — {label}",
                     fontsize=PLOT_FONT_SIZES["suptitle"], fontweight="bold",
                     color=VT_MAROON)
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
    marker_ids = TARGET_MARKER_IDS

    print(f"\n  Fitting model on {len(TRAIN_SAMPLES)} training files...")
    model, mu, std = fit_model(marker_ids)

    print(f"\n  Evaluating on test files...")
    for dir_key, fname in TEST_SAMPLES:
        p = resolve(dir_key, fname)
        if not p.exists():
            print(f"  [WARN] Missing {p} — skipped"); continue

        label = f"{dir_key}/{fname.replace('trajectories_','').replace('.h5','')}"
        disp, pos_abs, time = load_h5(p)
        X, ts = slice_markers(disp, time, marker_ids)

        if len(X) <= HISTORY_WINDOW:
            print(f"  [WARN] {label} too short — skipped"); continue

        print(f"\n  ── {label}  "
              f"(seed={HISTORY_WINDOW} real frames, "
              f"predict={len(X) - HISTORY_WINDOW} frames autoregressively)")

        y_pred, y_true = predict_autoregressive(model, mu, std, X)

        print(f"  Metrics:")
        stats = compute_metrics(y_pred, y_true, marker_ids)

        plot_absolute(label, pos_abs, time, y_pred, y_true,
                      marker_ids, stats,
                      out_dir / f"traj_{label.replace('/', '_')}.pdf")

        ts_pred = ts[HISTORY_WINDOW:HISTORY_WINDOW + len(y_pred)]
        plot_phase(label, y_pred, y_true, ts_pred, marker_ids, stats,
                   out_dir / f"phase_{label.replace('/', '_')}.pdf")

    print(f"\n  Done.  Outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
