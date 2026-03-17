"""
Trajectory Failure-Mode Diagnostics
==================================

Uses the same data-loading style as the user's trajectory predictor.

Main goals:
1. Train the same Ridge autoregressive predictor.
2. Roll out on test samples.
3. Diagnose likely failure modes from raw trajectories:
   - phase mismatch / drift
   - frequency mismatch
   - amplitude mismatch
   - damping mismatch
   - marker-specific spatial mismatch
   - autoregressive accumulation

Outputs:
- time-series comparison for each marker/dimension
- lag-vs-time plots
- amplitude-envelope plots
- FFT spectrum comparison
- damping-envelope comparison
- per-marker summary CSV

Usage:
    python diagnose_trajectory_failure_modes.py
"""

from pathlib import Path
import csv
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "base": f"{ROOT_DIR}/soft_state_100g",
    # "near": f"{ROOT_DIR}/soft_state_100g_near",
}

COOR_DIR = "coor_0"

TARGET_MARKER_IDS = [4, 9, 13, 18]

TRAIN_SAMPLES = [
    ("base", "trajectories_sample_0.h5"),
    ("base", "trajectories_sample_1.h5"),
    # ("base", "trajectories_sample_2.h5"),
]

TEST_SAMPLES = [
    ("base", "trajectories_sample_2.h5"),
]

T_START        = 0.0
T_END          = 3.0
HISTORY_WINDOW = 15
RIDGE_ALPHA    = 1e-3

OUTPUT_DIR = f"{ROOT_DIR}/soft_state_100g/failure_mode_diagnostics"

# Sliding-window lag diagnostics
LAG_WINDOW_FRAMES = 25     # should be >= ~1 oscillation if possible
LAG_MAX_FRAMES    = 12     # max +/- lag searched

EPS = 1e-12

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
           "#56B4E9", "#D55E00", "#F0E442", "#000000"]

PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# ══════════════════════════════════════════════════════════════════════════════
# DATA I/O
# ══════════════════════════════════════════════════════════════════════════════

def resolve(dir_key, fname):
    return Path(DIRS[dir_key]) / COOR_DIR / fname


def load_h5(path):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]

    # same NaN filling style
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
    disp = pos - baseline
    return disp, pos, time


def slice_markers(disp, time, marker_ids):
    i0 = int(np.searchsorted(time, T_START, side="left"))
    i1 = (int(np.searchsorted(time, T_END, side="right"))
          if T_END is not None else len(time))
    cols = [c for m in marker_ids for c in (disp[i0:i1, m, 0], disp[i0:i1, m, 1])]
    return np.stack(cols, axis=1), time[i0:i1] - time[i0]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def fit_model(marker_ids):
    Xw_all, Yw_all = [], []
    for dir_key, fname in TRAIN_SAMPLES:
        p = resolve(dir_key, fname)
        if not p.exists():
            print(f"[WARN] Missing {p} — skipped")
            continue

        disp, _, time = load_h5(p)
        X, _ = slice_markers(disp, time, marker_ids)
        n = len(X) - HISTORY_WINDOW
        if n <= 0:
            continue

        Xw_all.append(np.stack([X[t:t+HISTORY_WINDOW].ravel() for t in range(n)]))
        Yw_all.append(X[HISTORY_WINDOW:])
        print(f"Train + {dir_key}/{fname}  T={len(X)}  pairs={n}")

    if not Xw_all:
        raise RuntimeError("No valid training window pairs found.")

    Xw = np.vstack(Xw_all)
    Yw = np.vstack(Yw_all)
    mu = Xw.mean(0)
    std = Xw.std(0)
    std[std < 1e-8] = 1.0

    model = Ridge(alpha=RIDGE_ALPHA).fit((Xw - mu) / std, Yw)
    print(f"Model fitted: X={Xw.shape}, Y={Yw.shape}")
    return model, mu, std


def predict_autoregressive(model, mu, std, X):
    n_predict = len(X) - HISTORY_WINDOW
    buf = X[:HISTORY_WINDOW].copy()
    preds = []

    for _ in range(n_predict):
        window = buf.ravel()[np.newaxis, :]
        y_hat = model.predict((window - mu) / std)[0]
        preds.append(y_hat)
        buf = np.vstack([buf[1:], y_hat])

    return np.array(preds), X[HISTORY_WINDOW:]


def predict_teacher_forced(model, mu, std, X):
    n_predict = len(X) - HISTORY_WINDOW
    preds = []
    for t in range(n_predict):
        window = X[t:t+HISTORY_WINDOW].ravel()[np.newaxis, :]
        y_hat = model.predict((window - mu) / std)[0]
        preds.append(y_hat)
    return np.array(preds), X[HISTORY_WINDOW:]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def moving_rms(x, win):
    if win <= 1:
        return np.sqrt(np.maximum(x**2, 0.0))
    pad = win // 2
    xp = np.pad(x**2, (pad, pad), mode="edge")
    kernel = np.ones(win) / win
    y = np.convolve(xp, kernel, mode="valid")
    return np.sqrt(np.maximum(y, 0.0))


def sliding_lag(true_sig, pred_sig, win=LAG_WINDOW_FRAMES, max_lag=LAG_MAX_FRAMES):
    """
    Sliding-window lag estimate from normalized cross-correlation.
    Positive lag means pred lags behind true.
    """
    n = len(true_sig)
    if n < win:
        return np.array([]), np.array([])

    centers = []
    best_lags = []

    for start in range(0, n - win + 1):
        a = true_sig[start:start+win]
        b = pred_sig[start:start+win]

        a = a - a.mean()
        b = b - b.mean()

        best_corr = -np.inf
        best_lag = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                aa = a[-lag:]
                bb = b[:len(b)+lag]
            elif lag > 0:
                aa = a[:len(a)-lag]
                bb = b[lag:]
            else:
                aa = a
                bb = b

            if len(aa) < max(5, win // 3):
                continue

            denom = (np.linalg.norm(aa) * np.linalg.norm(bb)) + EPS
            corr = float(np.dot(aa, bb) / denom)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        centers.append(start + win // 2)
        best_lags.append(best_lag)

    return np.array(centers), np.array(best_lags)


def dominant_frequency(sig, dt):
    """
    Dominant frequency from a Hann-windowed FFT magnitude, excluding DC.
    Also normalizes by the window sum for more comparable spectrum heights.
    """
    sig = np.asarray(sig, dtype=float)
    sig = sig - np.mean(sig)
    n = len(sig)

    if n < 8 or dt <= 0:
        return np.nan, None, None

    window = np.hanning(n)
    sig_w = sig * window

    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(sig_w)) / (np.sum(window) + 1e-12)

    if len(freqs) <= 1:
        return np.nan, freqs, spec

    spec[0] = 0.0
    idx = int(np.argmax(spec))
    return float(freqs[idx]), freqs, spec


def fit_log_decay(sig, dt, win=7):
    """
    Rough decay-rate estimate from smoothed envelope.
    Uses RMS envelope and linear fit to log(envelope).
    """
    env = moving_rms(sig, win=win)
    env = np.maximum(env, EPS)
    t = np.arange(len(sig)) * dt

    # Avoid overly tiny tail values
    valid = env > 0.1 * np.max(env)
    if np.sum(valid) < 5:
        valid = env > 0.02 * np.max(env)

    if np.sum(valid) < 5:
        return np.nan, env

    y = np.log(env[valid])
    x = t[valid]

    # y = b + s*x ; for exp(-alpha t), slope s = -alpha
    s, b = np.polyfit(x, y, 1)
    alpha = -float(s)
    return alpha, env


def classify_failure_mode(lag_abs_mean, lag_slope, freq_rel_err,
                          amp_ratio, decay_rel_err,
                          tf_rms, ar_rms):
    """
    Very rough heuristic label. This is only a helper, not ground truth.
    """
    # autoregressive amplification
    rollout_ratio = ar_rms / max(tf_rms, EPS)

    if rollout_ratio > 2.5 and tf_rms < ar_rms * 0.5:
        return "autoregressive_accumulation"

    if abs(lag_slope) > 0.05 or lag_abs_mean > 3.0:
        if freq_rel_err > 0.08:
            return "frequency_mismatch"
        return "phase_drift"

    if abs(np.log(max(amp_ratio, EPS))) > np.log(1.35):
        return "amplitude_mismatch"

    if decay_rel_err > 0.25:
        return "damping_mismatch"

    return "mixed_or_spatial"


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_time_series(ts, y_true, y_pred, y_tf, marker_ids, out_path):
    M = len(marker_ids)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(M, 2, figsize=(8.2, 2.2 * M),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.35})
        if M == 1:
            axes = axes[np.newaxis, :]

        for i, m in enumerate(marker_ids):
            col = PALETTE[i % len(PALETTE)]
            ix, iy = 2*i, 2*i+1
            for ax, j, name in [(axes[i, 0], ix, "x"), (axes[i, 1], iy, "y")]:
                ax.plot(ts, y_true[:, j], color="0.65", lw=1.1, label="true")
                ax.plot(ts, y_tf[:, j], color="#56B4E9", lw=1.0, ls="--", label="1-step")
                ax.plot(ts, y_pred[:, j], color=col, lw=1.0, label="autoregressive")
                ax.set_title(f"Marker {m} — {name}(t)")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("disp (px)")
                if i == 0 and name == "x":
                    ax.legend(frameon=True, framealpha=0.9)

        fig.suptitle("True vs one-step vs autoregressive time series", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_lag(ts, y_true, y_pred, marker_ids, dt, out_path):
    M = len(marker_ids)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(M, 2, figsize=(8.2, 2.2 * M),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.35})
        if M == 1:
            axes = axes[np.newaxis, :]

        for i, m in enumerate(marker_ids):
            ix, iy = 2*i, 2*i+1
            for ax, j, name in [(axes[i, 0], ix, "x"), (axes[i, 1], iy, "y")]:
                centers, lags = sliding_lag(y_true[:, j], y_pred[:, j])
                if len(centers):
                    ax.plot(ts[centers], lags * dt, lw=1.2)
                    ax.axhline(0.0, color="0.75", lw=0.8, ls="--")
                ax.set_title(f"Marker {m} — lag({name})")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("lag (s)")

        fig.suptitle("Sliding lag from cross-correlation", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_envelope(ts, y_true, y_pred, marker_ids, out_path):
    M = len(marker_ids)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(M, 2, figsize=(8.2, 2.2 * M),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.35})
        if M == 1:
            axes = axes[np.newaxis, :]

        for i, m in enumerate(marker_ids):
            col = PALETTE[i % len(PALETTE)]
            ix, iy = 2*i, 2*i+1
            for ax, j, name in [(axes[i, 0], ix, "x"), (axes[i, 1], iy, "y")]:
                env_t = moving_rms(y_true[:, j], 9)
                env_p = moving_rms(y_pred[:, j], 9)
                ax.plot(ts, env_t, color="0.65", lw=1.1, label="true env")
                ax.plot(ts, env_p, color=col, lw=1.1, label="pred env")
                ax.set_title(f"Marker {m} — envelope({name})")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("RMS env (px)")
                if i == 0 and name == "x":
                    ax.legend(frameon=True, framealpha=0.9)

        fig.suptitle("Amplitude envelope comparison", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_fft(y_true, y_pred, marker_ids, dt, out_path):
    M = len(marker_ids)
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(M, 2, figsize=(8.2, 2.2 * M),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.35})
        if M == 1:
            axes = axes[np.newaxis, :]

        for i, m in enumerate(marker_ids):
            col = PALETTE[i % len(PALETTE)]
            ix, iy = 2*i, 2*i+1
            for ax, j, name in [(axes[i, 0], ix, "x"), (axes[i, 1], iy, "y")]:
                f_t, freqs_t, spec_t = dominant_frequency(y_true[:, j], dt)
                f_p, freqs_p, spec_p = dominant_frequency(y_pred[:, j], dt)

                if freqs_t is not None:
                    ax.plot(freqs_t, spec_t, color="0.65", lw=1.1, label=f"true f={f_t:.2f} Hz")
                if freqs_p is not None:
                    ax.plot(freqs_p, spec_p, color=col, lw=1.0, label=f"pred f={f_p:.2f} Hz")

                ax.set_title(f"Marker {m} — FFT({name})")
                ax.set_xlabel("frequency (Hz)")
                ax.set_ylabel("|FFT|")
                if i == 0 and name == "x":
                    ax.legend(frameon=True, framealpha=0.9)

        fig.suptitle("Dominant frequency comparison", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def plot_scatter_summary(rows, out_path):
    """
    Scatter to visualize specialist-vs-average behavior:
    x = one-step RMS, y = autoregressive RMS
    color = marker id
    """
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        marker_to_color = {}
        for i, m in enumerate(sorted(set(r["marker"] for r in rows))):
            marker_to_color[m] = PALETTE[i % len(PALETTE)]

        for r in rows:
            ax.scatter(r["tf_rms"], r["ar_rms"],
                       color=marker_to_color[r["marker"]],
                       s=40, alpha=0.9)

            ax.annotate(f'{r["sample_label"]}-m{r["marker"]}',
                        (r["tf_rms"], r["ar_rms"]),
                        textcoords="offset points", xytext=(3, 3), fontsize=6)

        lim = max([max(r["tf_rms"], r["ar_rms"]) for r in rows] + [1.0])
        ax.plot([0, lim], [0, lim], color="0.7", ls="--", lw=1.0)
        ax.set_xlabel("1-step RMS error")
        ax.set_ylabel("Autoregressive RMS error")
        ax.set_title("1-step vs autoregressive error")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def summarize_marker(ts, y_true, y_pred, y_tf, marker_id, i_marker):
    dt = float(np.mean(np.diff(ts))) if len(ts) > 1 else np.nan
    out = {"marker": marker_id}

    for dim_name, j in [("x", 2*i_marker), ("y", 2*i_marker+1)]:
        true_sig = y_true[:, j]
        pred_sig = y_pred[:, j]
        tf_sig   = y_tf[:, j]

        # Errors
        ar_rms = float(np.sqrt(np.mean((pred_sig - true_sig)**2)))
        tf_rms = float(np.sqrt(np.mean((tf_sig - true_sig)**2)))

        # Lag trend
        centers, lags = sliding_lag(true_sig, pred_sig)
        if len(centers) >= 2:
            t_lag = ts[centers]
            lag_sec = lags * dt
            lag_abs_mean = float(np.mean(np.abs(lag_sec)))
            lag_slope = float(np.polyfit(t_lag, lag_sec, 1)[0])  # s lag / s time
        else:
            lag_abs_mean = np.nan
            lag_slope = np.nan

        # Frequencies
        f_true, _, _ = dominant_frequency(true_sig, dt)
        f_pred, _, _ = dominant_frequency(pred_sig, dt)
        if np.isfinite(f_true) and f_true > 0 and np.isfinite(f_pred):
            freq_rel_err = float(abs(f_pred - f_true) / f_true)
        else:
            freq_rel_err = np.nan

        # Envelope / amplitude
        env_true = moving_rms(true_sig, 9)
        env_pred = moving_rms(pred_sig, 9)
        amp_ratio = float((np.mean(env_pred) + EPS) / (np.mean(env_true) + EPS))

        # Damping
        alpha_true, _ = fit_log_decay(true_sig, dt, win=9)
        alpha_pred, _ = fit_log_decay(pred_sig, dt, win=9)
        if np.isfinite(alpha_true) and alpha_true > 0 and np.isfinite(alpha_pred):
            decay_rel_err = float(abs(alpha_pred - alpha_true) / alpha_true)
        else:
            decay_rel_err = np.nan

        label = classify_failure_mode(
            lag_abs_mean=0.0 if np.isnan(lag_abs_mean) else lag_abs_mean,
            lag_slope=0.0 if np.isnan(lag_slope) else lag_slope,
            freq_rel_err=0.0 if np.isnan(freq_rel_err) else freq_rel_err,
            amp_ratio=amp_ratio,
            decay_rel_err=0.0 if np.isnan(decay_rel_err) else decay_rel_err,
            tf_rms=tf_rms,
            ar_rms=ar_rms,
        )

        out[f"{dim_name}_tf_rms"] = tf_rms
        out[f"{dim_name}_ar_rms"] = ar_rms
        out[f"{dim_name}_lag_abs_mean_s"] = lag_abs_mean
        out[f"{dim_name}_lag_slope"] = lag_slope
        out[f"{dim_name}_f_true_hz"] = f_true
        out[f"{dim_name}_f_pred_hz"] = f_pred
        out[f"{dim_name}_freq_rel_err"] = freq_rel_err
        out[f"{dim_name}_amp_ratio"] = amp_ratio
        out[f"{dim_name}_alpha_true"] = alpha_true
        out[f"{dim_name}_alpha_pred"] = alpha_pred
        out[f"{dim_name}_decay_rel_err"] = decay_rel_err
        out[f"{dim_name}_failure_mode"] = label

    return out


def save_csv(rows, out_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker_ids = TARGET_MARKER_IDS
    model, mu, std = fit_model(marker_ids)

    csv_rows = []
    scatter_rows = []

    for dir_key, fname in TEST_SAMPLES:
        p = resolve(dir_key, fname)
        if not p.exists():
            print(f"[WARN] Missing {p} — skipped")
            continue

        sample_label = f"{dir_key}_{fname.replace('trajectories_', '').replace('.h5', '')}"
        print(f"\n=== Diagnosing {sample_label} ===")

        disp, pos_abs, time = load_h5(p)
        X, ts = slice_markers(disp, time, marker_ids)

        if len(X) <= HISTORY_WINDOW:
            print(f"[WARN] Too short: {sample_label}")
            continue

        y_pred, y_true = predict_autoregressive(model, mu, std, X)
        y_tf, _ = predict_teacher_forced(model, mu, std, X)
        ts_pred = ts[HISTORY_WINDOW:HISTORY_WINDOW + len(y_pred)]
        dt = float(np.mean(np.diff(ts_pred)))

        # Figures
        plot_time_series(
            ts_pred, y_true, y_pred, y_tf, marker_ids,
            out_dir / f"{sample_label}_time_series.png"
        )
        plot_lag(
            ts_pred, y_true, y_pred, marker_ids, dt,
            out_dir / f"{sample_label}_lag.png"
        )
        plot_envelope(
            ts_pred, y_true, y_pred, marker_ids,
            out_dir / f"{sample_label}_envelope.png"
        )
        plot_fft(
            y_true, y_pred, marker_ids, dt,
            out_dir / f"{sample_label}_fft.png"
        )

        # Per-marker summary
        for i, m in enumerate(marker_ids):
            row = summarize_marker(ts_pred, y_true, y_pred, y_tf, m, i)
            row["sample_label"] = sample_label
            csv_rows.append(row)

            scatter_rows.append({
                "sample_label": sample_label,
                "marker": m,
                "tf_rms": 0.5 * (row["x_tf_rms"] + row["y_tf_rms"]),
                "ar_rms": 0.5 * (row["x_ar_rms"] + row["y_ar_rms"]),
            })

            print(
                f"Marker {m}: "
                f"x={row['x_failure_mode']}, "
                f"y={row['y_failure_mode']}, "
                f"AR RMS≈{0.5*(row['x_ar_rms']+row['y_ar_rms']):.3f}"
            )

    save_csv(csv_rows, out_dir / "failure_mode_summary.csv")
    if scatter_rows:
        plot_scatter_summary(scatter_rows, out_dir / "one_step_vs_autoregressive.png")

    print(f"\nSaved diagnostics to:\n{out_dir.resolve()}")


if __name__ == "__main__":
    main()
