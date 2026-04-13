"""
improved_modal_channel_validation_for_origami.py
================================================

Improved validation of trajectory mismatch between two conditions.

Adds:
    1) phase-only channel
    2) fit-quality metrics (R^2, NRMSE)
    3) filtering of poor fits
    4) mode matching by nearest frequency
    5) cleaner summary outputs

Data structure expected:
ROOT_DIR/
    soft_state_100g/
        coor_0/
            trajectories_sample_0.h5
            ...
        coor_1/
        coor_2/
        coor_3/

H5 structure:
    time_series/nodes/positions
    time_series/time
"""

from pathlib import Path
import re
import json
import warnings
from itertools import permutations

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.signal import periodogram, hilbert, find_peaks


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

REF_CONDITION = "soft_state_100g"
CMP_CONDITION = "soft_state_100g_near"

COOR_DIRS = ["coor_0", "coor_1", "coor_2", "coor_3"]

SAMPLE_IDS = None
# SAMPLE_IDS = list(range(12))

T_START = 0.0
T_END   = 3.0

BASELINE_FRAMES = 1
EXCLUDE_MARKERS = []

N_MODES = 2

# Fit quality thresholds for "trusted" decomposition
MIN_R2 = 0.75
MAX_NRMSE = 0.45

OUTPUT_DIR = f"{ROOT_DIR}/modal_validation_improved__{REF_CONDITION}__vs__{CMP_CONDITION}"


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

EPS = 1e-12


def sample_id_from_name(fname: str):
    m = re.search(r"sample_(\d+)", fname)
    return int(m.group(1)) if m else None


def file_for_sample(condition: str, coor_dir: str, sample_id: int) -> Path:
    return Path(ROOT_DIR) / condition / coor_dir / f"trajectories_sample_{sample_id}.h5"


def available_sample_ids(condition: str, coor_dir: str):
    folder = Path(ROOT_DIR) / condition / coor_dir
    if not folder.exists():
        return []
    ids = []
    for p in folder.glob("trajectories_sample_*.h5"):
        sid = sample_id_from_name(p.name)
        if sid is not None:
            ids.append(sid)
    return sorted(ids)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_h5(path: Path, baseline_frames: int = 1):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]

    for n in range(pos.shape[1]):
        for ax in range(2):
            col = pos[:, n, ax]
            nan_mask = np.isnan(col)
            if nan_mask.all():
                col[:] = 0.0
            elif nan_mask.any():
                idx = np.where(~nan_mask)[0]
                col[:idx[0]]  = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx) - 1):
                    if idx[i+1] - idx[i] > 1:
                        col[idx[i]+1:idx[i+1]] = col[idx[i]]

    baseline = pos[:min(baseline_frames, pos.shape[0])].mean(axis=0, keepdims=True)
    return pos - baseline, time


def extract_features(disp, time, t_start, t_end, exclude_markers):
    i0 = int(np.searchsorted(time, t_start, side="left"))
    i1 = int(np.searchsorted(time, t_end, side="right")) if t_end is not None else len(time)
    X  = disp[i0:i1].reshape(i1 - i0, -1)
    ts = time[i0:i1] - time[i0]

    if exclude_markers:
        N = X.shape[1] // 2
        keep = [c for n in range(N) if n not in exclude_markers for c in (2*n, 2*n+1)]
        X = X[:, keep]

    return X, ts


# ══════════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def dominant_scalar_signal(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    s = U[:, 0] * S[0]
    idx = np.argmax(np.abs(s))
    if s[idx] < 0:
        s = -s
    return s


# ══════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════

def damped_sum(t: np.ndarray, params: np.ndarray, n_modes: int) -> np.ndarray:
    y = np.full_like(t, params[0], dtype=float)
    k = 1
    for _ in range(n_modes):
        A     = params[k + 0]
        sigma = params[k + 1]
        f     = params[k + 2]
        phi   = params[k + 3]
        y += A * np.exp(-sigma * t) * np.cos(2*np.pi*f*t + phi)
        k += 4
    return y


def estimate_initial_params(y: np.ndarray, dt: float, n_modes: int) -> np.ndarray:
    fs = 1.0 / dt
    t = np.arange(len(y)) * dt

    offset0 = float(np.mean(y[-max(5, len(y)//10):]))

    freqs, power = periodogram(y - np.mean(y), fs=fs)
    valid = (freqs > 0.1) & (freqs < fs / 3)
    freqs = freqs[valid]
    power = power[valid]

    peak_freqs = []
    if len(freqs):
        peaks, _ = find_peaks(power, distance=max(1, len(power)//20))
        if len(peaks):
            order = peaks[np.argsort(power[peaks])[::-1]]
            peak_freqs = [float(freqs[i]) for i in order[:n_modes]]
        else:
            peak_freqs = [float(freqs[np.argmax(power)])]

    while len(peak_freqs) < n_modes:
        peak_freqs.append(peak_freqs[-1] * 1.4 if peak_freqs else 1.0)

    env = np.abs(hilbert(y - offset0)) + EPS
    m = max(10, len(t)//2)
    coeff = np.polyfit(t[:m], np.log(env[:m]), 1)
    sigma0 = max(0.05, float(-coeff[0]))

    A0 = np.std(y) * 1.5
    p0 = [offset0]
    for i in range(n_modes):
        p0.extend([A0 / (i + 1), sigma0, peak_freqs[i], 0.0])

    return np.array(p0, dtype=float)


def fit_damped_modes(y: np.ndarray, dt: float, n_modes: int):
    t = np.arange(len(y)) * dt
    fs = 1.0 / dt
    p0 = estimate_initial_params(y, dt, n_modes)

    lb = [-np.inf]
    ub = [ np.inf]
    for _ in range(n_modes):
        lb.extend([-np.inf, 0.0, 0.05, -4*np.pi])
        ub.extend([ np.inf, 20.0, fs/2.5,  4*np.pi])

    lb = np.array(lb, float)
    ub = np.array(ub, float)

    def residual(p):
        return damped_sum(t, p, n_modes) - y

    res = least_squares(
        residual,
        x0=p0,
        bounds=(lb, ub),
        max_nfev=12000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    p = res.x
    yhat = damped_sum(t, p, n_modes)
    return p, yhat


def params_to_mode_array(p: np.ndarray, n_modes: int) -> np.ndarray:
    """
    Returns shape (n_modes, 4):
    [A, sigma, freq, phi]
    """
    out = []
    k = 1
    for _ in range(n_modes):
        out.append(p[k:k+4].copy())
        k += 4
    return np.array(out, dtype=float)


def rebuild_from_modes(offset: float, modes: np.ndarray) -> np.ndarray:
    p = [offset]
    for row in modes:
        p.extend(row.tolist())
    return np.array(p, dtype=float)


def wrap_phase(x):
    return (x + np.pi) % (2*np.pi) - np.pi


def match_modes_by_frequency(p_ref: np.ndarray, p_cmp: np.ndarray, n_modes: int):
    """
    Reorder comparison modes to best match reference modes by frequency proximity.
    Small phase penalty added as tie-breaker.
    """
    ref_modes = params_to_mode_array(p_ref, n_modes)
    cmp_modes = params_to_mode_array(p_cmp, n_modes)

    best_perm = None
    best_cost = np.inf
    for perm in permutations(range(n_modes)):
        cost = 0.0
        for i, j in enumerate(perm):
            df = ref_modes[i, 2] - cmp_modes[j, 2]
            dphi = wrap_phase(ref_modes[i, 3] - cmp_modes[j, 3])
            cost += (df ** 2) + 0.05 * (dphi ** 2)
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    cmp_modes_reordered = cmp_modes[list(best_perm)]
    p_cmp_reordered = rebuild_from_modes(p_cmp[0], cmp_modes_reordered)
    return p_ref.copy(), p_cmp_reordered.copy()


def replace_channel(
    p_ref: np.ndarray,
    p_cmp: np.ndarray,
    n_modes: int,
    *,
    use_amp=False,
    use_sigma=False,
    use_freq=False,
    use_phase=False
):
    ref_modes = params_to_mode_array(p_ref, n_modes)
    cmp_modes = params_to_mode_array(p_cmp, n_modes)

    new_modes = ref_modes.copy()
    if use_amp:
        new_modes[:, 0] = cmp_modes[:, 0]
    if use_sigma:
        new_modes[:, 1] = cmp_modes[:, 1]
    if use_freq:
        new_modes[:, 2] = cmp_modes[:, 2]
    if use_phase:
        new_modes[:, 3] = cmp_modes[:, 3]

    return rebuild_from_modes(p_cmp[0], new_modes)


# ══════════════════════════════════════════════════════════════════════
# FIT QUALITY
# ══════════════════════════════════════════════════════════════════════

def compute_fit_metrics(y: np.ndarray, yhat: np.ndarray):
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y))**2)) + EPS
    r2 = 1.0 - sse / sst

    yrng = float(np.max(y) - np.min(y)) + EPS
    rmse = float(np.sqrt(np.mean(resid**2)))
    nrmse = rmse / yrng

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
    }


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze_pair(ref_path: Path, cmp_path: Path, coor_dir: str, sample_id: int):
    disp_ref, time_ref = load_h5(ref_path, baseline_frames=BASELINE_FRAMES)
    disp_cmp, time_cmp = load_h5(cmp_path, baseline_frames=BASELINE_FRAMES)

    Xr, tsr = extract_features(disp_ref, time_ref, T_START, T_END, EXCLUDE_MARKERS)
    Xc, tsc = extract_features(disp_cmp, time_cmp, T_START, T_END, EXCLUDE_MARKERS)

    n = min(len(tsr), len(tsc))
    Xr = Xr[:n]
    Xc = Xc[:n]
    t  = tsr[:n]
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1/60

    sr = dominant_scalar_signal(Xr)
    sc = dominant_scalar_signal(Xc)

    pr_raw, sr_hat = fit_damped_modes(sr, dt, N_MODES)
    pc_raw, sc_hat = fit_damped_modes(sc, dt, N_MODES)

    fit_ref = compute_fit_metrics(sr, sr_hat)
    fit_cmp = compute_fit_metrics(sc, sc_hat)

    pr, pc = match_modes_by_frequency(pr_raw, pc_raw, N_MODES)

    # Recompute fitted signals after matching order
    sr_hat = damped_sum(t, pr, N_MODES)
    sc_hat = damped_sum(t, pc, N_MODES)

    p_amp   = replace_channel(pr, pc, N_MODES, use_amp=True)
    p_freq  = replace_channel(pr, pc, N_MODES, use_freq=True)
    p_damp  = replace_channel(pr, pc, N_MODES, use_sigma=True)
    p_phase = replace_channel(pr, pc, N_MODES, use_phase=True)

    s_amp   = damped_sum(t, p_amp,   N_MODES)
    s_freq  = damped_sum(t, p_freq,  N_MODES)
    s_damp  = damped_sum(t, p_damp,  N_MODES)
    s_phase = damped_sum(t, p_phase, N_MODES)

    delta_total = sc_hat - sr_hat
    delta_amp   = s_amp   - sr_hat
    delta_freq  = s_freq  - sr_hat
    delta_damp  = s_damp  - sr_hat
    delta_phase = s_phase - sr_hat

    total_norm = np.linalg.norm(delta_total) + EPS

    ref_modes = params_to_mode_array(pr, N_MODES)
    cmp_modes = params_to_mode_array(pc, N_MODES)

    trusted = (
        fit_ref["r2"] >= MIN_R2 and
        fit_cmp["r2"] >= MIN_R2 and
        fit_ref["nrmse"] <= MAX_NRMSE and
        fit_cmp["nrmse"] <= MAX_NRMSE
    )

    result = {
        "coor_dir": coor_dir,
        "sample_id": sample_id,
        "ref_file": str(ref_path),
        "cmp_file": str(cmp_path),

        "ref_dom_amp":   float(ref_modes[0, 0]),
        "cmp_dom_amp":   float(cmp_modes[0, 0]),
        "ref_dom_sigma": float(ref_modes[0, 1]),
        "cmp_dom_sigma": float(cmp_modes[0, 1]),
        "ref_dom_freq_hz": float(ref_modes[0, 2]),
        "cmp_dom_freq_hz": float(cmp_modes[0, 2]),
        "ref_dom_phase": float(ref_modes[0, 3]),
        "cmp_dom_phase": float(cmp_modes[0, 3]),

        "dom_amp_shift":      float(cmp_modes[0, 0] - ref_modes[0, 0]),
        "dom_sigma_shift":    float(cmp_modes[0, 1] - ref_modes[0, 1]),
        "dom_freq_shift_hz":  float(cmp_modes[0, 2] - ref_modes[0, 2]),
        "dom_phase_shift":    float(wrap_phase(cmp_modes[0, 3] - ref_modes[0, 3])),

        "ref_r2": fit_ref["r2"],
        "cmp_r2": fit_cmp["r2"],
        "ref_nrmse": fit_ref["nrmse"],
        "cmp_nrmse": fit_cmp["nrmse"],
        "trusted": bool(trusted),

        "ref_rms": float(np.sqrt(np.mean(sr**2))),
        "cmp_rms": float(np.sqrt(np.mean(sc**2))),
        "rms_ratio_cmp_over_ref": float(np.sqrt(np.mean(sc**2)) / (np.sqrt(np.mean(sr**2)) + EPS)),

        "total_diff_norm": float(total_norm),
        "amp_only_norm":   float(np.linalg.norm(delta_amp)),
        "freq_only_norm":  float(np.linalg.norm(delta_freq)),
        "damp_only_norm":  float(np.linalg.norm(delta_damp)),
        "phase_only_norm": float(np.linalg.norm(delta_phase)),

        "amp_ratio_vs_total":   float(np.linalg.norm(delta_amp)   / total_norm),
        "freq_ratio_vs_total":  float(np.linalg.norm(delta_freq)  / total_norm),
        "damp_ratio_vs_total":  float(np.linalg.norm(delta_damp)  / total_norm),
        "phase_ratio_vs_total": float(np.linalg.norm(delta_phase) / total_norm),
    }

    extras = {
        "t": t,
        "sr": sr,
        "sc": sc,
        "sr_hat": sr_hat,
        "sc_hat": sc_hat,
        "s_amp": s_amp,
        "s_freq": s_freq,
        "s_damp": s_damp,
        "s_phase": s_phase,
        "delta_total": delta_total,
        "delta_amp": delta_amp,
        "delta_freq": delta_freq,
        "delta_damp": delta_damp,
        "delta_phase": delta_phase,
    }

    return result, extras


# ══════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════

def save_pair_plot(out_dir, result, ex):
    pair_name = f"{result['coor_dir']}__sample_{result['sample_id']}"
    plot_dir = Path(out_dir) / "pair_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    t = ex["t"]
    sr = ex["sr"]
    sc = ex["sc"]
    sr_hat = ex["sr_hat"]
    sc_hat = ex["sc_hat"]

    plt.figure(figsize=(12, 11))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, sr, label="reference raw")
    ax1.plot(t, sc, label="comparison raw", alpha=0.85)
    ax1.plot(t, sr_hat, "--", label="reference fit")
    ax1.plot(t, sc_hat, "--", label="comparison fit")
    ax1.set_title(
        f"{pair_name}\n"
        f"dom f: {result['ref_dom_freq_hz']:.3f} → {result['cmp_dom_freq_hz']:.3f} Hz, "
        f"dom sigma: {result['ref_dom_sigma']:.3f} → {result['cmp_dom_sigma']:.3f}, "
        f"dom phase shift: {result['dom_phase_shift']:.3f} rad\n"
        f"R² ref={result['ref_r2']:.3f}, cmp={result['cmp_r2']:.3f}, "
        f"trusted={result['trusted']}"
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Signal")
    ax1.legend()

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(t, ex["delta_total"], label="total fitted difference")
    ax2.plot(t, ex["delta_amp"], label="amp-only")
    ax2.plot(t, ex["delta_freq"], label="freq-only")
    ax2.plot(t, ex["delta_damp"], label="damp-only")
    ax2.plot(t, ex["delta_phase"], label="phase-only")
    ax2.set_title(
        f"ratios | amp={result['amp_ratio_vs_total']:.3f}, "
        f"freq={result['freq_ratio_vs_total']:.3f}, "
        f"damp={result['damp_ratio_vs_total']:.3f}, "
        f"phase={result['phase_ratio_vs_total']:.3f}"
    )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Difference")
    ax2.legend()

    ax3 = plt.subplot(4, 1, 3)
    fr, Pr = periodogram(sr - np.mean(sr), fs=1/(t[1]-t[0]))
    fc, Pc = periodogram(sc - np.mean(sc), fs=1/(t[1]-t[0]))
    ax3.plot(fr, Pr, label="reference PSD")
    ax3.plot(fc, Pc, label="comparison PSD")
    ax3.set_xlim(0, 15)
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("Power")
    ax3.set_title("Periodogram")
    ax3.legend()

    ax4 = plt.subplot(4, 1, 4)
    vals = [
        result["amp_ratio_vs_total"],
        result["freq_ratio_vs_total"],
        result["damp_ratio_vs_total"],
        result["phase_ratio_vs_total"],
    ]
    ax4.bar(["amp", "freq", "damp", "phase"], vals)
    ax4.set_ylabel("Norm ratio vs total")
    ax4.set_title("Channel ratios")

    plt.tight_layout()
    plt.savefig(plot_dir / f"{pair_name}.png", dpi=200)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    fail_rows = []

    for coor_dir in COOR_DIRS:
        ref_ids = set(available_sample_ids(REF_CONDITION, coor_dir))
        cmp_ids = set(available_sample_ids(CMP_CONDITION, coor_dir))
        common_ids = sorted(ref_ids & cmp_ids)

        if SAMPLE_IDS is not None:
            common_ids = [sid for sid in SAMPLE_IDS if sid in common_ids]

        print(f"\n=== {coor_dir} ===")
        print(f"Common sample ids: {common_ids}")

        for sid in common_ids:
            ref_path = file_for_sample(REF_CONDITION, coor_dir, sid)
            cmp_path = file_for_sample(CMP_CONDITION, coor_dir, sid)

            try:
                result, extras = analyze_pair(ref_path, cmp_path, coor_dir, sid)
                all_results.append(result)
                save_pair_plot(out_dir, result, extras)

                print(
                    f"sample_{sid}: "
                    f"R2(ref,cmp)=({result['ref_r2']:.3f},{result['cmp_r2']:.3f}) | "
                    f"amp={result['amp_ratio_vs_total']:.3f}, "
                    f"freq={result['freq_ratio_vs_total']:.3f}, "
                    f"damp={result['damp_ratio_vs_total']:.3f}, "
                    f"phase={result['phase_ratio_vs_total']:.3f}, "
                    f"trusted={result['trusted']}"
                )

            except Exception as e:
                warnings.warn(f"Failed on {coor_dir}/sample_{sid}: {e}")
                fail_rows.append({
                    "coor_dir": coor_dir,
                    "sample_id": sid,
                    "error": str(e),
                })

    if not all_results:
        raise RuntimeError("No successful pairs analyzed.")

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "channel_summary_all.csv", index=False)

    if fail_rows:
        pd.DataFrame(fail_rows).to_csv(out_dir / "failures.csv", index=False)

    trusted_df = df[df["trusted"]].copy()
    trusted_df.to_csv(out_dir / "channel_summary_trusted_only.csv", index=False)

    agg_all = df.groupby("coor_dir", as_index=False)[[
        "amp_ratio_vs_total",
        "freq_ratio_vs_total",
        "damp_ratio_vs_total",
        "phase_ratio_vs_total",
        "dom_freq_shift_hz",
        "dom_sigma_shift",
        "dom_phase_shift",
        "ref_r2",
        "cmp_r2",
    ]].mean()
    agg_all.to_csv(out_dir / "aggregate_by_class_all.csv", index=False)

    if len(trusted_df):
        agg_trusted = trusted_df.groupby("coor_dir", as_index=False)[[
            "amp_ratio_vs_total",
            "freq_ratio_vs_total",
            "damp_ratio_vs_total",
            "phase_ratio_vs_total",
            "dom_freq_shift_hz",
            "dom_sigma_shift",
            "dom_phase_shift",
            "ref_r2",
            "cmp_r2",
        ]].mean()
        agg_trusted.to_csv(out_dir / "aggregate_by_class_trusted_only.csv", index=False)
    else:
        agg_trusted = pd.DataFrame()

    summary = {
        "n_pairs_total": int(len(df)),
        "n_pairs_trusted": int(len(trusted_df)),
        "thresholds": {
            "MIN_R2": MIN_R2,
            "MAX_NRMSE": MAX_NRMSE,
        },
        "all_pairs": {
            "amp_ratio_mean":   float(df["amp_ratio_vs_total"].mean()),
            "freq_ratio_mean":  float(df["freq_ratio_vs_total"].mean()),
            "damp_ratio_mean":  float(df["damp_ratio_vs_total"].mean()),
            "phase_ratio_mean": float(df["phase_ratio_vs_total"].mean()),
            "mean_abs_dom_freq_shift_hz": float(df["dom_freq_shift_hz"].abs().mean()),
            "mean_abs_dom_sigma_shift":   float(df["dom_sigma_shift"].abs().mean()),
            "mean_abs_dom_phase_shift":   float(np.mean(np.abs(df["dom_phase_shift"]))),
            "mean_ref_r2": float(df["ref_r2"].mean()),
            "mean_cmp_r2": float(df["cmp_r2"].mean()),
        },
        "trusted_only": None
    }

    if len(trusted_df):
        summary["trusted_only"] = {
            "amp_ratio_mean":   float(trusted_df["amp_ratio_vs_total"].mean()),
            "freq_ratio_mean":  float(trusted_df["freq_ratio_vs_total"].mean()),
            "damp_ratio_mean":  float(trusted_df["damp_ratio_vs_total"].mean()),
            "phase_ratio_mean": float(trusted_df["phase_ratio_vs_total"].mean()),
            "mean_abs_dom_freq_shift_hz": float(trusted_df["dom_freq_shift_hz"].abs().mean()),
            "mean_abs_dom_sigma_shift":   float(trusted_df["dom_sigma_shift"].abs().mean()),
            "mean_abs_dom_phase_shift":   float(np.mean(np.abs(trusted_df["dom_phase_shift"]))),
            "mean_ref_r2": float(trusted_df["ref_r2"].mean()),
            "mean_cmp_r2": float(trusted_df["cmp_r2"].mean()),
        }

    with open(out_dir / "aggregate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Overall bar plot: all pairs
    plt.figure(figsize=(7.5, 4.2))
    plt.bar(
        ["amplitude", "frequency", "damping", "phase"],
        [
            summary["all_pairs"]["amp_ratio_mean"],
            summary["all_pairs"]["freq_ratio_mean"],
            summary["all_pairs"]["damp_ratio_mean"],
            summary["all_pairs"]["phase_ratio_mean"],
        ],
    )
    plt.ylabel("Mean channel norm / total difference norm")
    plt.title(f"{REF_CONDITION} vs {CMP_CONDITION} — all pairs")
    plt.tight_layout()
    plt.savefig(out_dir / "aggregate_channel_ratios_all.png", dpi=200)
    plt.close()

    # Trusted-only bar plot
    if len(trusted_df):
        plt.figure(figsize=(7.5, 4.2))
        plt.bar(
            ["amplitude", "frequency", "damping", "phase"],
            [
                summary["trusted_only"]["amp_ratio_mean"],
                summary["trusted_only"]["freq_ratio_mean"],
                summary["trusted_only"]["damp_ratio_mean"],
                summary["trusted_only"]["phase_ratio_mean"],
            ],
        )
        plt.ylabel("Mean channel norm / total difference norm")
        plt.title(f"{REF_CONDITION} vs {CMP_CONDITION} — trusted pairs only")
        plt.tight_layout()
        plt.savefig(out_dir / "aggregate_channel_ratios_trusted_only.png", dpi=200)
        plt.close()

    print("\nDone.")
    print(f"Outputs saved to: {out_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()