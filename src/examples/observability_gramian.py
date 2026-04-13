"""
Gramian configuration comparison at 20g
=========================================

Focused test: at fixed mass (20g), does the Gramian encoding score
correctly rank soft > mix > stiff in classification accuracy?

This is the cleanest test of whether the Gramian captures
configuration effects, because mass is held constant so any
difference in encoding score must come from the configuration
changing A₀ and C, not from m scaling δA.

Actual LOO-CV at 20g (from your experiments):
  soft_20g  : 75.0%
  mix_20g   : 71.9%
  stiff_20g : 60.0%

Expected Gramian ranking if framework is useful:
  E(soft_20g) > E(mix_20g) > E(stiff_20g)

Current result from full script:
  E(soft_20g)  = 28,728   ← LOWEST
  E(mix_20g)   = 35,500   ← HIGHEST
  E(stiff_20g) = 30,325   ← MIDDLE

Problem: all three use the SAME A₀ (soft empty baseline).
This script tests what happens when we use the mean trajectory
of each configuration as its own baseline for system identification,
which is the correct approach when stiff/mix empty trials are absent.

Two approaches tested:
  A) Use per-configuration mean loaded trajectory for system ID
     (proxy: treat each config's own 20g mean as the reference)
  B) Use the sensitivity trajectory relative to soft empty,
     but weight it by the per-configuration Gramian built from
     the loaded trajectory's own spectral content
"""

from pathlib import Path
import h5py
import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov
from scipy.signal import welch, find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────

ROOT_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data"

DIRS = {
    "soft_empty":  f"{ROOT_DIR}/soft_state_noload",
    "soft_20g":    f"{ROOT_DIR}/soft_state_20g",
    "stiff_20g":   f"{ROOT_DIR}/stiff_state_20g",
    "mix_20g":     f"{ROOT_DIR}/mix_state_20g_right",
}

SAMPLES = {
    "soft_empty": {"coor_0": list(range(8))},
    "soft_20g":   {"coor_0": list(range(8)), "coor_1": list(range(8)),
                   "coor_2": list(range(8)), "coor_3": list(range(8))},
    "stiff_20g":  {"coor_0": list(range(8)), "coor_1": list(range(8)),
                   "coor_2": list(range(8)), "coor_3": list(range(8))},
    "mix_20g":    {"coor_0": list(range(8)), "coor_1": list(range(8)),
                   "coor_2": list(range(8)), "coor_3": list(range(8))},
}

ACTUAL_ACC = {"soft_20g": 0.750, "mix_20g": 0.719, "stiff_20g": 0.600}
MASS_KG    = 0.020

T_START, T_END    = 0.0, 3.0
BASELINE_FRAMES   = 1
N_MODES           = 6
OUTPUT_DIR        = Path(ROOT_DIR) / "soft_state_noload"

COLORS = {"soft_20g": "#56B4E9", "stiff_20g": "#0072B2", "mix_20g": "#E69F00"}
LABELS = {"soft_20g": "Soft 20g", "stiff_20g": "Stiff 20g", "mix_20g": "Mix 20g"}

NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_h5(path):
    with h5py.File(str(path), "r") as f:
        pos  = f["time_series/nodes/positions"][:]
        time = f["time_series/time"][:]
    for n in range(pos.shape[1]):
        for ax in range(pos.shape[2]):
            col = pos[:, n, ax]
            nan_mask = np.isnan(col)
            if nan_mask.all():   col[:] = 0.0
            elif nan_mask.any():
                idx = np.where(~nan_mask)[0]
                col[:idx[0]] = col[idx[0]]
                col[idx[-1]:] = col[idx[-1]]
                for i in range(len(idx)-1):
                    if idx[i+1]-idx[i] > 1:
                        col[idx[i]+1:idx[i+1]] = col[idx[i]]
    baseline = pos[:min(BASELINE_FRAMES, pos.shape[0])].mean(axis=0, keepdims=True)
    return pos - baseline, time


def load_mean(cond_key):
    trials, ts_ref = [], None
    base = Path(DIRS[cond_key])
    for coor, ids in SAMPLES[cond_key].items():
        for sid in ids:
            p = base / coor / f"trajectories_sample_{sid}.h5"
            if not p.exists():   continue
            disp, time = load_h5(p)
            i0 = int(np.searchsorted(time, T_START))
            i1 = int(np.searchsorted(time, T_END, side="right"))
            X  = disp[i0:i1].reshape(i1-i0, -1)
            ts = time[i0:i1] - time[i0]
            trials.append(X)
            if ts_ref is None: ts_ref = ts
    if not trials: return None, None
    min_T = min(t.shape[0] for t in trials)
    return np.stack([t[:min_T] for t in trials]).mean(0), ts_ref[:min_T]


# ── SYSTEM IDENTIFICATION ─────────────────────────────────────────────────────

def identify_system(mean_traj, ts, n_modes):
    """
    Build A₀ and C from the ringdown trajectory of a given condition.
    Uses spectral peaks of the trajectory to identify modal frequencies.
    """
    T, D = mean_traj.shape
    dt   = float(ts[1] - ts[0])
    fs   = 1.0 / dt

    # collect spectral peaks
    all_freqs = []
    for d in range(D):
        f, Pxx = welch(mean_traj[:, d], fs=fs, nperseg=min(256, T))
        peaks, _ = find_peaks(Pxx, height=np.max(Pxx)*0.03, distance=2)
        for pk in peaks:
            if f[pk] > 0.1:
                all_freqs.append(f[pk])

    if not all_freqs:
        all_freqs = np.linspace(0.5, 5.0, n_modes).tolist()

    all_freqs = np.sort(all_freqs)
    if len(all_freqs) >= n_modes:
        bins = np.array_split(all_freqs, n_modes)
        modal_hz = np.array([b.mean() for b in bins if len(b)])
    else:
        modal_hz = np.array(all_freqs)
        while len(modal_hz) < n_modes:
            modal_hz = np.append(modal_hz, modal_hz[-1]*1.4)
    modal_hz = modal_hz[:n_modes]
    omega = 2 * np.pi * modal_hz

    # damping from RMS envelope
    t_arr   = ts.astype(float)
    rms_env = np.maximum(np.sqrt(np.mean(mean_traj**2, axis=1)), 1e-12)
    log_rms = np.log(rms_env)
    valid   = log_rms > np.max(log_rms) - 3
    if valid.sum() > 4:
        c = np.polyfit(t_arr[valid], log_rms[valid], 1)
        zeta_avg = max(0.01, -c[0] / (np.mean(omega) + 1e-10))
    else:
        zeta_avg = 0.05
    zetas = np.full(n_modes, np.clip(zeta_avg, 0.01, 0.5))

    # A₀: block diagonal
    nm2 = 2 * n_modes
    A0  = np.zeros((nm2, nm2))
    for r in range(n_modes):
        w, z = omega[r], zetas[r]
        A0[2*r:2*r+2, 2*r:2*r+2] = [[0, 1], [-w**2, -2*z*w]]

    # C: modal projection
    C = np.zeros((D, nm2))
    for r in range(n_modes):
        w, z = omega[r], zetas[r]
        wd   = w * np.sqrt(max(1-z**2, 0.01))
        tmpl = np.exp(-z*w*t_arr) * np.cos(wd*t_arr)
        tmpl /= (np.linalg.norm(tmpl) + 1e-10)
        for d in range(D):
            C[d, 2*r] = float(np.dot(mean_traj[:, d], tmpl))

    return A0, C, omega, zetas


# ── GRAMIAN ───────────────────────────────────────────────────────────────────

def gramian(A0, C, ts):
    dt   = float(ts[1] - ts[0])
    A0_d = expm(A0 * dt)
    eigs = np.linalg.eigvals(A0_d)
    if np.any(np.abs(eigs) >= 1.0):
        A0_d *= 0.98 / np.max(np.abs(eigs))
    Q = C.T @ C
    try:
        Wo = solve_discrete_lyapunov(A0_d.T, Q)
    except Exception:
        Wo = np.zeros_like(Q)
        Ak = np.eye(A0_d.shape[0])
        for _ in range(len(ts)):
            Wo += Ak.T @ Q @ Ak
            Ak = A0_d @ Ak
    return (Wo + Wo.T) / 2.0


def delta_A(empty_mean, loaded_mean, ts, A0, omega, zetas):
    """Estimate δA from frequency shifts between empty and loaded."""
    t_arr   = ts.astype(float)
    n_modes = len(omega)
    dA      = np.zeros_like(A0)

    for r in range(n_modes):
        w0, z0 = omega[r], zetas[r]
        wd0    = w0 * np.sqrt(max(1-z0**2, 0.01))

        # search for loaded frequency in ±40% band
        w_search  = np.linspace(w0*0.6, w0*1.4, 50)
        best_corr, best_w = 0.0, w0
        D = loaded_mean.shape[1]
        for w_try in w_search:
            wd_try = w_try * np.sqrt(max(1-z0**2, 0.01))
            tmpl   = np.exp(-z0*w_try*t_arr) * np.cos(wd_try*t_arr)
            tmpl  /= (np.linalg.norm(tmpl) + 1e-10)
            corr   = 0.0
            for d in range(D):
                corr += abs(float(np.dot(loaded_mean[:, d], tmpl)))
            if corr > best_corr:
                best_corr, best_w = corr, w_try

        w1  = best_w
        dw2 = w1**2 - w0**2

        # damping shift
        rms = np.maximum(np.sqrt(np.mean(loaded_mean**2, axis=1)), 1e-12)
        log_rms = np.log(rms)
        valid   = log_rms > np.max(log_rms) - 3
        if valid.sum() > 4:
            c  = np.polyfit(t_arr[valid], log_rms[valid], 1)
            z1 = max(0.01, -c[0] / (w1 + 1e-10))
        else:
            z1 = z0

        dA[2*r+1, 2*r  ] = -dw2
        dA[2*r+1, 2*r+1] = -(2*z1*w1 - 2*z0*w0)

    return dA


def encoding_score(Wo, dA):
    P0    = np.eye(dA.shape[0])
    dAP   = dA @ P0 @ dA.T
    score = float(np.trace(Wo @ dAP))
    return np.sqrt(max(score, 0.0))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("═"*60)
    print("Gramian configuration comparison at 20g")
    print("Mass is fixed — any Gramian difference = config effect")
    print("═"*60)

    # load means
    means, tss = {}, {}
    for cond in DIRS:
        m, ts = load_mean(cond)
        if m is None:
            print(f"  [WARN] could not load {cond}")
        else:
            means[cond] = m
            tss[cond]   = ts
            print(f"  Loaded {cond}: shape {m.shape}")

    empty_mean = means["soft_empty"]
    ts_ref     = tss["soft_empty"]

    print("\n" + "─"*60)
    print("APPROACH A: per-config system identification")
    print("Each configuration identified from its OWN loaded trajectory")
    print("─"*60)

    results_A = {}
    for cond in ["soft_20g", "stiff_20g", "mix_20g"]:
        if cond not in means:  continue
        loaded = means[cond]
        ts     = tss[cond]
        T      = min(loaded.shape[0], empty_mean.shape[0])
        lm, em = loaded[:T], empty_mean[:T]
        ts_a   = ts[:T]

        # identify system from THIS config's loaded trajectory
        A0_c, C_c, omega_c, zetas_c = identify_system(lm, ts_a, N_MODES)
        Wo_c  = gramian(A0_c, C_c, ts_a)
        dA_c  = delta_A(em, lm, ts_a, A0_c, omega_c, zetas_c)
        E_c   = encoding_score(Wo_c, dA_c)

        results_A[cond] = E_c
        print(f"  {cond:12s}  E = {E_c:.4e}  "
              f"acc = {ACTUAL_ACC[cond]:.3f}  "
              f"modal_freqs_hz = {(omega_c/(2*np.pi)).round(2)}")

    print("\n" + "─"*60)
    print("APPROACH B: soft empty baseline, but config-specific Gramian")
    print("δA estimated from sensitivity, Gramian built from config trajectory")
    print("─"*60)

    results_B = {}
    # identify soft system from soft empty
    A0_soft, C_soft, omega_soft, zetas_soft = identify_system(
        empty_mean, ts_ref, N_MODES)
    Wo_soft = gramian(A0_soft, C_soft, ts_ref)

    for cond in ["soft_20g", "stiff_20g", "mix_20g"]:
        if cond not in means:  continue
        loaded = means[cond]
        ts     = tss[cond]
        T      = min(loaded.shape[0], empty_mean.shape[0])
        lm, em = loaded[:T], empty_mean[:T]
        ts_a   = ts[:T]

        # Gramian from config's loaded trajectory spectral content
        A0_c, C_c, _, _ = identify_system(lm, ts_a, N_MODES)
        Wo_c = gramian(A0_c, C_c, ts_a)

        # δA from soft baseline (consistent reference)
        dA_c = delta_A(em, lm, ts_a, A0_soft, omega_soft, zetas_soft)
        E_c  = encoding_score(Wo_c, dA_c)

        results_B[cond] = E_c
        print(f"  {cond:12s}  E = {E_c:.4e}  acc = {ACTUAL_ACC[cond]:.3f}")

    # ── ranking analysis ───────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("RANKING ANALYSIS")
    print("─"*60)
    true_rank  = ["soft_20g", "mix_20g", "stiff_20g"]   # by accuracy desc

    for label, results in [("Approach A", results_A), ("Approach B", results_B)]:
        if not results:  continue
        pred_rank = sorted(results, key=results.get, reverse=True)
        correct   = (pred_rank == true_rank)
        print(f"\n  {label}:")
        for c in ["soft_20g", "mix_20g", "stiff_20g"]:
            if c in results:
                print(f"    {c:12s}  E={results[c]:.3e}  "
                      f"acc={ACTUAL_ACC[c]:.3f}")
        print(f"  Predicted rank: {pred_rank}")
        print(f"  True rank:      {true_rank}")
        print(f"  Correct ranking: {'YES ✓' if correct else 'NO ✗'}")

    # ── FIGURE ─────────────────────────────────────────────────────────────
    conds = ["soft_20g", "mix_20g", "stiff_20g"]
    accs  = [ACTUAL_ACC[c]*100 for c in conds]
    cols  = [COLORS[c] for c in conds]
    labs  = [LABELS[c] for c in conds]

    with plt.rc_context(NATURE_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                 gridspec_kw={"wspace": 0.45})

        for ax_idx, (label, results) in enumerate(
                [("Approach A\n(per-config system ID)", results_A),
                 ("Approach B\n(config Gramian + soft δA)", results_B)]):

            ax = axes[ax_idx]
            if not results:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center")
                continue

            scores = [results.get(c, np.nan) for c in conds]
            for i, c in enumerate(conds):
                if np.isnan(scores[i]):  continue
                ax.scatter(scores[i], accs[i], color=cols[i],
                           s=80, zorder=3, label=labs[i])
                ax.annotate(labs[i], (scores[i], accs[i]),
                            fontsize=6, xytext=(5, 3),
                            textcoords="offset points")

            # rank arrows: draw lines connecting rank order
            valid = [(scores[i], accs[i]) for i in range(len(conds))
                     if not np.isnan(scores[i])]
            if len(valid) > 1:
                valid_sorted = sorted(valid, key=lambda x: x[0])
                xs = [v[0] for v in valid_sorted]
                ys = [v[1] for v in valid_sorted]
                ax.plot(xs, ys, color="#AAAAAA", lw=0.8,
                        linestyle="--", zorder=1)

            ax.set_xlabel("Gramian encoding score $E$")
            ax.set_ylabel("LOO-CV accuracy (%)")
            ax.set_title(label, fontweight="bold")
            ax.set_ylim(50, 105)
            ax.axhline(25, color="#CCCCCC", lw=0.8, linestyle=":")

            # check ranking
            pred_rank = sorted(
                [c for c in conds if not np.isnan(results.get(c, np.nan))],
                key=lambda c: results[c], reverse=True)
            ranking_correct = (pred_rank == true_rank)
            color_box = "#E1F5EE" if ranking_correct else "#FAECE7"
            border_col = "#0F6E56" if ranking_correct else "#993C1D"
            ax.text(0.05, 0.05,
                    f"Ranking: {'✓ correct' if ranking_correct else '✗ wrong'}",
                    transform=ax.transAxes, fontsize=6.5,
                    verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor=color_box,
                              edgecolor=border_col, linewidth=0.8))

        fig.suptitle("Gramian config comparison at 20g — fixed mass, "
                     "config effect only",
                     fontsize=8.5, fontweight="bold", y=1.04)
        outpath = OUTPUT_DIR / "gramian_config_20g.png"
        fig.savefig(str(outpath))
        plt.close(fig)

    # ── FIGURE 2: spectral comparison across configs ───────────────────────
    with plt.rc_context(NATURE_RC):
        fig, ax = plt.subplots(figsize=(4.5, 3.0))

        for cond in ["soft_20g", "stiff_20g", "mix_20g"]:
            if cond not in means: continue
            T_  = min(means[cond].shape[0], empty_mean.shape[0])
            A0_, C_, _, _ = identify_system(means[cond][:T_], tss[cond][:T_],
                                            N_MODES)
            Wo_ = gramian(A0_, C_, tss[cond][:T_])
            eigs = np.sort(np.linalg.eigvalsh(Wo_))[::-1]
            n_show = min(len(eigs), 10)
            ax.semilogy(np.arange(1, n_show+1), eigs[:n_show],
                        "o-", color=COLORS[cond], lw=1.2, ms=4,
                        label=LABELS[cond])

        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue of $\\mathcal{W}_o$")
        ax.set_title("Gramian spectrum per config at 20g\n"
                     "(should differ if config changes observability)",
                     fontweight="bold")
        ax.legend(frameon=True)
        outpath2 = OUTPUT_DIR / "gramian_spectrum_20g_configs.png"
        fig.savefig(str(outpath2))
        plt.close(fig)

    print(f"\nSaved:")
    print(f"  {OUTPUT_DIR}/gramian_config_20g.png")
    print(f"  {OUTPUT_DIR}/gramian_spectrum_20g_configs.png")

    print("""
INTERPRETATION
──────────────
If BOTH approaches rank soft > mix > stiff:
  → Gramian captures configuration effects.
  → Framework is valid for Story 2 at fixed mass.
  → Strong result: proceed with Gramian as theory.

If ONLY Approach A ranks correctly:
  → The configuration signal comes from per-config spectral
    differences, not from the δA estimation.
  → Gramian is useful but requires proper per-config system ID.
  → Note this limitation in the paper; collect stiff/mix
    empty trials to strengthen the argument.

If NEITHER ranks correctly:
  → Gramian does not capture configuration effects.
  → Use Gramian only for mass trend (Story 3a).
  → Story 2 explained by mode shape argument from Step 5b only.
""")


if __name__ == "__main__":
    main()