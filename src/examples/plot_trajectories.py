import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os

# ============================================================
# CONFIGURATION — edit these paths
# ============================================================

INPUT_H5   = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state_100g/coor_0/trajectories_sample_3.h5"
OUTPUT_PNG = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/soft_state_100g/coor_0/trajectories_sample_3.png"  # None = show only

# ============================================================
# Load
# ============================================================

with h5py.File(INPUT_H5, "r") as f:
    time      = f["time_series/time"][:]
    positions = f["time_series/nodes/positions"][:]
    n_frames  = int(f.attrs.get("n_frames", time.shape[0]))
    n_nodes   = int(f.attrs.get("n_nodes",  positions.shape[1]))

x_all = positions[:, :, 0]   # (n_frames, n_nodes)
y_all = positions[:, :, 1]

print(f"Loaded: {n_frames} frames, {n_nodes} nodes")
print(f"Time range: {time[0]:.3f}s - {time[-1]:.3f}s")

# Slice 1.5s to 5.0s using exact frame indices at 29.97 Hz
FPS       = 29.97
T_START   = 0.0
T_END     = 7.0
start_idx = int(round(T_START * FPS))   # = frame 44
end_idx   = int(round(T_END   * FPS))   # = frame 149
time      = time[start_idx:end_idx] - time[start_idx]  # relative: starts at 0
x_all     = positions[start_idx:end_idx, :, 0]
y_all     = positions[start_idx:end_idx, :, 1]
print(f"Sliced frames {start_idx}-{end_idx} ({T_START}s-{T_END}s at {FPS}Hz): {len(time)} frames")

# ============================================================
# Layout: 7 node-columns x 6 node-rows
# Each node cell = 2 mini-rows: X on top, Y on bottom
# X and Y get INDEPENDENT y-axes so neither gets squashed
# ============================================================

N_COLS      = 7
N_NODE_ROWS = 6    # 7x6 = 42 slots >= 37 nodes

cmap   = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(n_nodes)]

fig = plt.figure(figsize=(N_COLS * 2.4, N_NODE_ROWS * 1.9 + 0.8))
fig.suptitle("Marker Trajectories  |  X (top) and Y (bottom) vs Time",
             fontsize=13, fontweight="bold", y=0.99)

# Outer grid: one cell per node
outer_gs = gridspec.GridSpec(
    N_NODE_ROWS, N_COLS,
    figure=fig,
    top=0.95, bottom=0.05,
    hspace=0.90, wspace=0.45
)

for node_id in range(n_nodes):
    node_row = node_id // N_COLS
    node_col = node_id  % N_COLS

    # Split each outer cell into two inner rows: X (top) / Y (bottom)
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=outer_gs[node_row, node_col],
        hspace=0.05          # tight gap between the two mini-plots
    )

    ax_x = fig.add_subplot(inner_gs[0])
    ax_y = fig.add_subplot(inner_gs[1], sharex=ax_x)  # shared time axis

    x     = x_all[:, node_id]
    y     = y_all[:, node_id]
    obs   = ~np.isnan(x)
    color = colors[node_id]

    # ── X trajectory (top) ───────────────────────────────
    ax_x.plot(time[obs], x[obs], color=color, linewidth=0.9)

    # ── Y trajectory (bottom) — independent y-axis ───────
    ax_y.plot(time[obs], y[obs], color=color, linewidth=0.9, linestyle="--")

    # ── Red shading for missing/NaN frames ────────────────
    if (~obs).any():
        for ax in (ax_x, ax_y):
            # shade each contiguous missing span
            in_gap   = False
            gap_start = None
            for fi, missing in enumerate(~obs):
                if missing and not in_gap:
                    gap_start = time[fi]
                    in_gap    = True
                elif not missing and in_gap:
                    ax.axvspan(gap_start, time[fi],
                               color="red", alpha=0.12, linewidth=0)
                    in_gap = False
            if in_gap:  # span reaches end
                ax.axvspan(gap_start, time[-1],
                           color="red", alpha=0.12, linewidth=0)

    # ── Styling ───────────────────────────────────────────
    ax_x.set_title(f"Node {node_id}", fontsize=7, pad=2, fontweight="bold")
    ax_x.set_ylabel("X (px)", fontsize=5, labelpad=1)
    ax_y.set_ylabel("Y (px)", fontsize=5, labelpad=1)
    ax_y.set_xlabel("t (s)",  fontsize=5, labelpad=1)

    plt.setp(ax_x.get_xticklabels(), visible=False)  # hide shared x ticks on top
    ax_x.tick_params(axis="both", labelsize=4)
    ax_y.tick_params(axis="both", labelsize=4)

    ax_x.grid(True, linewidth=0.25, alpha=0.5)
    ax_y.grid(True, linewidth=0.25, alpha=0.5)

    # missing % annotation
    nan_pct = 100 * (~obs).sum() / max(len(obs), 1)
    if nan_pct > 0:
        ax_x.text(0.98, 0.96, f"miss {nan_pct:.0f}%",
                  transform=ax_x.transAxes, fontsize=4,
                  ha="right", va="top", color="red")

# ── Hide unused slots ─────────────────────────────────────────
for spare in range(n_nodes, N_NODE_ROWS * N_COLS):
    r = spare // N_COLS
    c = spare  % N_COLS
    fig.add_subplot(outer_gs[r, c]).set_visible(False)

# ── Shared legend ─────────────────────────────────────────────
legend_handles = [
    Line2D([0], [0], color="gray", linewidth=1.2,
           label="X position (observed)"),
    Line2D([0], [0], color="gray", linewidth=1.2, ls="--",
           label="Y position (observed)"),
    Line2D([0], [0], color="red", linewidth=6, alpha=0.25,
           label="Missing / NaN frames"),
]
fig.legend(handles=legend_handles,
           loc="lower center", ncol=3,
           fontsize=8, frameon=True,
           bbox_to_anchor=(0.5, 0.01))

if OUTPUT_PNG:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PNG)), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")

plt.show()