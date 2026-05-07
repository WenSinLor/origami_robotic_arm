import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR    = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/com_demo_100g_com/coor_0"
CSV_FILE    = "com_demo_100g_com.csv"   # e.g. "com_demo.csv" or "com_demo_2.csv"
OUTPUT_PNG  = None             # e.g. ".../com_demo_segments.png", or None = show only

N_SEGMENTS   = 20              # number of segments (= number of clicks)
SEG_DURATION = 7.0             # seconds per segment

# ============================================================
# Load CSV
# ============================================================

csv_path = os.path.join(DATA_DIR, CSV_FILE)
df = pd.read_csv(csv_path)

if "Time" not in df.columns:
    raise ValueError(f"'Time' column not found in {CSV_FILE}")

t  = df["Time"].values
dt = t[1] - t[0]

channels    = [c for c in df.columns if c != "Time"]
n_channels  = len(channels)
voltages    = np.abs(df[channels].values)   # absolute values
short_names = channels

cmap      = plt.get_cmap("tab20")
ch_colors = [cmap(i % 20) for i in range(n_channels)]

print(f"Loaded {CSV_FILE}: {len(df)} samples, {t[-1]:.1f}s, dt={dt*1e3:.1f}ms")
print("Channels:", channels)

# ============================================================
# Interactive selection: click N_SEGMENTS times to set slice starts
# ============================================================

fig_sel, ax_sel = plt.subplots(figsize=(16, 4))
for ch_idx, ch_name in enumerate(channels):
    ax_sel.plot(
        t,
        voltages[:, ch_idx],
        color=ch_colors[ch_idx],
        linewidth=0.6,
        alpha=0.8,
        label=short_names[ch_idx],
        rasterized=True
    )

ax_sel.set_title(
    f"{CSV_FILE} — Click {N_SEGMENTS} times to select segment start points  "
    f"(right-click or Backspace to undo last)",
    fontsize=10,
    fontweight="bold"
)
ax_sel.set_xlabel("Time (s)")
ax_sel.set_ylabel("|Voltage| (V)")
ax_sel.legend(fontsize=6, ncol=6, loc="upper right", framealpha=0.7)
ax_sel.grid(True, linewidth=0.25, alpha=0.5)
fig_sel.tight_layout()

print(f"Click {N_SEGMENTS} times on the plot to select segment start times.")
print("Right-click or Backspace to undo the last click. Close window when done.")
clicks = plt.ginput(N_SEGMENTS, timeout=-1)
plt.close(fig_sel)

seg_starts = sorted(x for x, _ in clicks)

print(f"\nSelected {len(seg_starts)} start times:")
for i, ts in enumerate(seg_starts):
    print(f"  Seg {i:2d}: t_start = {ts:.3f}s")

if len(seg_starts) != N_SEGMENTS:
    raise RuntimeError(f"Expected {N_SEGMENTS} clicks, but got {len(seg_starts)}.")

# ============================================================
# Slice segments from clicked start times
# ============================================================

n_seg_samples = int(round(SEG_DURATION / dt))
seg_time      = np.arange(n_seg_samples) * dt
segments      = np.empty((N_SEGMENTS, n_seg_samples, n_channels))

for i, ts in enumerate(seg_starts):
    idx0 = int(round(ts / dt))
    idx1 = idx0 + n_seg_samples
    if idx1 > len(voltages):
        raise RuntimeError(f"Segment {i} (t={ts:.2f}s) exceeds recording length")
    segments[i] = voltages[idx0:idx1, :]

# ============================================================
# Save segments as HDF5
# positions shape: (F, N, 1)
# ============================================================

out_dir = Path(DATA_DIR) / Path(CSV_FILE).stem
out_dir.mkdir(parents=True, exist_ok=True)

for i, ts in enumerate(seg_starts):
    h5_path = out_dir / f"trajectories_sample_{i}.h5"
    pos_arr = segments[i, :, :, np.newaxis]   # (F, N, 1)
    F, N, _ = pos_arr.shape

    with h5py.File(str(h5_path), "w") as f:
        f.attrs["source_file"] = csv_path
        f.attrs["video_fps"]   = 1.0 / dt
        f.attrs["num_markers"] = N
        f.attrs["num_frames"]  = F

        ts_grp = f.create_group("time_series")
        ts_grp.create_dataset("time", data=seg_time, compression="gzip")

        nodes = ts_grp.create_group("nodes")
        ds = nodes.create_dataset("positions", data=pos_arr, compression="gzip")
        ds.attrs["units"] = "volts"
        ds.attrs["axes"]  = "frame channel voltage"

        nodes.create_dataset("node_ids", data=np.arange(N, dtype=np.int32))
        nodes.create_dataset("channel_names", data=np.array(short_names, dtype="S"))

    print(f"  Saved: {h5_path}  shape={pos_arr.shape}")

# ============================================================
# Plot: 4 rows × 5 cols = 20 subplots, one per segment
# Each subplot overlays all channels
# ============================================================

N_ROWS, N_COLS = 4, 5

fig, axes = plt.subplots(
    N_ROWS, N_COLS,
    figsize=(N_COLS * 3.5, N_ROWS * 2.5),
    sharex=True,
    sharey=True
)

fig.suptitle(
    f"COM Demo Segments — {CSV_FILE}  |  {N_SEGMENTS}×{SEG_DURATION:.0f}s  |  manual selection",
    fontsize=12,
    fontweight="bold"
)

for seg_i in range(N_SEGMENTS):
    row = seg_i // N_COLS
    col = seg_i % N_COLS
    ax  = axes[row, col]

    for ch_idx in range(n_channels):
        ax.plot(
            seg_time,
            segments[seg_i, :, ch_idx],
            color=ch_colors[ch_idx],
            linewidth=0.7,
            alpha=0.85,
            label=short_names[ch_idx],
            rasterized=True
        )

    ax.set_title(f"Seg {seg_i}  (t={seg_starts[seg_i]:.1f}s)", fontsize=7, fontweight="bold")
    ax.grid(True, linewidth=0.25, alpha=0.5)
    ax.tick_params(labelsize=6)

    if col == 0:
        ax.set_ylabel("|Voltage| (V)", fontsize=7)
    if row == N_ROWS - 1:
        ax.set_xlabel("Time (s)", fontsize=7)

axes[0, -1].legend(
    fontsize=5,
    ncol=3,
    loc="upper right",
    framealpha=0.7,
    handlelength=1.0
)

fig.tight_layout()

# ============================================================
# Save / show
# ============================================================

if OUTPUT_PNG:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PNG)), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")

plt.show()