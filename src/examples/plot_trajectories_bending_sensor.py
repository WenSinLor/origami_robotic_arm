import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR   = "/home/wensin/Documents/origami_robotic_arm/data/soft_state_100g_near_bending_sensor"
OUTPUT_PNG = "/home/wensin/Documents/origami_robotic_arm/data/soft_state_100g_near_bending_sensor/bending_sensor_overview.pdf"  # None = show only

COOR_FOLDERS = ["coor_0", "coor_1", "coor_2", "coor_3"]
COOR_LABELS  = ["Coor 0", "Coor 1", "Coor 2", "Coor 3"]

# Time window to plot (seconds). Set both to None to plot all.
T_START = None
T_END   = None

# ============================================================
# Load all CSVs
# ============================================================

datasets = {}
for folder in COOR_FOLDERS:
    csv_path = os.path.join(DATA_DIR, folder, f"soft_state_100g_near_{folder}.csv")
    df = pd.read_csv(csv_path)
    datasets[folder] = df
    print(f"Loaded {folder}: {len(df)} samples, t=[{df['Time'].iloc[0]:.3f}, {df['Time'].iloc[-1]:.3f}]s")

channels = [c for c in datasets[COOR_FOLDERS[0]].columns if c != "Time"]
n_channels = len(channels)  # 12

cmap   = plt.get_cmap("tab20")
colors = [cmap(i) for i in range(n_channels)]

# ============================================================
# Layout: 2x2 grid, one subplot per coordinate
# Each subplot overlays all 12 channels
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
fig.suptitle("Bending Sensor — Channels Overlaid per Coordinate  |  soft_state_100g",
             fontsize=13, fontweight="bold")

axes_flat = axes.flatten()

for co_idx, (folder, label) in enumerate(zip(COOR_FOLDERS, COOR_LABELS)):
    ax = axes_flat[co_idx]
    df = datasets[folder]

    t = df["Time"].values

    # Apply time window
    if T_START is not None or T_END is not None:
        t0   = T_START if T_START is not None else t[0]
        t1   = T_END   if T_END   is not None else t[-1]
        mask = (t >= t0) & (t <= t1)
        df_plot = df[mask]
        t       = t[mask]
    else:
        df_plot = df

    for ch_idx, ch_name in enumerate(channels):
        short_name = ch_name.split("_")[-1]  # e.g. "ai0"
        ax.plot(t, df_plot[ch_name].values,
                color=colors[ch_idx], linewidth=0.8,
                label=short_name, rasterized=True)

    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Voltage (V)", fontsize=9)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6, ncol=4, loc="upper right",
              framealpha=0.7, handlelength=1.2)

fig.tight_layout()

# ============================================================
# Save / show
# ============================================================

if OUTPUT_PNG:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PNG)), exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PNG}")

plt.show()
