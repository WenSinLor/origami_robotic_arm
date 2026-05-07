import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/com_demo_60g/coor_2"
CSV_FILE = "com_demo_60g_coor_2.csv"   # change to "com_demo_2.csv" if needed
OUTPUT_PNG = "/Users/albertlor/Documents/Academic_PhD/origami_robotic_arm/data/com_demo_60g/coor_2/com_demo_plot.png"  # None = show only

# Time window to plot (seconds). Set both to None to plot all.
T_START = None
T_END   = None

# If None, automatically plot all columns except Time
CHANNELS_TO_PLOT = None

# ============================================================
# Load one CSV
# ============================================================

csv_path = os.path.join(DATA_DIR, CSV_FILE)
df = pd.read_csv(csv_path)

print(f"Loaded {CSV_FILE}: {len(df)} samples")
print("Columns:", list(df.columns))

# Check Time column
if "Time" not in df.columns:
    raise ValueError(f"'Time' column not found in {CSV_FILE}")

t = df["Time"].values

# Apply time window
if T_START is not None or T_END is not None:
    t0 = T_START if T_START is not None else t[0]
    t1 = T_END   if T_END   is not None else t[-1]
    mask = (t >= t0) & (t <= t1)
    df_plot = df[mask]
    t = df_plot["Time"].values
else:
    df_plot = df

# Select channels
if CHANNELS_TO_PLOT is None:
    channels = [c for c in df.columns if c != "Time"]
else:
    channels = CHANNELS_TO_PLOT

n_channels = len(channels)
cmap = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(n_channels)]

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(f"COM Demo — Channels Overlaid  |  {CSV_FILE}",
             fontsize=13, fontweight="bold")

for ch_idx, ch_name in enumerate(channels):
    y = np.abs(df_plot[ch_name].values)
    ax.plot(
        t,
        y,
        color=colors[ch_idx],
        linewidth=1.0,
        label=ch_name,
        rasterized=True
    )

ax.set_title("Selected CSV", fontsize=11, fontweight="bold")
ax.set_xlabel("Time (s)", fontsize=10)
ax.set_ylabel("Voltage (V)", fontsize=10)   # change label if needed
ax.grid(True, linewidth=0.3, alpha=0.5)
ax.tick_params(labelsize=9)
ax.legend(fontsize=8, ncol=4, loc="upper right",
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