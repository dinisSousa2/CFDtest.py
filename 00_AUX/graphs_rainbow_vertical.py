import matplotlib.pyplot as plt
import numpy as np
import os
import json
from matplotlib.colors import LinearSegmentedColormap

# -------------------------------
# 1️⃣ Load ParaView JSON colormap
# -------------------------------
json_path = r"D:\Dinis\WorkRservoir\FPSO Frade\rainbow.json"

with open(json_path, "r") as f:
    cmap_data = json.load(f)

# The JSON is a list with one element (the colormap dictionary)
if isinstance(cmap_data, list) and len(cmap_data) > 0:
    cmap_dict = cmap_data[0]
else:
    cmap_dict = cmap_data

# Extract RGBPoints
if "RGBPoints" in cmap_dict:
    pts = np.array(cmap_dict["RGBPoints"]).reshape(-1, 4)
else:
    raise ValueError("No 'RGBPoints' key found in the JSON colormap structure.")

values = pts[:, 0]
colors = pts[:, 1:4]

# Normalize values between 0 and 1 for matplotlib
values_norm = (values - values.min()) / (values.max() - values.min())

# Create Matplotlib colormap
cmap_name = cmap_dict.get("Name", "Rainbow")
cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(values_norm, colors)))

# -------------------------------
# 2️⃣ Plot vertical colorbar
# -------------------------------
fig, ax = plt.subplots(figsize=(1.2, 6))  # width, height
fig.subplots_adjust(left=0.4, right=0.7)  # adjust margins for vertical layout

# Define normalization range to match your actual data
vmin = 0.000063
vmax = 0.11
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Create the colorbar (VERTICAL)
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='vertical'
)

# Define tick locations
ticks = [0.000063, 0.02, 0.04, 0.06, 0.08, 0.11]
cb.set_ticks(ticks)

# Create custom tick labels
tick_labels = []
for i, tick in enumerate(ticks):
    if i == 0:
        tick_labels.append(f"{tick:.6f}")
    elif i == 1:
        tick_labels.append(f"{tick:.2f}")
    elif i == len(ticks) - 1:
        tick_labels.append(f"{tick:.2f}")
    else:
        tick_labels.append(f"{tick:.2f}")

cb.set_ticklabels(tick_labels)

# Make tick labels larger
cb.ax.tick_params(labelsize=14)

# -------------------------------
# 3️⃣ Save figure
# -------------------------------
save_path = os.path.join(os.getcwd(), f"colorbar_{cmap_name.replace(' ', '_')}_vertical.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"✅ Vertical colorbar saved successfully (transparent) at: {save_path}")
