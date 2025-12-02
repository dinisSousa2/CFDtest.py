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
# 2️⃣ Plot colorbar
# -------------------------------
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Define normalization range to match your actual data
vmin = 90
vmax = 1600
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Create the colorbar
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='horizontal'
)

# Define tick locations
ticks = [90, 400, 800, 1200, 1600]
cb.set_ticks(ticks)

# Create custom tick labels with different formatting
tick_labels = []
for i, tick in enumerate(ticks):
    if i == 0:  # First element - 4 decimal places
        tick_labels.append(f"{tick:.0f}")
    elif i == 1:
        tick_labels.append(f"{tick:.0f}")
    elif i == len(ticks) - 1:  # Last element - 2 decimal places
        tick_labels.append(f"{tick:.0f}")
    else:  # All other elements - 1 decimal place
        tick_labels.append(f"{tick:.0f}")

cb.set_ticklabels(tick_labels)

# Make tick labels larger
cb.ax.tick_params(labelsize=14)

# -------------------------------
# 3️⃣ Save figure
# -------------------------------
save_path = os.path.join(os.getcwd(), f"colorbar_{cmap_name.replace(' ', '_')}.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"✅ Colorbar saved successfully (transparent) at: {save_path}")
