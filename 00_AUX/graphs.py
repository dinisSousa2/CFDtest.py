import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap

# Use coolwarm colormap
cmap = plt.cm.coolwarm

# Create a figure and axis for the colorbar
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)

# Define normalization range to match your ACTUAL data range
vmin = 0
vmax = 5000
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Create the colorbar
cb = plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='horizontal'
)

# Set your desired tick locations
ticks = [0, 1000, 2000, 3000, 4000, 5000]
cb.set_ticks(ticks)

# Create custom tick labels with different formatting
tick_labels = []
for i, tick in enumerate(ticks):
    if i == 0:  # First element - 4 decimal places
        tick_labels.append(f"{tick:.0f}")
    elif i == len(ticks) - 1:  # Last element - 2 decimal places
        tick_labels.append(f"{tick:.0f}")
    else:  # All other elements - 1 decimal place
        tick_labels.append(f"{tick:.0f}")

cb.set_ticklabels(tick_labels)

# Make tick labels larger
cb.ax.tick_params(labelsize=14)

# Get current directory and define save path
save_path = os.path.join(os.getcwd(), "colorbar_coolwarm.png")

# Save the figure with transparent background
plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)

print(f"✅ Colorbar saved successfully (transparent) at: {save_path}")