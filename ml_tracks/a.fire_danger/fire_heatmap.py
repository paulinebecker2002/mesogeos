import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from pathlib import Path
import re

# Base directory for saved models
base_path = Path('saved/models/transformer')

# Get latest run folder by modification time
run_dirs = sorted(base_path.glob('*/'), key=lambda p: p.stat().st_mtime, reverse=True)
latest_run = run_dirs[0]
print(f"Using latest run folder: {latest_run}")

# Find latest softmax_probs and coords files by epoch number
softmax_files = sorted(latest_run.glob('softmax_probs_epoch*.npy'),
                       key=lambda f: int(re.search(r'epoch(\d+)', f.name).group(1)), reverse=True)
coords_files = sorted(latest_run.glob('coords_epoch*.npy'),
                      key=lambda f: int(re.search(r'epoch(\d+)', f.name).group(1)), reverse=True)

latest_softmax_file = softmax_files[0]
latest_coords_file = coords_files[0]

print(f"Loading: {latest_softmax_file.name}")
print(f"Loading: {latest_coords_file.name}")

# Load data
probs = np.load(latest_softmax_file)
coords = np.load(latest_coords_file)

print(f"Softmax probs shape: {probs.shape}")
print(f"Coords shape: {coords.shape}")

# Calculate bounds with padding
padding_lon = 0.1  # longitude padding in degrees
padding_lat = 0.1 # latitude padding in degrees

lon_min, lon_max = coords[:, 0].min() - padding_lon, coords[:, 0].max() + padding_lon
lat_min, lat_max = coords[:, 1].min() - padding_lat, coords[:, 1].max() + padding_lat

print(f"Auto Bounds -> Lon: [{lon_min}, {lon_max}], Lat: [{lat_min}, {lat_max}]")

# Create grid (correct order: lon is X, lat is Y)
grid_lon, grid_lat = np.mgrid[lon_min:lon_max:1000j, lat_min:lat_max:1000j]

# Interpolate softmax probabilities on grid
grid_probs = griddata(coords, probs, (grid_lon, grid_lat), method='linear')
nan_mask = np.isnan(grid_probs)
if np.any(nan_mask):
    grid_probs[nan_mask] = griddata(coords, probs, (grid_lon[nan_mask], grid_lat[nan_mask]), method='nearest')

# Plot with Cartopy
fig = plt.figure(figsize=(14, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Heatmap overlay
c = ax.pcolormesh(grid_lon, grid_lat, grid_probs.T,  # <-- .T is critical for correct orientation
                  cmap='Spectral_r', vmin=0, vmax=1, shading='auto')

# Colorbar
cb = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cb.set_label('Fire Danger Probability')

print(f"Grid probs stats -> min: {grid_probs.min()}, max: {grid_probs.max()}, nan_count: {np.isnan(grid_probs).sum()}")
if np.all(grid_probs == 0):
    raise ValueError("Interpolated grid is all zeros. Possibly out of convex hull of input coordinates.")
# Title & Gridlines
plt.title(f'Predicted Fire Danger ({latest_softmax_file.stem})')
ax.gridlines(draw_labels=True, linewidth=0.3)

# Save heatmap
output_path = latest_run / f'{latest_softmax_file.stem}_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to {output_path}")

plt.show()
