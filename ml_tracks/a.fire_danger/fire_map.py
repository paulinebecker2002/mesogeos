import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

def load_softmax_csv(csv_path):
    print(f"🔹 Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    coords = df[['lon', 'lat']].values
    probs = df['prob'].values
    return coords, probs

def interpolate_probs(coords, probs, grid_resolution=1000, padding=0.1):
    lon_min, lon_max = coords[:, 0].min() - padding, coords[:, 0].max() + padding
    lat_min, lat_max = coords[:, 1].min() - padding, coords[:, 1].max() + padding
    print(f"📦 Bounds: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(lon_min, lon_max, grid_resolution),
        np.linspace(lat_min, lat_max, grid_resolution)
    )

    grid_probs = griddata(coords, probs, (grid_lon, grid_lat), method='linear')

    # Fill NaNs using nearest neighbor fallback
    if np.any(np.isnan(grid_probs)):
        print("⚠️ Filling NaNs via nearest...")
        grid_probs = np.where(np.isnan(grid_probs),
                              griddata(coords, probs, (grid_lon, grid_lat), method='nearest'),
                              grid_probs)

    return grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max

def plot_cartopy_heatmap(grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max, out_file):
    print("🗺️ Plotting with Cartopy...")
    fig = plt.figure(figsize=(14, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    c = ax.pcolormesh(grid_lon, grid_lat, grid_probs,
                      cmap='Spectral_r', vmin=0, vmax=1, shading='auto')
    cb = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cb.set_label("Fire Danger Probability")

    plt.title("Fire Danger Heatmap")
    ax.gridlines(draw_labels=True, linewidth=0.3)

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap to {out_file}")

def plot_points_on_map(coords, probs, output_path):
    lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
    lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

    fig = plt.figure(figsize=(12, 6))
    ax = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_global()  # statt set_extent
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=probs, cmap='Spectral_r', s=10, transform=ccrs.PlateCarree())

    cb = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.05)
    cb.set_label("Fire Danger Probability")

    plt.title("Fire Danger Probabilities (Point Map)")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved point map to {output_path}")


def main():
    csv_path = '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/models/mlp/0516_100408/val_softmax_outputs_epoch1.csv'
    coords, probs = load_softmax_csv(csv_path)
    grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max = interpolate_probs(coords, probs)
    #plot_cartopy_heatmap(grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max, Path(csv_path).with_name('val_softmax_cartopy_heatmap.png'))
    output_path = Path(csv_path).with_name('point_map.png')
    plot_points_on_map(coords, probs, output_path)


if __name__ == "__main__":
    main()
