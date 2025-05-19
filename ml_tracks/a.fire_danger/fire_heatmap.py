import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_csv(csv_path):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Softmax probs shape: {df['prob'].shape}")
    print(f"Coords shape: {df[['lon', 'lat']].shape}")
    return df

def rasterize_binned(df, resolution=0.25):
    # Option A: Binbasierte Mittelwerte
    lat_bins = np.arange(df['lat'].min(), df['lat'].max(), resolution)
    lon_bins = np.arange(df['lon'].min(), df['lon'].max(), resolution)

    df['lat_bin'] = pd.cut(df['lat'], bins=lat_bins)
    df['lon_bin'] = pd.cut(df['lon'], bins=lon_bins)

    grid = df.groupby(['lat_bin', 'lon_bin'])['prob'].mean().reset_index()
    grid['lat_center'] = grid['lat_bin'].apply(lambda x: x.mid)
    grid['lon_center'] = grid['lon_bin'].apply(lambda x: x.mid)

    grid_lat = grid['lat_center'].values
    grid_lon = grid['lon_center'].values
    grid_probs = grid['prob'].values

    return grid_lon, grid_lat, grid_probs

def interpolate_grid(df, resolution=1000, padding=0.1):
    # Option B: Interpolation
    coords = df[['lon', 'lat']].values
    probs = df['prob'].values

    lon_min, lon_max = coords[:, 0].min() - padding, coords[:, 0].max() + padding
    lat_min, lat_max = coords[:, 1].min() - padding, coords[:, 1].max() + padding

    print(f"Auto Bounds -> Lon: [{lon_min}, {lon_max}], Lat: [{lat_min}, {lat_max}]")

    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:complex(resolution),
                         lat_min:lat_max:complex(resolution)]

    grid_probs = griddata(coords, probs, (grid_lon, grid_lat), method='linear')
    nan_mask = np.isnan(grid_probs)
    if np.any(nan_mask):
        grid_probs[nan_mask] = griddata(coords, probs, (grid_lon[nan_mask], grid_lat[nan_mask]), method='nearest')

    return grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max

def interpolate_grid_with_default(df, resolution=1000, padding=0.1, default_value=0.0):
    # Interpolation mit Defaultwert für leere Zellen
    coords = df[['lon', 'lat']].values
    probs = df['prob'].values

    lon_min, lon_max = coords[:, 0].min() - padding, coords[:, 0].max() + padding
    lat_min, lat_max = coords[:, 1].min() - padding, coords[:, 1].max() + padding

    print(f"📦 Bounds: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}]")

    # Erzeuge Grid
    grid_lon, grid_lat = np.mgrid[lon_min:lon_max:complex(resolution), lat_min:lat_max:complex(resolution)]

    # Interpolieren mit linearer Methode
    grid_probs = griddata(coords, probs, (grid_lon, grid_lat), method='linear')

    # Leere Zellen mit Defaultwert füllen
    nan_mask = np.isnan(grid_probs)
    if np.any(nan_mask):
        print("⚠️ Filling NaNs with default value...")
        grid_probs[nan_mask] = default_value

    return grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max


def plot_heatmap(grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max, out_path):
    if grid_lon.size == 0 or grid_lat.size == 0 or grid_probs.size == 0:
        raise ValueError("Grid arrays are empty. Check interpolation or input data.")

    print(f"Grid probs stats -> min: {np.min(grid_probs)}, max: {np.max(grid_probs)}, nan_count: {np.isnan(grid_probs).sum()}")
    if np.all(grid_probs == 0):
        raise ValueError("Interpolated grid is all zeros. Possibly out of convex hull of input coordinates.")

    print("DEBUG: Entering plot_heatmap")
    print("grid_lon.shape:", grid_lon.shape)
    print("grid_lat.shape:", grid_lat.shape)
    print("grid_probs.shape:", grid_probs.shape)
    print("lon_min:", lon_min, "lon_max:", lon_max)
    print("lat_min:", lat_min, "lat_max:", lat_max)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 🔧 Fix für Cartopy-Fehler: explizit globale Kartenausdehnung setzen
    ax.set_global()

    # Falls du nur dein Zielgebiet zeigen willst:
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    print("grid_lon shape:", grid_lon.shape)
    print("grid_lat shape:", grid_lat.shape)
    print("grid_probs shape:", grid_probs.shape)
    assert grid_lon.shape == grid_lat.shape == grid_probs.shape, "Grid shapes mismatch!"

    c = ax.pcolormesh(grid_lon, grid_lat, grid_probs.T, cmap='Spectral_r', vmin=0, vmax=1, shading='auto')
    cb = plt.colorbar(c, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cb.set_label('Fire Danger Probability')

    ax.gridlines(draw_labels=True, linewidth=0.3)
    plt.title('Predicted Fire Danger Heatmap')

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {out_path}")
    plt.close()

def plot_debug_scatter(coords, probs, lon_min, lon_max, lat_min, lat_max, output_path="debug_scatter.png"):
    """
    Plottet einen einfachen Scatter-Plot der Klassifikationspunkte auf einer Karte (ohne Interpolation).
    Nützlich zum Debuggen von Interpolations- oder Cartopy-Fehlern.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    print("⚠️ Running fallback debug scatter plot...")
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgrey', edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=probs, cmap="Spectral_r", s=10, transform=ccrs.PlateCarree())
    plt.colorbar(sc, ax=ax, label="Fire Danger Probability")

    plt.title("Debug Scatter of Predicted Probabilities")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Debug scatter plot saved to {output_path}")

def main():
    csv_path = '/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/models/mlp/0516_100408/val_softmax_outputs_epoch1.csv'
    out_path = Path(csv_path).with_name('debug.png')

    df = load_csv(csv_path)
    coords = df[['lon', 'lat']].values
    probs = df['prob'].values

    # Option B: Interpoliertes Grid
    grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max = interpolate_grid_with_default(df)

    # Plot
    plot_heatmap(grid_lon, grid_lat, grid_probs, lon_min, lon_max, lat_min, lat_max, out_path)
    #plot_debug_scatter(grid_lon, grid_lat, lon_min, lon_max, lat_min, lat_max, output_path="debug_scatter.png")

    # Optional: Für Option A – auskommentieren, wenn gewünscht
    # grid_lon, grid_lat, grid_probs = rasterize_binned(df)
    # plot_heatmap(grid_lon, grid_lat, np.reshape(grid_probs, (len(set(grid_lon)), len(set(grid_lat)))),
    #              min(grid_lon), max(grid_lon), min(grid_lat), max(grid_lat), Path(csv_path).with_name('val_softmax_binned.png'))

if __name__ == "__main__":
    main()
