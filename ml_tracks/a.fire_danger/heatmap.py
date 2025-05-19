import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/code/ml_tracks/a.fire_danger/saved/models/mlp/0516_100408/val_softmax_outputs_epoch1.csv')

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())
sc = plt.scatter(df['lon'], df['lat'], c=df['prob'], cmap='turbo', s=10, vmin=0, vmax=1)
plt.colorbar(sc, label='Fire Danger')
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_extent([-10, 37, 30, 47])  # Mediterranean
plt.title('Predicted Fire Danger for Validation Samples')
plt.savefig('med_fire_danger_map.png', dpi=300)
plt.show()
