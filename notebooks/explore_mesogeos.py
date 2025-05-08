import xarray as xr
import matplotlib.pyplot as plt
import zarr

# Optional: Logging für Cluster-Jobs
import logging
logging.basicConfig(level=logging.INFO)

# ---- Öffne das Dataset
# Pfad zur Zarr-Datei anpassen, falls nötig
zarr_path = "/hkfs/work/workspace/scratch/uyxib-pauline_gddpfa/mesogeos/mesogeos_cube.zarr"
logging.info(f"Öffne Zarr Dataset von: {zarr_path}")
ds = xr.open_zarr(zarr_path)

# ---- Initialisiere Ergebnisliste
all_ignitions = []

# ---- Iteriere über alle Zeitschritte
time_coords = ds['time'].values
logging.info(f"{len(time_coords)} Zeitschritte werden durchsucht...")

for i, t in enumerate(time_coords):
    ignition = ds['ignition_points'].isel(time=i)
    ignition_flat = ignition.stack(z=("y", "x"))
    ignited = ignition_flat.where(ignition_flat > 0, drop=True)

    # Extrahiere Zeitstempel als string
    time_str = str(t)[:10]

    # Füge alle aktiven Ignition-Points für diesen Zeitschritt zur Liste hinzu
    for y, x, val in zip(ignited['y'].values, ignited['x'].values, ignited.values):
        all_ignitions.append((time_str, float(x), float(y), float(val)))

    if i % 500 == 0:
        logging.info(f"{i}/{len(time_coords)} Zeitschritte verarbeitet...")

# ---- Beispielhafte Ausgabe
logging.info("Beispielhafte Ignition Points (max. 10):")
for point in all_ignitions[:10]:
    print(point)

logging.info(f"Insgesamt {len(all_ignitions)} Ignition Points gefunden.")

# ---- NDVI für den 100. Tag plotten
#plt.figure()
#ds.isel(time=100).ndvi.plot(cmap="RdYlGn", vmax=1, vmin=-1)
#plt.title("NDVI am Tag 100")
#plt.savefig("ndvi_day100.png", dpi=150)
#plt.close()
#logging.info("NDVI-Plot gespeichert als ndvi_day100.png")

# ---- Durchschnittliche Landoberflächentemperatur (LST) 2022
#plt.figure()
#ds.sel(time=slice('2022-01-01', None)).lst_day.mean(dim=('x', 'y')).plot()
#plt.title("Durchschnittliche LST 2022")
#plt.ylabel("LST [°C]")
#plt.xlabel("Zeit")
#plt.savefig("avg_lst_2022.png", dpi=150)
#plt.close()
#logging.info("LST-Plot gespeichert als avg_lst_2022.png")
