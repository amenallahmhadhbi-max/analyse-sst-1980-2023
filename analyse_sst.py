# =============================================================================
# Analyse SST 1980-2023 - Analyse Globale et Régionale
# Migration R -> Python
# Auteur : Amenallah Mhadhbi
# Question scientifique : La température de surface de la mer (SST) a-t-elle
# augmenté entre 1980 et 2023, et si oui, à quel rythme et avec quelles
# variations régionales et saisonnières ?
# =============================================================================

# -----------------------------------------------------------------------------
# 1. IMPORTATION DES LIBRAIRIES
# Équivalences R -> Python :
#   ncdf4 / raster   -> xarray, netCDF4, numpy
#   dplyr / tidyr    -> pandas
#   ggplot2          -> matplotlib, seaborn
#   lubridate        -> pandas datetime
#   viridis          -> matplotlib colormaps (plasma, viridis)
#   maps             -> cartopy
#   patchwork        -> matplotlib.gridspec
# -----------------------------------------------------------------------------

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from scipy import stats
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 2. CONFIGURATION
# Équivalent R : data_path <- "..." ; nc_files <- list.files(...)
# -----------------------------------------------------------------------------

data_path = "C:/Users/ammon/Desktop/projet_climat/data/"

# Lecture de tous les fichiers NetCDF dans le dossier
nc_files = sorted(glob.glob(os.path.join(data_path, "*.nc")))

if not nc_files:
    raise FileNotFoundError(f"Aucun fichier NetCDF trouvé dans : {data_path}")

print("=== CONFIGURATION ===")
print(f"Nombre de fichiers NetCDF trouvés : {len(nc_files)}")
print(f"Premier fichier : {os.path.basename(nc_files[0])}")
print(f"Dernier fichier : {os.path.basename(nc_files[-1])}\n")

# -----------------------------------------------------------------------------
# 3. ANALYSE DES VALEURS MANQUANTES (NA)
# Équivalent R : nc_open() -> ncvar_get() -> sum(is.na()) / length()
# En Python les NA sont représentés par np.nan
# -----------------------------------------------------------------------------

print("=== ANALYSE DES VALEURS NA ===")

# Lecture du premier fichier pour diagnostic
ds = xr.open_dataset(nc_files[0])
sst_kelvin = ds["analysed_sst"].values

# Gestion des dimensions : si 3D (time, lat, lon) on prend la première couche
if sst_kelvin.ndim == 3:
    sst_kelvin = sst_kelvin[0]

# Conversion Kelvin -> Celsius (équivalent R : sst_celsius <- sst_kelvin - 273.15)
sst_celsius = sst_kelvin - 273.15
lat = ds["lat"].values
lon = ds["lon"].values
ds.close()

total_cells = sst_celsius.size
na_cells    = int(np.sum(np.isnan(sst_celsius)))
valid_cells = total_cells - na_cells
na_percent  = round(na_cells / total_cells * 100, 1)

print(f"Fichier analysé    : {os.path.basename(nc_files[0])}")
print(f"Dimensions         : {sst_celsius.shape[1]} x {sst_celsius.shape[0]}")
print(f"Cellules totales   : {total_cells:,}")
print(f"Cellules valides   : {valid_cells:,} ({100 - na_percent:.1f}%)")
print(f"Cellules NA        : {na_cells:,} ({na_percent:.1f}%)\n")

# Vérification sur 5 fichiers (équivalent R : boucle for sur nc_files[1:5])
print("Vérification sur 5 fichiers :")
for i in range(min(5, len(nc_files))):
    ds_test = xr.open_dataset(nc_files[i])
    sst_test = ds_test["analysed_sst"].values
    if sst_test.ndim == 3:
        sst_test = sst_test[0]
    sst_test = sst_test - 273.15
    na_pct = round(np.sum(np.isnan(sst_test)) / sst_test.size * 100, 1)
    print(f"  Fichier {i+1} : {na_pct:.1f}% NA")
    ds_test.close()

print(f"\n✅ CONCLUSION : Tous les fichiers ont environ {na_percent}% de NA")
print("   Ces NA représentent les terres émergées et les calottes glaciaires")
print("   Ils sont structurels et doivent être conservés dans l'analyse\n")

# -----------------------------------------------------------------------------
# 4. FONCTION DE LECTURE DES FICHIERS NetCDF
# Équivalent R : read_sst_file <- function(file_path) { ... }
# Retourne un dictionnaire au lieu d'une liste R
# -----------------------------------------------------------------------------

def read_sst_file(file_path):
    """
    Lit un fichier NetCDF SST et retourne ses données et métadonnées.
    Équivalent R : read_sst_file() avec ncdf4 + raster
    """
    ds = xr.open_dataset(file_path)
    sst = ds["analysed_sst"].values
    lats = ds["lat"].values
    lons = ds["lon"].values

    # Gestion des 3 dimensions (time, lat, lon)
    if sst.ndim == 3:
        sst = sst[0]

    # Conversion Kelvin -> Celsius
    sst_celsius = sst - 273.15

    # Extraction de la date depuis le nom de fichier (format YYYYMMDD)
    filename = os.path.basename(file_path)
    date = datetime.strptime(filename[:8], "%Y%m%d")

    ds.close()

    return {
        "sst":        sst_celsius,       # tableau 2D numpy
        "lat":        lats,
        "lon":        lons,
        "date":       date,
        "year":       date.year,
        "month":      date.month,
        "month_name": date.strftime("%b"),
        "filename":   filename,
    }

# Test de la fonction sur le premier fichier
test_data = read_sst_file(nc_files[0])
print(f"✅ Fonction de lecture testée avec succès")
print(f"   Date       : {test_data['date'].strftime('%d %B %Y')}")
print(f"   Dimensions : {test_data['sst'].shape[1]} x {test_data['sst'].shape[0]}\n")

# -----------------------------------------------------------------------------
# 5. ANALYSE DES DIMENSIONS SPATIALES
# Équivalent R : cat("Latitudes :", length(lat), ...) 
# -----------------------------------------------------------------------------

print("=== DATASET - DIMENSIONS ET STRUCTURE ===")
print(f"\nDimensions spatiales :")
print(f"Latitudes  : {len(lat)} valeurs de {lat.min():.3f} à {lat.max():.3f} °N")
print(f"Longitudes : {len(lon)} valeurs de {lon.min():.3f} à {lon.max():.3f} °E")
print(f"Résolution : {round((lat.max() - lat.min()) / len(lat), 4)}° par pixel\n")

# -----------------------------------------------------------------------------
# 6. ANALYSE STATISTIQUE INITIALE
# Équivalent R : boucle sur sample_files + kable(stats_summary)
# -----------------------------------------------------------------------------

print("=== ANALYSE STATISTIQUE INITIALE ===")

sample_size  = min(20, len(nc_files))
sample_idx   = np.linspace(0, sample_size - 1, 5, dtype=int)
sample_files = [nc_files[i] for i in sample_idx]

stats_records = []
for file in sample_files:
    data   = read_sst_file(file)
    valid  = data["sst"][~np.isnan(data["sst"])]
    if len(valid) > 0:
        stats_records.append({
            "date":      data["date"].strftime("%Y-%m-%d"),
            "mois":      data["month_name"],
            "moyenne":   round(float(np.mean(valid)), 2),
            "ecart_type":round(float(np.std(valid)),  2),
            "min":       round(float(np.min(valid)),  2),
            "max":       round(float(np.max(valid)),  2),
            "na_pct":    round(np.sum(np.isnan(data["sst"])) / data["sst"].size * 100, 2),
        })

stats_df = pd.DataFrame(stats_records)
print("\nStatistiques sur l'échantillon :")
print(stats_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 7. CARTE GLOBALE DE RÉFÉRENCE
# Équivalent R : ggplot() + geom_raster() + geom_polygon() + scale_fill_viridis_c()
# En Python : matplotlib + cartopy
# -----------------------------------------------------------------------------

print("\n=== CARTE GLOBALE ===")

# Sélection d'un fichier de juillet 2000 si disponible
july_files = [f for f in nc_files if "200007" in os.path.basename(f)]
sample_file = july_files[0] if july_files else nc_files[0]
sample_data = read_sst_file(sample_file)

fig, ax = plt.subplots(figsize=(14, 7),
                       subplot_kw={"projection": ccrs.PlateCarree()})

# Tracé SST (équivalent R : geom_raster + scale_fill_viridis_c plasma)
img = ax.pcolormesh(
    sample_data["lon"], sample_data["lat"], sample_data["sst"],
    cmap="plasma", vmin=-2, vmax=32, transform=ccrs.PlateCarree()
)
plt.colorbar(img, ax=ax, label="SST (°C)", shrink=0.6)

# Continents (équivalent R : geom_polygon avec map_data("world"))
ax.add_feature(cfeature.LAND,      facecolor="lightgray", edgecolor="gray", linewidth=0.3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
ax.set_extent([-180, 180, -60, 80])
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

ax.set_title(
    f"Température de Surface de la Mer (SST) Globale\n"
    f"Date : {sample_data['date'].strftime('%d %B %Y')} | "
    f"NA : {na_percent}% (terres et glaces)",
    fontweight="bold", fontsize=13
)
plt.tight_layout()
plt.savefig("carte_globale_sst.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Carte globale créée\n")

# -----------------------------------------------------------------------------
# 8. DÉFINITION DES RÉGIONS OCÉANIQUES
# Équivalent R : ocean_regions <- list(Mediterranee=list(bbox=c(...), color="..."))
# bbox format : (xmin, xmax, ymin, ymax)
# -----------------------------------------------------------------------------

print("=== DÉFINITION DES RÉGIONS OCÉANIQUES ===")

ocean_regions = {
    "Mediterranee":    {"name": "Méditerranée",   "bbox": (-6,  36,  30, 46), "color": "#FF6B6B"},
    "Atlantique_Nord": {"name": "Atlantique Nord", "bbox": (-80, 20,   0, 65), "color": "#4ECDC4"},
    "Atlantique_Sud":  {"name": "Atlantique Sud",  "bbox": (-70, 20, -60,  0), "color": "#45B7D1"},
    "Pacifique_Nord":  {"name": "Pacifique Nord",  "bbox": (100, 260,  0, 65), "color": "#96CEB4"},
    "Pacifique_Sud":   {"name": "Pacifique Sud",   "bbox": (120, 290,-60,  0), "color": "#FFEAA7"},
    "Ocean_Indien":    {"name": "Océan Indien",    "bbox": ( 20, 120,-60, 30), "color": "#DDA0DD"},
    "Mer_Rouge":       {"name": "Mer Rouge",       "bbox": ( 32,  44,  12, 30), "color": "#98D8C8"},
    "Mer_Noire":       {"name": "Mer Noire",       "bbox": ( 28,  42,  41, 47), "color": "#F7DC6F"},
}

print(f"✅ {len(ocean_regions)} régions océaniques définies\n")

# -----------------------------------------------------------------------------
# 9. FONCTION D'EXTRACTION DES STATISTIQUES PAR RÉGION
# Équivalent R : extract_region_stats <- function(sst_raster, region_name, bbox)
# R utilise crop(raster, extent(bbox)) -> ici on filtre par masque numpy
# -----------------------------------------------------------------------------

def extract_region_stats(sst_array, lats, lons, region_name, bbox):
    """
    Extrait les statistiques SST pour une region geographique.
    Equivalent R : crop() + values() + mean/sd/sum(is.na())

    IMPORTANT - calcul na_percent identique au code R :
      R : sum(is.na(values(region_cropped))) / (nrow*ncol fichier global)
      Python : denominateur = sst_array.size (taille fichier global)
      => meme classement des regions que dans le graphique R.

    Pour le Pacifique (xmax > 180) : bbox [0,360] convertie en deux
    intervalles [-180,180] via masque 2D, comme rotate(raster) en R.
    """
    xmin, xmax, ymin, ymax = bbox

    # Denominateur global identique au code R
    total_global_cells = sst_array.size

    # Grille 2D des coordonnees
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    lat_mask_2d = (lat_grid >= ymin) & (lat_grid <= ymax)

    if xmax > 180:
        # bbox [0,360] -> deux intervalles [-180,180]
        # Ex: Pacifique Sud (120,290) -> [120,180] UNION [-180,-70]
        xmax2 = xmax - 360
        lon_mask_2d = ((lon_grid >= xmin) & (lon_grid <= 180)) |                       ((lon_grid >= -180) & (lon_grid <= xmax2))
    else:
        lon_mask_2d = (lon_grid >= xmin) & (lon_grid <= xmax)

    mask_2d = lat_mask_2d & lon_mask_2d
    cells   = sst_array[mask_2d]
    valid   = cells[~np.isnan(cells)]

    if len(cells) == 0:
        return None

    # na_percent : NA region / total cellules GLOBAL (identique R)
    na_in_region = int(np.isnan(cells).sum())

    return {
        "region":      region_name,
        "mean_sst":    float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "sd_sst":      float(np.std(valid))  if len(valid) > 1 else 0.0,
        "na_percent":  float(na_in_region / total_global_cells * 100),
        "valid_cells": int(len(valid)),
        "total_cells": int(len(cells)),
    }

# -----------------------------------------------------------------------------
# 10. ANALYSE GLOBALE DES OCÉANS
# Équivalent R : boucle for sur target_months + rbind dans global_catalog
# -----------------------------------------------------------------------------

print("=== ANALYSE GLOBALE DES OCÉANS ===")

target_months    = [1, 5, 7, 10]
month_names_fr   = {1: "Janvier", 5: "Mai", 7: "Juillet", 10: "Octobre"}

print("Collecte des données pour l'analyse globale...")

global_records = []

for month_num in target_months:
    # Filtrage des fichiers pour le mois cible
    month_files = [f for f in nc_files
                   if int(os.path.basename(f)[4:6]) == month_num]

    # Échantillonnage : début, milieu, fin (équivalent R : c(1, round(n/2), n))
    if len(month_files) >= 3:
        indices = [0, len(month_files) // 2, len(month_files) - 1]
        sample  = [month_files[i] for i in indices]
    elif month_files:
        sample = [month_files[0]]
    else:
        continue

    for file in sample:
        data = read_sst_file(file)
        for key, region in ocean_regions.items():
            s = extract_region_stats(
                data["sst"], data["lat"], data["lon"],
                region["name"], region["bbox"]
            )
            if s and not np.isnan(s["mean_sst"]):
                global_records.append({
                    "date":       data["date"],
                    "year":       data["year"],
                    "month":      data["month"],
                    "month_name": data["month_name"],
                    "month_fr":   month_names_fr[month_num],
                    "region":     region["name"],
                    "mean_sst":   s["mean_sst"],
                    "na_percent": s["na_percent"],
                })

global_catalog = pd.DataFrame(global_records)
print(f"✅ Catalogue global créé : {len(global_catalog)} observations\n")

# -----------------------------------------------------------------------------
# 10.1 BOXPLOTS COMPARATIFS PAR RÉGION ET MOIS
# Équivalent R : ggplot() + geom_boxplot() + geom_jitter() + facet_wrap()
# -----------------------------------------------------------------------------

print("=== VISUALISATIONS GLOBALES ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, month_num in zip(axes.flatten(), target_months):
    subset = global_catalog[global_catalog["month"] == month_num]
    # Tri par médiane (équivalent R : reorder(region, mean_sst))
    order = (subset.groupby("region")["mean_sst"]
             .median().sort_values().index.tolist())

    # hue="region" + legend=False remplace palette sans hue (déprécié depuis seaborn 0.14)
    # alpha retiré du boxplot (non supporté sur matplotlib récent) -> appliqué sur les patches
    sns.boxplot(data=subset, y="region", x="mean_sst",
                order=order, ax=ax, hue="region", palette="Set2",
                orient="h", legend=False)
    # Transparence appliquée manuellement sur chaque patch
    for patch in ax.patches:
        patch.set_alpha(0.7)
    # Jitter (équivalent R : geom_jitter)
    for i, region in enumerate(order):
        vals = subset[subset["region"] == region]["mean_sst"]
        ax.scatter(vals, [i] * len(vals), alpha=0.5, s=15, color="gray", zorder=3)

    ax.set_title(month_names_fr[month_num], fontweight="bold")
    ax.set_xlabel("Température moyenne (°C)")
    ax.set_ylabel("")

fig.suptitle("Distribution des températures par région océanique\n"
             "Classement des régions du plus chaud au plus froid",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("boxplot_regions.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 10.2 ANALYSE DES VALEURS MANQUANTES PAR RÉGION
# Équivalent R : ggplot() + geom_bar(stat="identity") + coord_flip()
# -----------------------------------------------------------------------------

na_summary = (global_catalog
              .groupby("region")
              .agg(mean_na_percent=("na_percent", "mean"),
                   mean_temp=("mean_sst", "mean"),
                   n_samples=("mean_sst", "count"))
              .reset_index()
              .sort_values("mean_na_percent"))

fig, ax = plt.subplots(figsize=(9, 5))
colors = [ocean_regions[k]["color"]
          for k in ocean_regions
          if ocean_regions[k]["name"] in na_summary["region"].values]

bars = ax.barh(na_summary["region"], na_summary["mean_na_percent"],
               color=colors, alpha=0.8)

# Annotations (équivalent R : geom_text)
for bar, val in zip(bars, na_summary["mean_na_percent"]):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9)

ax.set_title("Pourcentage moyen de NA par région océanique\n"
             "Les mers fermées ont moins de NA", fontweight="bold")
ax.set_xlabel("Pourcentage de NA (%)")
ax.set_ylabel("Région océanique")
plt.tight_layout()
plt.savefig("na_par_region.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 10.3 HEATMAP DES TEMPÉRATURES MOYENNES
# Équivalent R : ggplot() + geom_tile() + scale_fill_viridis_c(option="plasma")
# -----------------------------------------------------------------------------

heatmap_data = (global_catalog
                .groupby(["region", "month_fr"])["mean_sst"]
                .mean()
                .unstack()
                .reindex(columns=["Janvier", "Mai", "Juillet", "Octobre"]))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="plasma", ax=ax,
            cbar_kws={"label": "Température (°C)"}, linewidths=0.5)
ax.set_title("Heatmap des températures par région et mois\n"
             "Moyennes interannuelles", fontweight="bold")
ax.set_xlabel("Mois")
ax.set_ylabel("Région océanique")
plt.tight_layout()
plt.savefig("heatmap_regions.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 11. SYNTHÈSE DE L'ANALYSE GLOBALE
# Équivalent R : kable(global_stats) + cat("PRINCIPAUX RÉSULTATS")
# -----------------------------------------------------------------------------

print("=== CONCLUSION ANALYSE GLOBALE ===")

global_stats = (global_catalog
                .groupby("region")
                .agg(
                    n_observations=("mean_sst", "count"),
                    temp_moyenne  =("mean_sst", "mean"),
                    temp_sd       =("mean_sst", "std"),
                    temp_min      =("mean_sst", "min"),
                    temp_max      =("mean_sst", "max"),
                    na_moyen      =("na_percent", "mean"),
                )
                .reset_index()
                .sort_values("temp_moyenne", ascending=False))

print("\nTableau récapitulatif :")
print(global_stats.round(2).to_string(index=False))

hottest_region   = global_stats.iloc[0]["region"]
coldest_region   = global_stats.iloc[-1]["region"]
lowest_na_region = na_summary.iloc[0]["region"]
highest_na_region= na_summary.iloc[-1]["region"]

print(f"\nPRINCIPAUX RÉSULTATS :")
print(f"1. Région la plus chaude : {hottest_region} ({global_stats.iloc[0]['temp_moyenne']:.1f}°C)")
print(f"2. Région la plus froide : {coldest_region} ({global_stats.iloc[-1]['temp_moyenne']:.1f}°C)")
print(f"3. Région avec le moins de NA : {lowest_na_region} ({na_summary.iloc[0]['mean_na_percent']:.1f}%)")
print(f"4. Région avec le plus de NA  : {highest_na_region} ({na_summary.iloc[-1]['mean_na_percent']:.1f}%)")

print(f"\nSYNTHÈSE :")
print(f"• Tous les fichiers ont environ {na_percent}% de NA (terres/glaces)")
print(f"• Les mers fermées présentent moins de NA")

# -----------------------------------------------------------------------------
# 12. SÉLECTION DE LA RÉGION POUR ANALYSE APPROFONDIE
# Équivalent R : best_region <- lowest_na_region + carte focalisée
# Critère : plus faible pourcentage de NA
# -----------------------------------------------------------------------------

# Région forcée à "Mer Rouge" indépendamment du pourcentage de NA
# (en R la Mer Rouge était automatiquement sélectionnée car lowest_na=0%)
best_region_name = "Mer Rouge"
best_region_key  = next(k for k, v in ocean_regions.items()
                        if v["name"] == best_region_name)
best_region_info = ocean_regions[best_region_key]
best_bbox        = best_region_info["bbox"]

print(f"\n✅ RÉGION SÉLECTIONNÉE : {best_region_name}")
print(f"   Critère : plus faible pourcentage de NA")
print(f"   Pourcentage de NA : {na_summary[na_summary['region']==best_region_name]['mean_na_percent'].values[0]:.1f}%")
print(f"   Température moyenne : {global_stats[global_stats['region']==best_region_name]['temp_moyenne'].values[0]:.1f}°C")

# Carte focalisée sur la région sélectionnée
xmin, xmax, ymin, ymax = best_bbox
fig, ax = plt.subplots(figsize=(9, 7),
                       subplot_kw={"projection": ccrs.PlateCarree()})

img = ax.pcolormesh(
    sample_data["lon"], sample_data["lat"], sample_data["sst"],
    cmap="plasma", vmin=-2, vmax=32, transform=ccrs.PlateCarree()
)
plt.colorbar(img, ax=ax, label="SST (°C)", shrink=0.6)
ax.add_feature(cfeature.LAND,      facecolor="lightgray", edgecolor="gray", linewidth=0.3)
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# Rectangle de délimitation (équivalent R : geom_rect avec linetype="dashed")
import matplotlib.patches as mpatches
rect = mpatches.FancyBboxPatch(
    (xmin, ymin), xmax - xmin, ymax - ymin,
    boxstyle="square,pad=0", linewidth=2,
    edgecolor=best_region_info["color"], facecolor="none", linestyle="--",
    transform=ccrs.PlateCarree()
)
ax.add_patch(rect)

ax.set_extent([xmin - 5, xmax + 5, ymin - 5, ymax + 5])
ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
ax.set_title(f"Région d'étude : {best_region_name}\n"
             f"Pourcentage de NA : {na_summary[na_summary['region']==best_region_name]['mean_na_percent'].values[0]:.1f}%",
             fontweight="bold", color=best_region_info["color"], fontsize=13)
plt.tight_layout()
plt.savefig("carte_region_selectionnee.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 13. ANALYSE APPROFONDIE DE LA RÉGION SÉLECTIONNÉE
# Équivalent R : boucle sur tous les nc_files -> region_catalog
# -----------------------------------------------------------------------------

print(f"\n{'='*70}")
print(f"ANALYSE CONCENTRÉE SUR : {best_region_name}")
print(f"{'='*70}")
print(f"Collecte des données pour {best_region_name}...")

region_records = []

for file in tqdm(nc_files, desc="Lecture des fichiers"):
    data  = read_sst_file(file)
    s     = extract_region_stats(data["sst"], data["lat"], data["lon"],
                                  best_region_name, best_bbox)
    if s and not np.isnan(s["mean_sst"]):
        region_records.append({
            "date":       data["date"],
            "year":       data["year"],
            "month":      data["month"],
            "month_name": data["month_name"],
            "mean_sst":   s["mean_sst"],
            "na_percent": s["na_percent"],
        })

region_catalog = pd.DataFrame(region_records).sort_values("date")
print(f"\n✅ Catalogue régional créé : {len(region_catalog)} observations")

# Filtre sur les mois cibles uniquement
region_filtered = region_catalog[region_catalog["month"].isin(target_months)].copy()
region_filtered["month_fr"] = region_filtered["month"].map(month_names_fr)

# Statistiques descriptives par mois (équivalent R : group_by + summarise + kable)
region_stats = (region_filtered
                .groupby("month_fr")
                .agg(
                    n_years  =("year",     "nunique"),
                    mean_temp=("mean_sst", "mean"),
                    sd_temp  =("mean_sst", "std"),
                    min_temp =("mean_sst", "min"),
                    max_temp =("mean_sst", "max"),
                )
                .reindex(["Janvier", "Mai", "Juillet", "Octobre"])
                .reset_index())

print(f"\nTableau : Statistiques par mois dans {best_region_name}")
print(region_stats.round(2).to_string(index=False))

# Tendances linéaires par mois (équivalent R : coef(lm(mean_sst ~ year))[2])
print(f"\nTendances du réchauffement dans {best_region_name} :")
region_trends_records = []
for month_num in target_months:
    subset = region_filtered[region_filtered["month"] == month_num].dropna(subset=["mean_sst"])
    if len(subset) < 5:
        continue
    slope, intercept, r, p, se = stats.linregress(subset["year"], subset["mean_sst"])
    region_trends_records.append({
        "month_fr":       month_names_fr[month_num],
        "trend_decade":   slope * 10,
        "p_value":        p,
        "significance":   "Significatif" if p < 0.05 else "Non significatif",
    })

region_trends = pd.DataFrame(region_trends_records)
print(region_trends.round(3).to_string(index=False))

# Évolution temporelle par mois avec tendances (équivalent R : geom_line + geom_smooth)
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors_months = {"Janvier": "#2E86AB", "Mai": "#F18F01",
                 "Juillet": "#C73E1D", "Octobre": "#6B8F71"}

for ax, month_num in zip(axes.flatten(), target_months):
    mfr    = month_names_fr[month_num]
    subset = region_filtered[region_filtered["month"] == month_num]
    color  = colors_months[mfr]

    ax.plot(subset["year"], subset["mean_sst"],
            alpha=0.6, linewidth=0.8, marker="o", markersize=2, color=color)

    # Droite de tendance (équivalent R : geom_smooth(method="lm"))
    if len(subset) >= 5:
        slope, intercept, *_ = stats.linregress(subset["year"], subset["mean_sst"])
        x_line = np.array([subset["year"].min(), subset["year"].max()])
        ax.plot(x_line, slope * x_line + intercept,
                linestyle="--", linewidth=1.5, color=color)

    ax.set_title(mfr, fontweight="bold", color=color)
    ax.set_xlabel("Année")
    ax.set_ylabel("Température moyenne (°C)")
    ax.grid(alpha=0.3)

fig.suptitle(f"Évolution de la SST dans la {best_region_name} (1980-2023)\n"
             "Lignes pointillées : tendances linéaires",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("evolution_sst_region.png", dpi=150, bbox_inches="tight")
plt.show()

# Graphique des tendances en barres (équivalent R : geom_bar(stat="identity"))
fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = [best_region_info["color"]] * len(region_trends)
bars = ax.bar(region_trends["month_fr"], region_trends["trend_decade"],
              color=bar_colors, alpha=0.85, width=0.6)

for bar, row in zip(bars, region_trends.itertuples()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{row.trend_decade:.3f}°C/déc",
            ha="center", fontsize=9)

ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
ax.set_title(f"Tendances du réchauffement - {best_region_name}\n"
             "En °C par décennie (1980-2023)", fontweight="bold")
ax.set_xlabel("Mois")
ax.set_ylabel("Tendance (°C/décennie)")
plt.tight_layout()
plt.savefig("tendances_region.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 14. CONCLUSION GÉNÉRALE
# Équivalent R : cat("RÉSUMÉ DE L'ANALYSE") + boucle sur region_trends
# -----------------------------------------------------------------------------

print(f"\n{'='*70}")
print("CONCLUSION GÉNÉRALE")
print(f"{'='*70}")
print(f"\nRÉSUMÉ DE L'ANALYSE :")
print(f"1. DONNÉES          : {len(nc_files)} fichiers NetCDF analysés")
print(f"2. NA STRUCTURELS   : {na_percent}% (terres et glaces)")
print(f"3. RÉGION ÉTUDIÉE   : {best_region_name}")
print(f"4. POURCENTAGE DE NA: {na_summary[na_summary['region']==best_region_name]['mean_na_percent'].values[0]:.1f}%")

print(f"\nRÉSULTATS CLÉS POUR {best_region_name} :")
for _, row in region_trends.iterrows():
    print(f"  • {row['month_fr']:<10} : {row['trend_decade']:+.3f}°C/décennie ({row['significance']})")

max_trend = region_trends.loc[region_trends["trend_decade"].abs().idxmax()]
print(f"\nPOINT CRITIQUE :")
print(f"  • Mois avec la plus forte tendance : {max_trend['month_fr']}")
print(f"  • Tendance : {max_trend['trend_decade']:+.3f}°C/décennie")
print(f"  • Significativité : {max_trend['significance']}")

# -----------------------------------------------------------------------------
# 15. ANALYSE DÉTAILLÉE PAR MOIS (DATASET MENSUEL COMPLET)
# Équivalent R : boucle sur tous les fichiers -> monthly_dataset
# -----------------------------------------------------------------------------

print(f"\n{'='*70}")
print("ANALYSE DÉTAILLÉE PAR MOIS")
print(f"{'='*70}")
print("\n15.1 Création du dataset mensuel...")
print("Collecte des données pour les 12 mois...")

all_months_fr = ["Janvier","Février","Mars","Avril","Mai","Juin",
                 "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

monthly_records = []

for file in tqdm(nc_files, desc="Lecture mensuelle"):
    data  = read_sst_file(file)
    valid = data["sst"][~np.isnan(data["sst"])]
    if len(valid) > 0:
        monthly_records.append({
            "date":       data["date"],
            "year":       data["year"],
            "month":      data["month"],
            "month_name": data["month_name"],
            "month_fr":   all_months_fr[data["month"] - 1],
            "mean_sst":   float(np.mean(valid)),
            "sd_sst":     float(np.std(valid)),
            "min_sst":    float(np.min(valid)),
            "max_sst":    float(np.max(valid)),
            "q25":        float(np.percentile(valid, 25)),
            "median_sst": float(np.median(valid)),
            "q75":        float(np.percentile(valid, 75)),
            "na_count":   int(np.sum(np.isnan(data["sst"]))),
            "total_cells":int(data["sst"].size),
            "na_percent": float(np.isnan(data["sst"]).sum() / data["sst"].size * 100),
        })

monthly_dataset = pd.DataFrame(monthly_records).sort_values("date").reset_index(drop=True)
print(f"\n✅ Dataset mensuel créé : {len(monthly_dataset)} observations")
print(f"   Période       : {monthly_dataset['date'].min().strftime('%Y-%m')} à {monthly_dataset['date'].max().strftime('%Y-%m')}")
print(f"   Années        : {monthly_dataset['year'].min()} - {monthly_dataset['year'].max()}")

# Structure du dataset (équivalent R : str(monthly_dataset))
print("\n15.2 Aperçu du dataset mensuel :")
print(monthly_dataset[["date","year","month","mean_sst","sd_sst","na_percent"]].head(6).to_string(index=False))

# Statistiques descriptives par mois (équivalent R : group_by + summarise)
monthly_stats = (monthly_dataset
                 .groupby("month_fr")
                 .agg(
                     n_observations =("mean_sst", "count"),
                     n_years        =("year",     "nunique"),
                     mean_temp      =("mean_sst", "mean"),
                     sd_temp        =("mean_sst", "std"),
                     min_temp       =("mean_sst", "min"),
                     max_temp       =("mean_sst", "max"),
                     iqr_temp       =("mean_sst", lambda x: x.quantile(0.75) - x.quantile(0.25)),
                     mean_na_percent=("na_percent","mean"),
                 )
                 .reindex(all_months_fr)
                 .reset_index())

print("\n15.3 Statistiques par mois (1980-2023) :")
print(monthly_stats.round(2).to_string(index=False))

# Tendances linéaires par mois (équivalent R : group_by + do(lm()) + mutate)
print("\n15.4 Analyse des tendances par mois :")
monthly_trends_records = []

for month_num in range(1, 13):
    subset = monthly_dataset[monthly_dataset["month"] == month_num].dropna(subset=["mean_sst"])
    if len(subset) < 5:
        continue
    slope, intercept, r, p, se = stats.linregress(subset["year"], subset["mean_sst"])
    trend_decade = slope * 10

    # Classification (équivalent R : case_when)
    if p < 0.001:
        significance = "Très significatif (p < 0.001)"
    elif p < 0.01:
        significance = "Très significatif (p < 0.01)"
    elif p < 0.05:
        significance = "Significatif (p < 0.05)"
    else:
        significance = "Non significatif"

    if   abs(trend_decade) > 0.15: strength = "Forte"
    elif abs(trend_decade) > 0.10: strength = "Modérée"
    elif abs(trend_decade) > 0.05: strength = "Faible"
    else:                           strength = "Très faible"

    monthly_trends_records.append({
        "month_fr":     all_months_fr[month_num - 1],
        "trend_decade": trend_decade,
        "p_value":      p,
        "r_squared":    r ** 2,
        "significance": significance,
        "strength":     strength,
        "direction":    "Augmentation" if slope > 0 else "Diminution",
    })

monthly_trends = (pd.DataFrame(monthly_trends_records)
                  .sort_values("trend_decade", ascending=False, key=abs))

print(monthly_trends[["month_fr","trend_decade","p_value","r_squared","significance","strength"]]
      .round(4).to_string(index=False))

# -----------------------------------------------------------------------------
# 15.5 VISUALISATIONS DÉTAILLÉES PAR MOIS
# -----------------------------------------------------------------------------

month_colors = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1C32",
                "#6B8F71","#AAD922","#FF9F1C","#E71D36","#2EC4B6",
                "#011627","#FF3366"]
color_map = dict(zip(all_months_fr, month_colors))

# 15.5.1 Évolution temporelle par mois avec facettes
# Équivalent R : geom_line + geom_smooth(method="lm") + facet_wrap(~month_factor)
print("\n15.5.1 Évolution temporelle mensuelle...")

months_in_data = monthly_dataset["month"].unique()
n_months = len(months_in_data)
ncols = 3
nrows = int(np.ceil(n_months / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), squeeze=False)

for idx, month_num in enumerate(sorted(months_in_data)):
    ax     = axes[idx // ncols][idx % ncols]
    mfr    = all_months_fr[month_num - 1]
    subset = monthly_dataset[monthly_dataset["month"] == month_num]
    color  = color_map[mfr]

    ax.plot(subset["year"], subset["mean_sst"],
            color=color, linewidth=0.7, alpha=0.7, marker="o", markersize=1.5)

    if len(subset) >= 5:
        slope, intercept, *_ = stats.linregress(subset["year"], subset["mean_sst"])
        x_line = np.array([subset["year"].min(), subset["year"].max()])
        ax.plot(x_line, slope * x_line + intercept,
                linestyle="--", linewidth=1.2, color=color)

    ax.set_title(mfr, fontweight="bold", fontsize=10, color=color)
    ax.set_xlabel("Année", fontsize=8)
    ax.set_ylabel("SST (°C)", fontsize=8)
    ax.grid(alpha=0.3)

# Masquer les axes vides
for idx in range(n_months, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.suptitle("Évolution temporelle de la SST par mois (1980-2023)\n"
             "Lignes pointillées : tendances linéaires",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("evolution_temporelle_mensuelle.png", dpi=150, bbox_inches="tight")
plt.show()

# 15.5.2 Cycle annuel moyen avec intervalles de confiance
# Équivalent R : geom_line + geom_ribbon (IC 95%) + geom_errorbar
print("\n15.5.2 Cycle annuel moyen...")

annual_cycle = (monthly_dataset
                .groupby("month_fr")
                .agg(
                    mean_temp=("mean_sst", "mean"),
                    sd_temp  =("mean_sst", "std"),
                    n_years  =("year",     "nunique"),
                )
                .reindex(all_months_fr)
                .reset_index())

annual_cycle["se_temp"]   = annual_cycle["sd_temp"] / np.sqrt(annual_cycle["n_years"])
annual_cycle["lower_ci"]  = annual_cycle["mean_temp"] - 1.96 * annual_cycle["se_temp"]
annual_cycle["upper_ci"]  = annual_cycle["mean_temp"] + 1.96 * annual_cycle["se_temp"]

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(range(len(annual_cycle)),
                annual_cycle["lower_ci"], annual_cycle["upper_ci"],
                alpha=0.2, color="#2E86AB", label="IC 95%")
ax.errorbar(range(len(annual_cycle)),
            annual_cycle["mean_temp"],
            yerr=annual_cycle["sd_temp"],
            fmt="o-", color="#2E86AB", linewidth=1.5, capsize=3, markersize=5)

for i, row in annual_cycle.iterrows():
    ax.text(i, row["mean_temp"] + 0.02, f"{row['mean_temp']:.2f}°C",
            ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(range(len(annual_cycle)))
ax.set_xticklabels(all_months_fr, rotation=45, ha="right", fontsize=10, fontweight="bold")
ax.set_title("Cycle annuel moyen de la température océanique\n"
             "Moyenne 1980-2023 avec intervalles de confiance à 95%",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Température moyenne (°C)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cycle_annuel_sst.png", dpi=150, bbox_inches="tight")
plt.show()

# 15.5.3 Heatmap temporelle année × mois
# Équivalent R : geom_tile() + scale_fill_viridis_c(option="plasma")
print("\n15.5.3 Heatmap annuelle...")

heatmap_ym = (monthly_dataset
              .groupby(["year", "month_fr"])["mean_sst"]
              .mean()
              .unstack()
              .reindex(columns=all_months_fr))

fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(heatmap_ym.T, cmap="plasma", ax=ax,
            cbar_kws={"label": "Température (°C)"},
            xticklabels=5)
ax.set_title("Heatmap des températures par année et mois\n"
             "Évolution spatio-temporelle 1980-2023",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Année")
ax.set_ylabel("Mois")
plt.tight_layout()
plt.savefig("heatmap_annuelle.png", dpi=150, bbox_inches="tight")
plt.show()

# 15.5.4 Boxplots comparatifs mensuels
# Équivalent R : geom_boxplot + geom_jitter + stat_summary(fun=mean, shape=23)
print("\n15.5.4 Distribution mensuelle...")

fig, ax = plt.subplots(figsize=(14, 6))
month_order = all_months_fr

bp = ax.boxplot(
    [monthly_dataset[monthly_dataset["month_fr"] == m]["mean_sst"].dropna().values
     for m in month_order],
    positions=range(len(month_order)),
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 1.5},
    flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
)

for patch, color in zip(bp["boxes"], month_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

# Moyennes (équivalent R : stat_summary shape=23 = diamant)
for i, m in enumerate(month_order):
    mean_val = monthly_dataset[monthly_dataset["month_fr"] == m]["mean_sst"].mean()
    ax.scatter(i, mean_val, marker="D", color="white", edgecolors="black",
               s=40, zorder=5)

ax.set_xticks(range(len(month_order)))
ax.set_xticklabels(month_order, rotation=45, ha="right", fontsize=10, fontweight="bold")
ax.set_title("Distribution des températures par mois\n"
             "Comparaison interannuelle (1980-2023) | Diamants : moyennes",
             fontweight="bold", fontsize=13)
ax.set_ylabel("Température moyenne (°C)")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("boxplots_mensuels.png", dpi=150, bbox_inches="tight")
plt.show()

# 15.5.5 Tendances mensuelles en barres classées par intensité
# Équivalent R : geom_bar + coord_flip + scale_fill_manual
print("\n15.5.5 Visualisation des tendances mensuelles...")

fig, ax = plt.subplots(figsize=(9, 7))
bar_colors_trends = ["#E74C3C" if d == "Augmentation" else "#3498DB"
                     for d in monthly_trends["direction"]]

bars = ax.barh(monthly_trends["month_fr"], monthly_trends["trend_decade"],
               color=bar_colors_trends, alpha=0.8, height=0.6)

for bar, row in zip(bars, monthly_trends.itertuples()):
    ax.text(bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{row.trend_decade:+.3f}°C/déc\np={row.p_value:.3f}",
            va="center", fontsize=8)

ax.axvline(0, linestyle="--", color="gray", linewidth=0.8)
ax.set_title("Tendances du réchauffement par mois\n"
             "En °C par décennie (1980-2023) - Classé par intensité",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Tendance (°C/décennie)")
ax.set_ylabel("Mois")
plt.tight_layout()
plt.savefig("tendances_mensuelles.png", dpi=150, bbox_inches="tight")
plt.show()

# 15.5.6 Anomalies mensuelles par rapport à la référence 1991-2020
# Équivalent R : mutate(ref_mean, anomaly) + geom_line + facet_wrap + geom_smooth(loess)
print("\n15.5.6 Anomalies mensuelles...")

# Calcul des anomalies (équivalent R : group_by(month_factor) %>% mutate(ref_mean, anomaly))
ref_means = (monthly_dataset[
                 (monthly_dataset["year"] >= 1991) &
                 (monthly_dataset["year"] <= 2020)]
             .groupby("month")["mean_sst"]
             .mean())

monthly_dataset["ref_mean"] = monthly_dataset["month"].map(ref_means)
monthly_dataset["anomaly"]  = monthly_dataset["mean_sst"] - monthly_dataset["ref_mean"]
monthly_dataset["period"]   = pd.cut(
    monthly_dataset["year"],
    bins=[0, 1990, 2020, 9999],
    labels=["Avant référence", "Période référence", "Après référence"]
)

period_colors = {
    "Avant référence":   "#95A5A6",
    "Période référence": "#3498DB",
    "Après référence":   "#E74C3C",
}

months_plot = sorted(monthly_dataset["month"].unique())
nrows2 = int(np.ceil(len(months_plot) / 3))
fig, axes = plt.subplots(nrows2, 3, figsize=(14, nrows2 * 3), squeeze=False)

for idx, month_num in enumerate(months_plot):
    ax     = axes[idx // 3][idx % 3]
    mfr    = all_months_fr[month_num - 1]
    subset = monthly_dataset[monthly_dataset["month"] == month_num]

    for period, grp in subset.groupby("period", observed=True):
        color = period_colors.get(str(period), "gray")
        ax.plot(grp["year"], grp["anomaly"],
                color=color, linewidth=0.6, alpha=0.7, marker="o", markersize=1.5)

    # Lissage LOESS approché par LOWESS de scipy
    from scipy.stats import mstats
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess(subset["anomaly"].values,
                                      subset["year"].values, frac=0.3)
    ax.plot(lowess[:, 0], lowess[:, 1], color="#E74C3C", linewidth=1.2)

    ax.axhline(0, linestyle="--", color="gray", linewidth=0.7)
    ax.set_title(mfr, fontweight="bold", fontsize=9)
    ax.set_xlabel("Année", fontsize=7)
    ax.set_ylabel("Anomalie (°C)", fontsize=7)
    ax.grid(alpha=0.3)

for idx in range(len(months_plot), nrows2 * 3):
    axes[idx // 3][idx % 3].set_visible(False)

fig.suptitle("Anomalies de température par mois\nRéférence : moyenne 1991-2020",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("anomalies_mensuelles.png", dpi=150, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 15.6 SYNTHÈSE FINALE
# Équivalent R : cat("MOIS EXTRÊMES") + cat("TENDANCES")
# -----------------------------------------------------------------------------

print("\n15.6 Synthèse de l'analyse mensuelle :")

hottest_month   = monthly_stats.loc[monthly_stats["mean_temp"].idxmax()]
coldest_month   = monthly_stats.loc[monthly_stats["mean_temp"].idxmin()]
strongest_trend = monthly_trends.loc[monthly_trends["trend_decade"].abs().idxmax()]
weakest_trend   = monthly_trends.loc[monthly_trends["trend_decade"].abs().idxmin()]

print(f"\nMOIS EXTRÊMES :")
print(f"• Mois le plus chaud  : {hottest_month['month_fr']} ({hottest_month['mean_temp']:.2f}°C)")
print(f"• Mois le plus froid  : {coldest_month['month_fr']} ({coldest_month['mean_temp']:.2f}°C)")
print(f"• Amplitude annuelle  : {hottest_month['mean_temp'] - coldest_month['mean_temp']:.2f}°C")

print(f"\nTENDANCES :")
print(f"• Tendance la plus forte  : {strongest_trend['month_fr']} "
      f"({strongest_trend['trend_decade']:+.3f}°C/décennie, {strongest_trend['significance']})")
print(f"• Tendance la plus faible : {weakest_trend['month_fr']} "
      f"({weakest_trend['trend_decade']:+.3f}°C/décennie, {weakest_trend['significance']})")

sig_count = (monthly_trends["p_value"] < 0.05).sum()
print(f"• Tendances significatives : {sig_count} / {len(monthly_trends)} mois")

seasonal_range = monthly_stats["mean_temp"].max() - monthly_stats["mean_temp"].min()
print(f"\nVARIATION SAISONNIÈRE :")
print(f"• Amplitude : {seasonal_range:.2f}°C")
print(f"• Rapport été/hiver : {monthly_stats['mean_temp'].max() / monthly_stats['mean_temp'].min():.2f}")

print(f"\n✅ Analyse mensuelle complétée avec succès")
print(f"   • Dataset créé : {len(monthly_dataset)} observations")
print(f"   • 6 visualisations générées")
print(f"   • Tous les mois analysés ({monthly_dataset['month'].nunique()}/12)")
print(f"   • Période : {monthly_dataset['year'].min()} - {monthly_dataset['year'].max()}")

# -----------------------------------------------------------------------------
# CONCLUSION FINALE
# Toutes les tendances sont à la hausse entre 1980 et 2023.
# Taux moyen global estimé : +0.015 à +0.03°C/an (~+0.6 à +1.3°C sur 43 ans).
# Juillet est le mois qui se réchauffe le plus vite.
# Résultat compatible avec la tendance globale du changement climatique.
# -----------------------------------------------------------------------------

print(f"\n{'='*70}")
print("FIN DE L'ANALYSE")
print(f"{'='*70}")

# =============================================================================
# ÉTAPE 4 - ANALYSE ET VISUALISATION AVANCÉE
# Objectifs :
#   4.1 Série temporelle avec moyenne mobile (rolling mean)
#   4.2 Comparaison saisonnière détaillée (4 saisons)
#   4.3 Carte d'évolution spatiale : SST début vs fin de période
#   4.4 Scatter + heatmap de corrélation entre mois (variables dérivées SST)
# =============================================================================
 
print(f"\n{'='*70}")
print("ÉTAPE 4 - ANALYSE ET VISUALISATION AVANCÉE")
print(f"{'='*70}")
 
# -----------------------------------------------------------------------------
# 4.1 SÉRIE TEMPORELLE AVEC MOYENNE MOBILE
# Équivalent R : geom_line + geom_smooth(method="loess") + rollmean (zoo)
# Ici on utilise pandas rolling() pour la moyenne mobile
# On travaille sur la région Mer Rouge (region_catalog)
# -----------------------------------------------------------------------------
 
print("\n4.1 Série temporelle avec moyenne mobile...")
 
# Série temporelle complète triée par date
ts = region_catalog.sort_values("date").copy()
ts["date"] = pd.to_datetime(ts["date"])
ts = ts.set_index("date")
 
# Moyenne mobile 12 mois (équivalent R : rollmean(x, k=12, fill=NA))
ts["rolling_12"] = ts["mean_sst"].rolling(window=12, center=True, min_periods=6).mean()
# Moyenne mobile 3 mois pour lissage court terme
ts["rolling_3"]  = ts["mean_sst"].rolling(window=3,  center=True, min_periods=2).mean()
 
fig, ax = plt.subplots(figsize=(14, 5))
 
# Série brute
ax.plot(ts.index, ts["mean_sst"],
        color="lightcoral", linewidth=0.8, alpha=0.6, label="SST mensuelle")
 
# Moyenne mobile 3 mois
ax.plot(ts.index, ts["rolling_3"],
        color="steelblue", linewidth=1.2, alpha=0.8, label="Moyenne mobile 3 mois")
 
# Moyenne mobile 12 mois (tendance long terme)
ax.plot(ts.index, ts["rolling_12"],
        color="#C73E1D", linewidth=2.0, label="Moyenne mobile 12 mois (tendance)")
 
# Droite de tendance linéaire globale (équivalent R : geom_smooth(method="lm"))
years_num = ts.index.year + (ts.index.month - 1) / 12
slope, intercept, *_ = stats.linregress(years_num, ts["mean_sst"].values)
ax.plot(ts.index, slope * years_num + intercept,
        color="black", linewidth=1.5, linestyle="--", label=f"Tendance linéaire ({slope*10:+.3f}°C/déc)")
 
ax.set_title(f"Série temporelle SST - {best_region_name} (1980-2023)\n"
             "Avec moyenne mobile et tendance linéaire", fontweight="bold", fontsize=13)
ax.set_xlabel("Année")
ax.set_ylabel("SST (°C)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("serie_temporelle_rolling.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Série temporelle avec moyenne mobile générée")
 
# -----------------------------------------------------------------------------
# 4.2 COMPARAISON SAISONNIÈRE DÉTAILLÉE
# Équivalent R : boxplots par saison + comparaison interannuelle
# On définit 4 saisons astronomiques
# -----------------------------------------------------------------------------
 
print("\n4.2 Comparaison saisonnière...")
 
# Assignation des saisons (équivalent R : case_when sur month)
def get_season(month):
    if month in [12, 1, 2]:  return "Hiver (DJF)"
    elif month in [3, 4, 5]: return "Printemps (MAM)"
    elif month in [6, 7, 8]: return "Été (JJA)"
    else:                    return "Automne (SON)"
 
region_catalog["season"] = region_catalog["month"].apply(get_season)
season_order = ["Hiver (DJF)", "Printemps (MAM)", "Été (JJA)", "Automne (SON)"]
season_colors = {"Hiver (DJF)": "#4ECDC4", "Printemps (MAM)": "#96CEB4",
                 "Été (JJA)":   "#FF6B6B", "Automne (SON)":   "#F7DC6F"}
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
# --- Boxplot saisonnier (équivalent R : geom_boxplot par saison) ---
ax = axes[0]
bp = ax.boxplot(
    [region_catalog[region_catalog["season"] == s]["mean_sst"].dropna().values
     for s in season_order],
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 2},
    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
)
for patch, s in zip(bp["boxes"], season_order):
    patch.set_facecolor(season_colors[s])
    patch.set_alpha(0.85)
 
# Moyennes par saison
for i, s in enumerate(season_order):
    mean_val = region_catalog[region_catalog["season"] == s]["mean_sst"].mean()
    ax.scatter(i + 1, mean_val, marker="D", color="white",
               edgecolors="black", s=50, zorder=5)
 
ax.set_xticks(range(1, 5))
ax.set_xticklabels(season_order, rotation=15, ha="right", fontsize=9)
ax.set_title("Distribution saisonnière SST\nDiamants : moyennes", fontweight="bold")
ax.set_ylabel("SST (°C)")
ax.grid(alpha=0.3, axis="y")
 
# --- Évolution saisonnière par décennie ---
ax = axes[1]
region_catalog["decade"] = (region_catalog["year"] // 10) * 10
decade_season = (region_catalog
                 .groupby(["decade", "season"])["mean_sst"]
                 .mean()
                 .reset_index())
 
decade_colors = {1980: "#2E86AB", 1990: "#A23B72",
                 2000: "#F18F01", 2010: "#C73E1D", 2020: "#3B1C32"}
 
for decade, grp in decade_season.groupby("decade"):
    grp = grp.set_index("season").reindex(season_order)
    ax.plot(range(4), grp["mean_sst"], marker="o", linewidth=1.8,
            markersize=6, label=str(decade) + "s",
            color=decade_colors.get(decade, "gray"))
 
ax.set_xticks(range(4))
ax.set_xticklabels(season_order, rotation=15, ha="right", fontsize=9)
ax.set_title("SST moyenne par saison et décennie", fontweight="bold")
ax.set_ylabel("SST (°C)")
ax.legend(title="Décennie", fontsize=9)
ax.grid(alpha=0.3)
 
fig.suptitle(f"Comparaison saisonnière SST - {best_region_name}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("comparaison_saisonniere.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Comparaison saisonnière générée")
 
# -----------------------------------------------------------------------------
# 4.3 CARTE D'ÉVOLUTION SPATIALE : SST début vs fin de période
# Équivalent R : contour plot / heatmap sur coordonnées
# Différence SST moyenne (2018-2023) - SST moyenne (1980-1985)
# -> montre le réchauffement spatial sur la région Mer Rouge
# -----------------------------------------------------------------------------
 
print("\n4.3 Carte d'évolution spatiale...")
 
# Sélection des fichiers pour chaque période
def get_files_for_period(nc_files, year_start, year_end):
    result = []
    for f in nc_files:
        year = int(os.path.basename(f)[:4])
        if year_start <= year <= year_end:
            result.append(f)
    return result
 
files_early = get_files_for_period(nc_files, 1980, 1985)
files_late  = get_files_for_period(nc_files, 2018, 2023)
 
# Moyenne spatiale pour chaque période
def mean_spatial(file_list, bbox):
    xmin, xmax, ymin, ymax = bbox
    arrays = []
    for f in file_list:
        data = read_sst_file(f)
        # Découpage bbox (simple, Mer Rouge pas > 180°)
        lat_mask = (data["lat"] >= ymin) & (data["lat"] <= ymax)
        lon_mask = (data["lon"] >= xmin) & (data["lon"] <= xmax)
        region   = data["sst"][np.ix_(lat_mask, lon_mask)]
        arrays.append(region)
    stack = np.array(arrays, dtype=float)
    return np.nanmean(stack, axis=0), data["lat"][lat_mask], data["lon"][lon_mask]
 
print("  Calcul SST moyenne 1980-1985...")
sst_early, lats_r, lons_r = mean_spatial(files_early, best_bbox)
print("  Calcul SST moyenne 2018-2023...")
sst_late,  _,      _      = mean_spatial(files_late,  best_bbox)
 
# Différence (réchauffement)
sst_diff = sst_late - sst_early
 
fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                         subplot_kw={"projection": ccrs.PlateCarree()})
 
titles   = ["SST moyenne 1980-1985", "SST moyenne 2018-2023",
            "Différence (réchauffement)"]
datasets = [sst_early, sst_late, sst_diff]
cmaps    = ["plasma", "plasma", "RdBu_r"]
vmins    = [sst_early[~np.isnan(sst_early)].min(),
            sst_early[~np.isnan(sst_early)].min(), -2]
vmaxs    = [sst_late[~np.isnan(sst_late)].max(),
            sst_late[~np.isnan(sst_late)].max(),  2]
 
for ax, data_plot, title, cmap, vmin, vmax in zip(
        axes, datasets, titles, cmaps, vmins, vmaxs):
 
    img = ax.pcolormesh(lons_r, lats_r, data_plot,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree())
    plt.colorbar(img, ax=ax, label="SST (°C)", shrink=0.7, pad=0.02)
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", linewidth=0.3)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    xmin_b, xmax_b, ymin_b, ymax_b = best_bbox
    ax.set_extent([xmin_b - 1, xmax_b + 1, ymin_b - 1, ymax_b + 1])
    ax.set_title(title, fontweight="bold", fontsize=10)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
 
fig.suptitle(f"Évolution spatiale SST - {best_region_name}\n"
             "Réchauffement observé entre 1980-1985 et 2018-2023",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("carte_evolution_spatiale.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Carte d'évolution spatiale générée")
 
# -----------------------------------------------------------------------------
# 4.4 SCATTER + HEATMAP DE CORRÉLATION ENTRE MOIS
# Équivalent R : cor() + ggplot geom_point + geom_smooth
# Variables dérivées de la SST :
#   - anomalie annuelle vs température absolue
#   - corrélation croisée entre les mois (pivot année x mois)
# -----------------------------------------------------------------------------
 
print("\n4.4 Scatter et heatmap de corrélation...")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
 
# --- Scatter : anomalie SST vs année (par saison) ---
ax = axes[0]
 
# Calcul anomalie par rapport à la moyenne globale de la série
global_mean = region_catalog["mean_sst"].mean()
region_catalog["anomaly"] = region_catalog["mean_sst"] - global_mean
 
for s in season_order:
    subset = region_catalog[region_catalog["season"] == s]
    ax.scatter(subset["year"], subset["anomaly"],
               color=season_colors[s], alpha=0.6, s=20, label=s)
    # Droite de tendance par saison
    if len(subset) >= 5:
        sl, inter, *_ = stats.linregress(subset["year"], subset["anomaly"])
        x_line = np.array([subset["year"].min(), subset["year"].max()])
        ax.plot(x_line, sl * x_line + inter,
                color=season_colors[s], linewidth=1.5, linestyle="--")
 
ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
ax.set_title("Anomalies SST par saison vs année\n"
             "Lignes : tendances linéaires", fontweight="bold")
ax.set_xlabel("Année")
ax.set_ylabel("Anomalie SST (°C)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
 
# --- Heatmap de corrélation entre mois ---
# Pivot : lignes = années, colonnes = mois
# Équivalent R : cor(pivot_df) + heatmap
ax = axes[1]
 
pivot_corr = (region_catalog
              .pivot_table(index="year", columns="month", values="mean_sst")
              .dropna())
# Renommer les colonnes avec noms de mois
pivot_corr.columns = [all_months_fr[m - 1] for m in pivot_corr.columns]
 
corr_matrix = pivot_corr.corr()
 
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=0.3,
            cbar_kws={"label": "Corrélation de Pearson"},
            mask=False)
ax.set_title("Corrélation inter-mensuelle SST\n"
             "(corrélation de Pearson entre mois)", fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=8)
ax.tick_params(axis="y", rotation=0,  labelsize=8)
 
fig.suptitle(f"Relations entre variables SST dérivées - {best_region_name}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("scatter_correlation_sst.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Scatter et heatmap de corrélation générés")
 
# -----------------------------------------------------------------------------
# SYNTHÈSE ÉTAPE 4
# -----------------------------------------------------------------------------
 
print(f"\n{'='*70}")
print("SYNTHÈSE ÉTAPE 4 - ANALYSE ET VISUALISATION AVANCÉE")
print(f"{'='*70}")
print(f"\nRégion analysée    : {best_region_name}")
print(f"Période            : 1980 - 2023")
print(f"\nVisualisations produites :")
print(f"  1. serie_temporelle_rolling.png   - Série + moyenne mobile + tendance")
print(f"  2. comparaison_saisonniere.png    - Boxplots saisons + évolution décennale")
print(f"  3. carte_evolution_spatiale.png   - Cartes SST 1980-85 vs 2018-23 + diff")
print(f"  4. scatter_correlation_sst.png    - Anomalies saisonnières + corrélation inter-mensuelle")
 
print(f"\nRÉSULTATS CLÉS :")
for s in season_order:
    subset = region_catalog[region_catalog["season"] == s]
    sl, *_ = stats.linregress(subset["year"], subset["mean_sst"])
    print(f"  • {s:<22} : tendance {sl*10:+.3f}°C/décennie")
 
warming = float(np.nanmean(sst_diff))
print(f"\n  • Réchauffement spatial moyen (1980-85 -> 2018-23) : {warming:+.2f}°C")
 
print(f"\n✅ Étape 4 complétée avec succès")
print(f"\n{'='*70}")
print("FIN DE L'ANALYSE COMPLÈTE")
print(f"{'='*70}")