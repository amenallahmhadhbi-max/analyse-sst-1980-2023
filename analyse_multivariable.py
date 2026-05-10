# =============================================================================
# Analyse multi-variables : SST + CHL + SLA — Mer Rouge 1993-2023
# Question scientifique étendue :
#   Comment la SST, la chlorophylle-a et le niveau de la mer
#   évoluent-ils ensemble dans la Mer Rouge depuis 1993 ?
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
from scipy import stats
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

DATA_SST = "C:/Users/ammon/Desktop/projet_climat/data"
DATA_CHL = "C:/Users/ammon/Desktop/projet_climat/data/data_chloro"
DATA_SLA = "C:/Users/ammon/Desktop/projet_climat/data/data_sea_level"

# Région Mer Rouge
BBOX = (32, 44, 12, 30)   # xmin, xmax, ymin, ymax

TARGET_MONTHS  = [1, 5, 7, 10]
MONTHS_FR      = {1: "Janvier", 5: "Mai", 7: "Juillet", 10: "Octobre"}

# Couleurs par variable
COLORS = {"SST": "#EF553B", "CHL": "#00CC96", "SLA": "#636EFA"}

# -----------------------------------------------------------------------------
# 2. FONCTIONS COMMUNES DE LECTURE ET EXTRACTION
# Chaque variable a son propre nom de champ NetCDF
# -----------------------------------------------------------------------------

VARIABLE_FIELDS = {
    "SST": "analysed_sst",    # Kelvin -> °C
    "CHL": "chlor_a",         # mg/m³ (nom dans fichiers ESACCI-OC)
    "SLA": "sla",             # mètres (nom dans fichiers dt_global)
}

def read_netcdf(file_path, variable):
    """
    Lit un fichier NetCDF et retourne les données pour la variable donnée.
    Gère automatiquement les conversions (SST Kelvin -> Celsius).
    """
    ds    = xr.open_dataset(file_path)
    field = VARIABLE_FIELDS[variable]

    # Essai du nom exact, sinon cherche un nom approchant
    if field not in ds:
        candidates = [v for v in ds.data_vars if variable.lower() in v.lower()]
        if candidates:
            field = candidates[0]
        else:
            ds.close()
            return None

    data = ds[field].values
    if data.ndim == 3:
        data = data[0]

    # Conversion SST Kelvin -> Celsius
    if variable == "SST":
        data = data - 273.15

    # Récupération des coordonnées (lat/lon)
    lat_key = "lat" if "lat" in ds else "latitude"
    lon_key = "lon" if "lon" in ds else "longitude"
    lats = ds[lat_key].values
    lons = ds[lon_key].values

    # Extraction de la date depuis le nom de fichier (format YYYYMMDD ou YYYYMM)
    fname = os.path.basename(file_path)
    date = None
    # Format CHL : ...OCx-19980110-fv6.0.nc -> chercher avant "-fv"
    if "-fv" in fname:
        try:
            part = fname.split("-fv")[0]  # ...OCx-19980110
            date_str = part[-8:]           # 19980110
            date = datetime.strptime(date_str, "%Y%m%d")
        except Exception:
            pass
    # Format SLA : dt_global_..._199301_vDT2024-M01.nc -> YYYYMM avant "_vDT"
    if date is None and "_vDT" in fname:
        try:
            part = fname.split("_vDT")[0]  # ..._199301
            date_str = part[-6:]            # 199301
            date = datetime.strptime(date_str, "%Y%m")
        except Exception:
            pass
    # Format SST : 19800110120000-ESACCI...
    if date is None:
        try:
            date = datetime.strptime(fname[:8], "%Y%m%d")
        except Exception:
            pass
    if date is None:
        try:
            date = datetime.strptime(fname[:6], "%Y%m")
        except Exception:
            pass

    ds.close()
    return {"data": data, "lat": lats, "lon": lons, "date": date}


def extract_region_mean(data_array, lats, lons, bbox):
    """Extrait la moyenne spatiale sur une région via masque 2D."""
    xmin, xmax, ymin, ymax = bbox
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = ((lat_grid >= ymin) & (lat_grid <= ymax) &
            (lon_grid >= xmin) & (lon_grid <= xmax))
    cells = data_array[mask]
    valid = cells[~np.isnan(cells)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def build_catalog(data_path, variable, bbox):
    """
    Construit un catalogue pandas DataFrame pour une variable donnée.
    Équivalent de la boucle de lecture dans analyse_sst.py.
    """
    nc_files = sorted(glob.glob(os.path.join(data_path, "*.nc")))
    if not nc_files:
        print(f"  ⚠️  Aucun fichier dans {data_path}")
        return pd.DataFrame()

    records = []
    for f in tqdm(nc_files, desc=f"Lecture {variable}"):
        result = read_netcdf(f, variable)
        if result is None or result["date"] is None:
            continue

        mean_val = extract_region_mean(
            result["data"], result["lat"], result["lon"], bbox
        )
        records.append({
            "date":  result["date"],
            "year":  result["date"].year,
            "month": result["date"].month,
            variable: mean_val,
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    print(f"  ✅ {variable} : {len(df)} observations")
    return df

# -----------------------------------------------------------------------------
# 3. CHARGEMENT DES TROIS VARIABLES
# Cache CSV par variable pour éviter de relire les NetCDF
# -----------------------------------------------------------------------------

print("=== CHARGEMENT DES DONNÉES ===\n")

def load_with_cache(data_path, variable, bbox):
    cache = f"cache_{variable.lower()}.csv"
    if os.path.exists(cache):
        print(f"  Cache trouvé pour {variable}")
        df = pd.read_csv(cache, parse_dates=["date"])
        df["year"]  = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        return df
    df = build_catalog(data_path, variable, bbox)
    if not df.empty:
        df.to_csv(cache, index=False)
        print(f"  Cache sauvegardé : {cache}")
    return df

df_sst = load_with_cache(DATA_SST, "SST", BBOX)
df_chl = load_with_cache(DATA_CHL, "CHL", BBOX)
df_sla = load_with_cache(DATA_SLA, "SLA", BBOX)

# Fusion sur l'intersection des dates communes (year + month)
# On garde uniquement les mois cibles
df_sst_f = df_sst[df_sst["month"].isin(TARGET_MONTHS)][["year","month","SST"]]
df_chl_f = df_chl[df_chl["month"].isin(TARGET_MONTHS)][["year","month","CHL"]]
df_sla_f = df_sla[df_sla["month"].isin(TARGET_MONTHS)][["year","month","SLA"]]

# Merge : inner join sur year + month (ne garde que les dates communes)
df = (df_sst_f
      .merge(df_chl_f, on=["year","month"], how="inner")
      .merge(df_sla_f, on=["year","month"], how="inner"))

df["month_fr"] = df["month"].map(MONTHS_FR)
df["date"]     = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
)

print(f"\n✅ Dataset combiné : {len(df)} observations")
print(f"   Période : {df['year'].min()} – {df['year'].max()}")
print(f"   Colonnes : {list(df.columns)}\n")

# Calcul des anomalies pour chaque variable
for var in ["SST", "CHL", "SLA"]:
    df[f"anom_{var}"] = df.groupby("month")[var].transform(
        lambda x: x - x.mean()
    )

# -----------------------------------------------------------------------------
# 4. STATISTIQUES DESCRIPTIVES DES TROIS VARIABLES
# Équivalent R : summary() + kable()
# -----------------------------------------------------------------------------

print("=== STATISTIQUES DESCRIPTIVES ===\n")

for var in ["SST", "CHL", "SLA"]:
    unit   = {"SST": "°C", "CHL": "mg/m³", "SLA": "m"}[var]
    sub_df = df[["year", var]].dropna()
    subset = sub_df[var]
    print(f"{var} ({unit}) :")
    print(f"  Moyenne : {subset.mean():.3f}")
    print(f"  Écart-type : {subset.std():.3f}")
    print(f"  Min : {subset.min():.3f}  |  Max : {subset.max():.3f}")
    if len(sub_df) >= 5:
        slope, _, r, p, _ = stats.linregress(sub_df["year"], subset)
        print(f"  Tendance : {slope*10:+.4f} {unit}/décennie  (p={p:.4f})\n")
    else:
        print(f"  Tendance : données insuffisantes\n")

# -----------------------------------------------------------------------------
# 5. VISUALISATION 1 — Séries temporelles des 3 variables
# Équivalent R : facet_wrap avec geom_line + geom_smooth
# -----------------------------------------------------------------------------

print("=== VIZ 1 : Séries temporelles ===")

fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
units = {"SST": "°C", "CHL": "mg/m³", "SLA": "m"}

for ax, var in zip(axes, ["SST", "CHL", "SLA"]):
    color = COLORS[var]

    # Série brute
    ax.plot(df["date"], df[var], color=color, linewidth=0.8,
            alpha=0.5, marker="o", markersize=2)

    # Moyenne mobile 12 mois
    rolling = df[var].rolling(12, center=True, min_periods=4).mean()
    ax.plot(df["date"], rolling, color=color, linewidth=2,
            label="Moyenne mobile 12 mois")

    # Tendance linéaire
    years_num = df["year"] + (df["month"] - 1) / 12
    slope, intercept, *_ = stats.linregress(years_num, df[var].ffill())
    ax.plot(df["date"],
            slope * years_num + intercept,
            color="black", linewidth=1.2, linestyle="--",
            label=f"Tendance ({slope*10:+.3f} {units[var]}/déc)")

    ax.set_ylabel(f"{var} ({units[var]})", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title(var, fontweight="bold", color=color, fontsize=12)

axes[-1].set_xlabel("Année")
fig.suptitle("Séries temporelles SST, CHL et SLA — Mer Rouge",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("series_temporelles_3var.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 1 sauvegardée\n")

# -----------------------------------------------------------------------------
# 6. VISUALISATION 2 — Scatter plots : SST vs CHL et SST vs SLA
# Équivalent R : geom_point + geom_smooth(method="lm") + stat_cor
# -----------------------------------------------------------------------------

print("=== VIZ 2 : Scatter SST vs CHL et SST vs SLA ===")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (var_y, var_x) in zip(axes, [("CHL", "SST"), ("SLA", "SST")]):
    subset = df[[var_x, var_y]].dropna()
    color  = COLORS[var_y]

    # Nuage de points coloré par mois
    for month_num in TARGET_MONTHS:
        m_df = df[df["month"] == month_num][[var_x, var_y]].dropna()
        ax.scatter(m_df[var_x], m_df[var_y],
                   label=MONTHS_FR[month_num], alpha=0.6, s=25)

    # Droite de régression globale
    slope, intercept, r, p, _ = stats.linregress(subset[var_x], subset[var_y])
    x_line = np.linspace(subset[var_x].min(), subset[var_x].max(), 100)
    ax.plot(x_line, slope * x_line + intercept,
            color="black", linewidth=1.5, linestyle="--")

    unit_x = {"SST": "°C", "CHL": "mg/m³", "SLA": "m"}[var_x]
    unit_y = {"SST": "°C", "CHL": "mg/m³", "SLA": "m"}[var_y]

    ax.set_xlabel(f"{var_x} ({unit_x})", fontsize=11)
    ax.set_ylabel(f"{var_y} ({unit_y})", fontsize=11)
    ax.set_title(
        f"{var_x} vs {var_y}\nr = {r:.3f}  |  p = {p:.4f}",
        fontweight="bold", fontsize=12
    )
    ax.legend(title="Mois", fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Relations entre SST, CHL et SLA — Mer Rouge",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("scatter_3variables.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 2 sauvegardée\n")

# -----------------------------------------------------------------------------
# 7. VISUALISATION 3 — Heatmap de corrélation entre les 3 variables
# Équivalent R : cor() + corrplot / geom_tile
# -----------------------------------------------------------------------------

print("=== VIZ 3 : Heatmap de corrélation ===")

corr_data = df[["SST", "CHL", "SLA"]].dropna()
corr_matrix = corr_data.corr(method="pearson")

# P-values pour chaque paire
pval_matrix = pd.DataFrame(np.ones((3,3)),
                            index=corr_matrix.index,
                            columns=corr_matrix.columns)
for col1 in corr_matrix.columns:
    for col2 in corr_matrix.columns:
        if col1 != col2:
            _, p = stats.pearsonr(corr_data[col1], corr_data[col2])
            pval_matrix.loc[col1, col2] = p

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".3f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=1,
            cbar_kws={"label": "Corrélation de Pearson"})

# Annotations p-value
for i in range(3):
    for j in range(3):
        if i != j:
            p = pval_matrix.iloc[i, j]
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(j + 0.5, i + 0.75, star,
                    ha="center", fontsize=10, color="black")

ax.set_title("Corrélation inter-variables SST / CHL / SLA\n"
             "(*** p<0.001, ** p<0.01, * p<0.05, ns = non significatif)",
             fontweight="bold", fontsize=11)
plt.tight_layout()
plt.savefig("heatmap_correlation_3var.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 3 sauvegardée\n")

# -----------------------------------------------------------------------------
# 8. VISUALISATION 4 — Anomalies normalisées des 3 variables (même axe)
# Permet de comparer les dynamiques temporelles sur une même échelle
# -----------------------------------------------------------------------------

print("=== VIZ 4 : Anomalies normalisées ===")

fig, ax = plt.subplots(figsize=(13, 5))

for var in ["SST", "CHL", "SLA"]:
    anom_col = f"anom_{var}"
    std_val  = df[anom_col].std()
    norm_anom = df[anom_col] / std_val   # Normalisation par écart-type
    unit  = {"SST": "°C", "CHL": "mg/m³", "SLA": "m"}[var]

    ax.plot(df["date"], norm_anom,
            color=COLORS[var], linewidth=1.2, label=var, alpha=0.8)

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.fill_between(df["date"], 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                alpha=0.03, color="red")
ax.set_title("Anomalies normalisées — SST, CHL, SLA — Mer Rouge\n"
             "(anomalie / écart-type → comparaison sur même échelle)",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Année")
ax.set_ylabel("Anomalie normalisée (σ)")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("anomalies_normalisees_3var.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure 4 sauvegardée\n")

# -----------------------------------------------------------------------------
# 9. SYNTHÈSE FINALE
# -----------------------------------------------------------------------------

print("="*70)
print("SYNTHÈSE — ANALYSE MULTI-VARIABLES")
print("="*70)
print(f"\nPériode commune : {df['year'].min()} – {df['year'].max()}")
print(f"Observations   : {len(df)}")

print("\nTendances par variable :")
for var in ["SST", "CHL", "SLA"]:
    unit  = {"SST": "°C", "CHL": "mg/m³", "SLA": "m"}[var]
    slope, _, r, p, _ = stats.linregress(df["year"], df[var].ffill())
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {var:<5} : {slope*10:+.4f} {unit}/déc  |  R²={r**2:.3f}  |  {sig}")

print("\nCorrélations :")
for pair in [("SST","CHL"), ("SST","SLA"), ("CHL","SLA")]:
    sub = df[[pair[0], pair[1]]].dropna()
    r, p = stats.pearsonr(sub[pair[0]], sub[pair[1]])
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {pair[0]} ↔ {pair[1]} : r = {r:+.3f}  ({sig})")

print("\n✅ Analyse multi-variables complétée")
print("   Figures produites :")
print("   - series_temporelles_3var.png")
print("   - scatter_3variables.png")
print("   - heatmap_correlation_3var.png")
print("   - anomalies_normalisees_3var.png")
