# =============================================================================
# TABLEAU DE BORD INTERACTIF - Analyse SST 1980-2023
# Outil : Dash (Plotly)
# Question scientifique : La SST a-t-elle augmenté entre 1980 et 2023 ?
#
# Lancement :
#   pip install dash plotly pandas numpy xarray netCDF4 scipy
#   python dashboard.py
#   Ouvrir : http://127.0.0.1:8050
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from scipy import stats

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_PATH = "C:/Users/ammon/Desktop/projet_climat/data/"   
# Régions océaniques (bbox : xmin, xmax, ymin, ymax)
OCEAN_REGIONS = {
    "Mer Rouge":       {"bbox": ( 32,  44,  12, 30), "color": "#EF553B"},
    "Méditerranée":    {"bbox": ( -6,  36,  30, 46), "color": "#636EFA"},
    "Mer Noire":       {"bbox": ( 28,  42,  41, 47), "color": "#AB63FA"},
    "Atlantique Nord": {"bbox": (-80,  20,   0, 65), "color": "#00CC96"},
    "Atlantique Sud":  {"bbox": (-70,  20, -60,  0), "color": "#19D3F3"},
    "Océan Indien":    {"bbox": ( 20, 120, -60, 30), "color": "#FF6692"},
    "Pacifique Nord":  {"bbox": (100, 260,   0, 65), "color": "#B6E880"},
    "Pacifique Sud":   {"bbox": (120, 290, -60,  0), "color": "#FECB52"},
}

SAISONS = {
    1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps",
    5: "Printemps", 6: "Été", 7: "Été", 8: "Été",
    9: "Automne", 10: "Automne", 11: "Automne", 12: "Hiver"
}

MOIS_FR = ["Janvier","Février","Mars","Avril","Mai","Juin",
           "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]

# =============================================================================
# 2. FONCTIONS DE LECTURE ET EXTRACTION
# =============================================================================

def read_sst_file(file_path):
    """Lit un fichier NetCDF SST -> dict avec sst, lat, lon, date."""
    ds  = xr.open_dataset(file_path)
    sst = ds["analysed_sst"].values
    if sst.ndim == 3:
        sst = sst[0]
    sst_c  = sst - 273.15
    lats   = ds["lat"].values
    lons   = ds["lon"].values
    fname  = os.path.basename(file_path)
    date   = datetime.strptime(fname[:8], "%Y%m%d")
    ds.close()
    return {"sst": sst_c, "lat": lats, "lon": lons,
            "date": date, "year": date.year, "month": date.month}


def extract_region_mean(sst_array, lats, lons, bbox):
    """Extrait la SST moyenne d'une région via masque 2D."""
    xmin, xmax, ymin, ymax = bbox
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lat_m = (lat_grid >= ymin) & (lat_grid <= ymax)
    if xmax > 180:
        xmax2  = xmax - 360
        lon_m  = ((lon_grid >= xmin) & (lon_grid <= 180)) | \
                 ((lon_grid >= -180) & (lon_grid <= xmax2))
    else:
        lon_m  = (lon_grid >= xmin) & (lon_grid <= xmax)
    cells = sst_array[lat_m & lon_m]
    valid = cells[~np.isnan(cells)]
    return float(np.mean(valid)) if len(valid) > 0 else np.nan


def extract_spatial_mean(file_list, bbox):
    """Moyenne spatiale 2D sur une liste de fichiers pour une bbox."""
    xmin, xmax, ymin, ymax = bbox
    arrays = []
    lats_r = lons_r = None
    for f in file_list:
        d = read_sst_file(f)
        lat_m = (d["lat"] >= ymin) & (d["lat"] <= ymax)
        lon_m = (d["lon"] >= xmin) & (d["lon"] <= xmax)
        region = d["sst"][np.ix_(lat_m, lon_m)]
        arrays.append(region)
        if lats_r is None:
            lats_r = d["lat"][lat_m]
            lons_r = d["lon"][lon_m]
    stack = np.array(arrays, dtype=float)
    return np.nanmean(stack, axis=0), lats_r, lons_r


# =============================================================================
# 3. CHARGEMENT DES DONNÉES (au démarrage)
# =============================================================================

print("Chargement des données...")

nc_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.nc")))
if not nc_files:
    raise FileNotFoundError(f"Aucun fichier NetCDF dans : {DATA_PATH}")

print(f"  {len(nc_files)} fichiers trouvés")

# ---- Système de cache CSV ----
# Au 1er lancement : lit tous les NetCDF et sauvegarde catalogue.csv
# Aux lancements suivants : relit directement le CSV (< 1 seconde)
CACHE_FILE = os.path.join(DATA_PATH, "catalogue_sst.csv")

if os.path.exists(CACHE_FILE):
    print(f"  Cache trouvé -> chargement depuis {CACHE_FILE}")
    df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    # Vérification de l'intégrité du cache
    required_cols = {"date", "year", "month", "region", "sst"}
    if not required_cols.issubset(df.columns) or df["sst"].isna().all():
        print("  ⚠️  Cache invalide -> reconstruction...")
        os.remove(CACHE_FILE)
        df = None
    else:
        print(f"  ✅ {len(df)} observations chargées en mémoire")
else:
    df = None

if df is None:
    print("  Première exécution : construction du catalogue (peut prendre 2-5 min)...")
    print("  Ce traitement ne sera fait qu'une seule fois.")
    records = []
    total = len(nc_files) * len(OCEAN_REGIONS)
    done  = 0
    for f in nc_files:
        d = read_sst_file(f)
        for region_name, info in OCEAN_REGIONS.items():
            mean_sst = extract_region_mean(
                d["sst"], d["lat"], d["lon"], info["bbox"])
            records.append({
                "date":    d["date"],
                "year":    d["year"],
                "month":   d["month"],
                "mois_fr": MOIS_FR[d["month"] - 1],
                "saison":  SAISONS[d["month"]],
                "region":  region_name,
                "sst":     mean_sst,
            })
            done += 1
            if done % 100 == 0:
                pct = done / total * 100
                print(f"    Progression : {done}/{total} ({pct:.0f}%)")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(CACHE_FILE, index=False)
    print(f"  ✅ Catalogue sauvegardé dans {CACHE_FILE}")

# Recalcul de mois_fr et saison depuis month (évite les problèmes d'encodage CSV)
df["mois_fr"] = df["month"].apply(lambda m: MOIS_FR[int(m) - 1])
df["saison"]  = df["month"].apply(lambda m: SAISONS[int(m)])

df["anomalie"] = df.groupby(["region", "month"])["sst"].transform(
    lambda x: x - x.mean()
)

years_available = sorted(df["year"].unique())
print(f"  Données chargées : {len(df)} observations | "
      f"{years_available[0]}-{years_available[-1]}")

# =============================================================================
# 4. LAYOUT DU TABLEAU DE BORD
# =============================================================================

app = dash.Dash(__name__, title="SST Dashboard 1980-2023")

app.layout = html.Div([

    # ---- En-tête ----
    html.Div([
        html.H1("🌊 Température de Surface de la Mer (SST)",
                style={"margin": "0", "color": "white", "fontSize": "24px"}),
        html.P("Analyse globale et régionale 1980–2023",
               style={"margin": "4px 0 0 0", "color": "#cce", "fontSize": "14px"}),
    ], style={"background": "#1a1a2e", "padding": "18px 30px",
              "borderBottom": "3px solid #636EFA"}),

    # ---- Panneaux de contrôle ----
    html.Div([

        # Sélecteur de région
        html.Div([
            html.Label("Région océanique", style={"fontWeight": "bold",
                                                   "fontSize": "13px"}),
            dcc.Dropdown(
                id="dropdown-region",
                options=[{"label": r, "value": r} for r in OCEAN_REGIONS],
                value="Mer Rouge",
                clearable=False,
                style={"fontSize": "13px"},
            ),
        ], style={"flex": "1", "minWidth": "200px"}),

        # Slider années
        html.Div([
            html.Label("Période d'analyse", style={"fontWeight": "bold",
                                                    "fontSize": "13px"}),
            dcc.RangeSlider(
                id="slider-years",
                min=years_available[0],
                max=years_available[-1],
                step=1,
                value=[int(years_available[0]), int(years_available[-1])],
                marks={y: str(y) for y in range(
                    years_available[0], years_available[-1] + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"flex": "3", "minWidth": "300px"}),

        # Sélecteur de variable affichée
        html.Div([
            html.Label("Variable", style={"fontWeight": "bold",
                                           "fontSize": "13px"}),
            dcc.RadioItems(
                id="radio-variable",
                options=[
                    {"label": " SST (°C)",       "value": "sst"},
                    {"label": " Anomalie (°C)",   "value": "anomalie"},
                ],
                value="sst",
                labelStyle={"display": "block", "fontSize": "13px",
                             "marginBottom": "4px"},
            ),
        ], style={"flex": "1", "minWidth": "150px"}),

    ], style={"display": "flex", "gap": "30px", "flexWrap": "wrap",
              "padding": "18px 30px", "background": "#f8f9fa",
              "borderBottom": "1px solid #ddd", "alignItems": "flex-end"}),

    # ---- Indicateurs texte ----
    html.Div(id="indicators",
             style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                    "padding": "14px 30px", "background": "#fff",
                    "borderBottom": "1px solid #eee"}),

    # ---- Graphiques principaux ----
    html.Div([

        # Ligne 1 : série temporelle + boxplots saisonniers
        html.Div([
            html.Div([dcc.Graph(id="graph-timeseries",
                                config={"displayModeBar": True})],
                     style={"flex": "2", "minWidth": "400px"}),
            html.Div([dcc.Graph(id="graph-seasonal",
                                config={"displayModeBar": True})],
                     style={"flex": "1", "minWidth": "280px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                  "marginBottom": "12px"}),

        # Ligne 2 : heatmap année×mois + carte spatiale
        html.Div([
            html.Div([dcc.Graph(id="graph-heatmap",
                                config={"displayModeBar": True})],
                     style={"flex": "1", "minWidth": "350px"}),
            html.Div([dcc.Graph(id="graph-map",
                                config={"displayModeBar": True})],
                     style={"flex": "1", "minWidth": "350px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

    ], style={"padding": "16px 30px", "background": "#fff"}),

    # ---- Pied de page ----
    html.Div([
        html.P("Source : ESA SST CCI – GHRSST L4 OSTIA | "
               "Auteur : Amenallah Mhadhbi | 2026",
               style={"margin": 0, "color": "#888", "fontSize": "12px"}),
    ], style={"padding": "12px 30px", "background": "#f8f9fa",
              "borderTop": "1px solid #ddd", "textAlign": "center"}),

], style={"fontFamily": "Segoe UI, Arial, sans-serif",
           "background": "#fafafa", "minHeight": "100vh"})


# =============================================================================
# 5. CALLBACKS
# =============================================================================

@app.callback(
    Output("indicators",      "children"),
    Output("graph-timeseries","figure"),
    Output("graph-seasonal",  "figure"),
    Output("graph-heatmap",   "figure"),
    Output("graph-map",       "figure"),
    Input("dropdown-region",  "value"),
    Input("slider-years",     "value"),
    Input("radio-variable",   "value"),
)
def update_all(region, year_range, variable):
    """
    Callback principal : met à jour tous les graphiques et indicateurs
    en fonction de la région, de la période et de la variable choisies.
    """
    import traceback
    try:
        return _update_all_inner(region, year_range, variable)
    except Exception as e:
        print("\n=== ERREUR CALLBACK ===")
        traceback.print_exc()
        print("========================\n")
        empty = go.Figure()
        empty.update_layout(
            title="Erreur - voir terminal",
            annotations=[{"text": str(e), "showarrow": False,
                          "xref": "paper", "yref": "paper",
                          "x": 0.5, "y": 0.5, "font": {"color": "red"}}]
        )
        return [html.P(f"Erreur : {e}", style={"color":"red"})],                empty, empty, empty, empty


def _update_all_inner(region, year_range, variable):
    # Sécurisation : year_range doit toujours être une liste de 2 entiers
    if not isinstance(year_range, (list, tuple)) or len(year_range) < 2:
        year_range = [int(years_available[0]), int(years_available[-1])]
    y0, y1 = int(year_range[0]), int(year_range[1])
    if y0 == y1:  # slider collapsed -> prendre toute la période
        y0, y1 = int(years_available[0]), int(years_available[-1])
    color    = OCEAN_REGIONS[region]["color"]
    var_label = "SST (°C)" if variable == "sst" else "Anomalie (°C)"

    # Filtre sur la région et la période
    df["year"]  = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    dff = df[(df["region"] == region) &
             (df["year"]   >= y0)     &
             (df["year"]   <= y1)].copy()

    # ------------------------------------------------------------------
    # INDICATEURS TEXTE
    # ------------------------------------------------------------------
    if dff.empty:
        indicators = [html.P("Aucune donnée pour cette sélection.")]
    else:
        last_year  = dff["year"].max()
        sst_last   = dff[dff["year"] == last_year]["sst"].mean()
        sst_first  = dff[dff["year"] == y0]["sst"].mean()
        sst_mean   = dff["sst"].mean()
        sst_max    = dff["sst"].max()
        sst_min    = dff["sst"].min()

        # Tendance linéaire sur la période (°C/décennie)
        dff_sorted = dff.sort_values("date")
        if len(dff_sorted) >= 5:
            years_num = dff_sorted["year"] + (dff_sorted["month"] - 1) / 12
            slope, *_ = stats.linregress(years_num, dff_sorted["sst"])
            trend_val = slope * 10
        else:
            trend_val = np.nan

        def indicator_card(title, value, unit="", color_val="#1a1a2e"):
            return html.Div([
                html.P(title, style={"margin": "0 0 2px 0", "fontSize": "11px",
                                     "color": "#666"}),
                html.P(f"{value} {unit}", style={"margin": 0, "fontSize": "20px",
                                                  "fontWeight": "bold",
                                                  "color": color_val}),
            ], style={"background": "#f0f4ff", "borderRadius": "8px",
                      "padding": "10px 16px", "borderLeft": f"4px solid {color}"})

        indicators = [
            indicator_card("SST moyenne", f"{sst_mean:.2f}", "°C"),
            indicator_card("SST dernière année", f"{sst_last:.2f}", "°C"),
            indicator_card("SST maximale", f"{sst_max:.2f}", "°C"),
            indicator_card("SST minimale", f"{sst_min:.2f}", "°C"),
            indicator_card(
                "Tendance",
                f"{trend_val:+.3f}" if not np.isnan(trend_val) else "N/A",
                "°C/décennie",
                color_val="#C73E1D" if trend_val > 0 else "#2E86AB"
            ),
            indicator_card(
                f"Variation {y0}→{y1}",
                f"{sst_last - sst_first:+.2f}" if not np.isnan(sst_first) else "N/A",
                "°C"
            ),
        ]

    # ------------------------------------------------------------------
    # GRAPHIQUE 1 : SÉRIE TEMPORELLE + MOYENNE MOBILE + TENDANCE
    # ------------------------------------------------------------------
    fig_ts = go.Figure()

    if not dff.empty:
        dff_ts = dff.sort_values("date")

        # Série brute
        fig_ts.add_trace(go.Scatter(
            x=dff_ts["date"], y=dff_ts[variable],
            mode="lines", name=var_label,
            line={"color": color, "width": 1},
            opacity=0.5,
        ))

        # Moyenne mobile 12 mois
        rolling = dff_ts[variable].rolling(12, center=True, min_periods=6).mean()
        fig_ts.add_trace(go.Scatter(
            x=dff_ts["date"], y=rolling,
            mode="lines", name="Moy. mobile 12 mois",
            line={"color": color, "width": 2.5},
        ))

        # Tendance linéaire
        years_num = dff_ts["year"] + (dff_ts["month"] - 1) / 12
        if len(dff_ts) >= 5:
            sl, inter, *_ = stats.linregress(years_num, dff_ts[variable])
            fig_ts.add_trace(go.Scatter(
                x=[dff_ts["date"].iloc[0], dff_ts["date"].iloc[-1]],
                y=[sl * years_num.iloc[0] + inter,
                   sl * years_num.iloc[-1] + inter],
                mode="lines", name=f"Tendance ({sl*10:+.3f}°C/déc)",
                line={"color": "black", "width": 1.5, "dash": "dash"},
            ))

    fig_ts.update_layout(
        title={"text": f"Série temporelle SST — {region}",
               "font": {"size": 14, "color": "#1a1a2e"}},
        xaxis_title="Année",
        yaxis_title=var_label,
        legend={"orientation": "h", "y": -0.2, "font": {"size": 10}},
        hovermode="x unified",
        template="plotly_white",
        margin={"t": 50, "b": 50, "l": 50, "r": 20},
    )

    # ------------------------------------------------------------------
    # GRAPHIQUE 2 : BOXPLOTS SAISONNIERS
    # ------------------------------------------------------------------
    fig_seas = go.Figure()
    saison_order  = ["Hiver", "Printemps", "Été", "Automne"]
    saison_colors = {"Hiver": "#4ECDC4", "Printemps": "#96CEB4",
                     "Été": "#FF6B6B", "Automne": "#F7DC6F"}

    for s in saison_order:
        vals = dff[dff["saison"] == s][variable].dropna()
        if len(vals) > 0:
            fig_seas.add_trace(go.Box(
                y=vals, name=s,
                marker_color=saison_colors[s],
                boxmean=True,   # affiche la moyenne
            ))

    fig_seas.update_layout(
        title={"text": "Distribution saisonnière",
               "font": {"size": 14, "color": "#1a1a2e"}},
        yaxis_title=var_label,
        showlegend=False,
        template="plotly_white",
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )

    # ------------------------------------------------------------------
    # GRAPHIQUE 3 : HEATMAP ANNÉE × MOIS
    # ------------------------------------------------------------------
    pivot = (dff.pivot_table(index="year", columns="mois_fr",
                              values=variable, aggfunc="mean")
               .reindex(columns=MOIS_FR))

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Plasma",
        colorbar={"title": {"text": var_label, "side": "right"}},
        hovertemplate="Année: %{y}<br>Mois: %{x}<br>"
                      + var_label + ": %{z:.2f}°C<extra></extra>",
    ))

    fig_hm.update_layout(
        title={"text": f"Heatmap {var_label} — année × mois",
               "font": {"size": 14, "color": "#1a1a2e"}},
        xaxis_title="Mois",
        yaxis_title="Année",
        template="plotly_white",
        margin={"t": 50, "b": 40, "l": 60, "r": 20},
    )

    # ------------------------------------------------------------------
    # GRAPHIQUE 4 : CARTE SPATIALE (différence fin vs début de période)
    # ------------------------------------------------------------------
    bbox = OCEAN_REGIONS[region]["bbox"]

    # Fichiers des 3 premières et 3 dernières années de la période
    files_start = [f for f in nc_files
                   if y0 <= int(os.path.basename(f)[:4]) <= y0 + 2]
    files_end   = [f for f in nc_files
                   if y1 - 2 <= int(os.path.basename(f)[:4]) <= y1]

    if files_start and files_end and bbox[1] <= 180:
        # Carte uniquement pour les régions sans correction Pacifique (simple)
        sst_start, lats_r, lons_r = extract_spatial_mean(files_start, bbox)
        sst_end,   _,      _      = extract_spatial_mean(files_end,   bbox)
        sst_diff = sst_end - sst_start

        fig_map = go.Figure(go.Heatmap(
            z=sst_diff,
            x=lons_r.tolist(),
            y=lats_r.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            colorbar={"title": {"text": "ΔT (°C)", "side": "right"}},
            hovertemplate="Lon: %{x:.2f}<br>Lat: %{y:.2f}<br>ΔT: %{z:.2f}°C<extra></extra>",
        ))
        map_title = (f"Réchauffement spatial {y0}-{y0+2} → {y1-2}-{y1}<br>"
                     f"<sup>Rouge = réchauffement | Bleu = refroidissement</sup>")
    else:
        # Fallback : carte SST moyenne de la période
        files_all = [f for f in nc_files
                     if y0 <= int(os.path.basename(f)[:4]) <= y1]
        if files_all and bbox[1] <= 180:
            sst_mean_sp, lats_r, lons_r = extract_spatial_mean(files_all, bbox)
            fig_map = go.Figure(go.Heatmap(
                z=sst_mean_sp,
                x=lons_r.tolist(),
                y=lats_r.tolist(),
                colorscale="Plasma",
                colorbar={"title": {"text": "SST (°C)", "side": "right"}},
            ))
            map_title = f"SST moyenne spatiale {y0}–{y1}"
        else:
            fig_map  = go.Figure()
            map_title = "Carte non disponible pour cette région"

    fig_map.update_layout(
        title={"text": map_title,
               "font": {"size": 13, "color": "#1a1a2e"}},
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        template="plotly_white",
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )

    return indicators, fig_ts, fig_seas, fig_hm, fig_map


# =============================================================================
# 6. LANCEMENT
# =============================================================================

if __name__ == "__main__":
    print("\n✅ Tableau de bord prêt !")
    print("   Ouvrir dans le navigateur : http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)