# =============================================================================
# TABLEAU DE BORD INTERACTIF — SST + CHL + SLA — Mer Rouge
# Lancement : python dashboard_multivariable.py
# Ouvrir    : http://127.0.0.1:8051
# =============================================================================

import os, glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from scipy import stats

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

PATHS = {
    "SST": {"dir": "C:/Users/ammon/Desktop/projet_climat/data",             "field": "analysed_sst",  "unit": "°C",    "color": "#EF553B", "convert": lambda x: x - 273.15},
    "CHL": {"dir": "C:/Users/ammon/Desktop/projet_climat/data/data_chloro",   "field": "chlor_a",       "unit": "mg/m³", "color": "#00CC96", "convert": lambda x: x},
    "SLA": {"dir": "C:/Users/ammon/Desktop/projet_climat/data/data_sea_level","field": "sla",           "unit": "m",     "color": "#636EFA", "convert": lambda x: x},
}

BBOX = (32, 44, 12, 30)  # Mer Rouge : xmin, xmax, ymin, ymax

MOIS_FR   = ["Janvier","Février","Mars","Avril","Mai","Juin",
             "Juillet","Août","Septembre","Octobre","Novembre","Décembre"]
SAISONS   = {1:"Hiver",2:"Hiver",3:"Printemps",4:"Printemps",
             5:"Printemps",6:"Été",7:"Été",8:"Été",
             9:"Automne",10:"Automne",11:"Automne",12:"Hiver"}

# =============================================================================
# 2. FONCTIONS
# =============================================================================

def get_date(fname):
    if "-fv" in fname:
        try: return datetime.strptime(fname.split("-fv")[0][-8:], "%Y%m%d")
        except: pass
    if "_vDT" in fname:
        try: return datetime.strptime(fname.split("_vDT")[0][-6:], "%Y%m")
        except: pass
    try: return datetime.strptime(fname[:8], "%Y%m%d")
    except: pass
    try: return datetime.strptime(fname[:6], "%Y%m")
    except: return None


def extract_mean(file_path, field, convert, bbox):
    xmin, xmax, ymin, ymax = bbox
    try:
        ds = xr.open_dataset(file_path)
        if field not in ds:
            candidates = [v for v in ds.data_vars if field.lower() in v.lower()]
            if not candidates:
                ds.close(); return np.nan
            field = candidates[0]
        data = ds[field].values
        if data.ndim == 3: data = data[0]
        if data.ndim == 4: data = data[0,0]
        data = convert(data.astype(float))
        lat_key = "lat" if "lat" in ds else "latitude"
        lon_key = "lon" if "lon" in ds else "longitude"
        lats = ds[lat_key].values
        lons = ds[lon_key].values
        ds.close()
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        mask = ((lat_grid >= ymin) & (lat_grid <= ymax) &
                (lon_grid >= xmin) & (lon_grid <= xmax))
        cells = data[mask]
        valid = cells[~np.isnan(cells)]
        return float(np.mean(valid)) if len(valid) > 0 else np.nan
    except:
        return np.nan


def build_catalog(var, cfg, bbox):
    cache = f"cache_{var.lower()}_mv.csv"
    if os.path.exists(cache):
        print(f"  Cache {var} trouvé")
        df = pd.read_csv(cache, parse_dates=["date"])
        df["year"]  = df["year"].astype(int)
        df["month"] = df["month"].astype(int)
        return df
    nc_files = sorted(glob.glob(os.path.join(cfg["dir"], "*.nc")))
    if not nc_files:
        print(f"  ⚠️  Aucun fichier {var}")
        return pd.DataFrame()
    records = []
    print(f"  Lecture {var} ({len(nc_files)} fichiers)...")
    for f in nc_files:
        d = get_date(os.path.basename(f))
        if not d: continue
        val = extract_mean(f, cfg["field"], cfg["convert"], bbox)
        records.append({"date": d, "year": d.year, "month": d.month, var: val})
    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df.to_csv(cache, index=False)
    print(f"  ✅ {var} : {len(df)} observations")
    return df

# =============================================================================
# 3. CHARGEMENT
# =============================================================================

print("Chargement des données...")

dfs = {}
for var, cfg in PATHS.items():
    dfs[var] = build_catalog(var, cfg, BBOX)

# Fusion
df = dfs["SST"][["date","year","month","SST"]].copy()
for var in ["CHL","SLA"]:
    if not dfs[var].empty:
        df = df.merge(dfs[var][["year","month",var]], on=["year","month"], how="left")
    else:
        df[var] = np.nan

df["date"]     = pd.to_datetime(df["date"])
df["month_fr"] = df["month"].apply(lambda m: MOIS_FR[int(m)-1])
df["saison"]   = df["month"].apply(lambda m: SAISONS[int(m)])

# Anomalies
for var in ["SST","CHL","SLA"]:
    df[f"anom_{var}"] = df.groupby("month")[var].transform(lambda x: x - x.mean())

years = sorted(df["year"].unique().tolist())
print(f"✅ Dataset : {len(df)} observations | {years[0]}–{years[-1]}")

# =============================================================================
# 4. LAYOUT
# =============================================================================

app = dash.Dash(__name__, title="SST · CHL · SLA — Mer Rouge")

VARS_OPTIONS = [
    {"label": "🌡 SST (°C)",         "value": "SST"},
    {"label": "🌿 CHL (mg/m³)",      "value": "CHL"},
    {"label": "🌊 SLA (m)",          "value": "SLA"},
    {"label": "📊 Anomalie SST",     "value": "anom_SST"},
    {"label": "📊 Anomalie CHL",     "value": "anom_CHL"},
    {"label": "📊 Anomalie SLA",     "value": "anom_SLA"},
]

app.layout = html.Div([

    # En-tête
    html.Div([
        html.H1("🌊 Tableau de bord — Mer Rouge",
                style={"margin":0,"color":"white","fontSize":"22px"}),
        html.P("SST · Chlorophylle-a · Niveau de la mer",
               style={"margin":"4px 0 0 0","color":"#aac","fontSize":"13px"}),
    ], style={"background":"#1a1a2e","padding":"16px 28px",
              "borderBottom":"3px solid #636EFA"}),

    # Contrôles
    html.Div([
        html.Div([
            html.Label("Variable principale", style={"fontWeight":"bold","fontSize":"12px"}),
            dcc.Dropdown(id="dd-var1", options=VARS_OPTIONS,
                         value="SST", clearable=False, style={"fontSize":"12px"}),
        ], style={"flex":"1","minWidth":"180px"}),

        html.Div([
            html.Label("Comparer avec", style={"fontWeight":"bold","fontSize":"12px"}),
            dcc.Dropdown(id="dd-var2",
                         options=[{"label":"Aucune","value":"none"}] + VARS_OPTIONS,
                         value="SLA", clearable=False, style={"fontSize":"12px"}),
        ], style={"flex":"1","minWidth":"180px"}),

        html.Div([
            html.Label("Période", style={"fontWeight":"bold","fontSize":"12px"}),
            dcc.RangeSlider(id="slider-years",
                            min=years[0], max=years[-1], step=1,
                            value=[years[0], years[-1]],
                            marks={y:str(y) for y in range(years[0],years[-1]+1,5)},
                            tooltip={"placement":"bottom"}),
        ], style={"flex":"3","minWidth":"300px"}),

        html.Div([
            html.Label("Affichage", style={"fontWeight":"bold","fontSize":"12px"}),
            dcc.RadioItems(id="radio-mode",
                           options=[{"label":" Valeur brute","value":"raw"},
                                    {"label":" Anomalie","value":"anom"}],
                           value="raw",
                           labelStyle={"display":"block","fontSize":"12px"}),
        ], style={"flex":"1","minWidth":"140px"}),

    ], style={"display":"flex","gap":"24px","flexWrap":"wrap",
              "padding":"16px 28px","background":"#f8f9fa",
              "borderBottom":"1px solid #ddd","alignItems":"flex-end"}),

    # Indicateurs
    html.Div(id="indicators",
             style={"display":"flex","gap":"12px","flexWrap":"wrap",
                    "padding":"12px 28px","background":"#fff",
                    "borderBottom":"1px solid #eee"}),

    # Graphiques
    html.Div([
        # Ligne 1 : série temporelle principale
        dcc.Graph(id="graph-ts", config={"displayModeBar":True},
                  style={"height":"320px"}),

        # Ligne 2 : 3 graphiques côte à côte
        html.Div([
            html.Div([dcc.Graph(id="graph-seasonal",
                                config={"displayModeBar":True},
                                style={"height":"280px"})],
                     style={"flex":"1","minWidth":"300px"}),
            html.Div([dcc.Graph(id="graph-scatter",
                                config={"displayModeBar":True},
                                style={"height":"280px"})],
                     style={"flex":"1","minWidth":"300px"}),
            html.Div([dcc.Graph(id="graph-heatmap",
                                config={"displayModeBar":True},
                                style={"height":"280px"})],
                     style={"flex":"1","minWidth":"300px"}),
        ], style={"display":"flex","gap":"8px","flexWrap":"wrap"}),

    ], style={"padding":"12px 28px","background":"#fff"}),

    # Pied
    html.Div([
        html.P("Source : ESA SST CCI | ESA OC CCI | CMEMS SLA | Auteur : Amenallah Mhadhbi | 2026",
               style={"margin":0,"color":"#888","fontSize":"11px"})
    ], style={"padding":"10px 28px","background":"#f8f9fa",
              "borderTop":"1px solid #ddd","textAlign":"center"}),

], style={"fontFamily":"Segoe UI, Arial, sans-serif","background":"#fafafa"})

# =============================================================================
# 5. CALLBACK
# =============================================================================

@app.callback(
    Output("indicators",    "children"),
    Output("graph-ts",      "figure"),
    Output("graph-seasonal","figure"),
    Output("graph-scatter", "figure"),
    Output("graph-heatmap", "figure"),
    Input("dd-var1",       "value"),
    Input("dd-var2",       "value"),
    Input("slider-years",  "value"),
    Input("radio-mode",    "value"),
)
def update(var1, var2, year_range, mode):
    import traceback
    try:
        return _update(var1, var2, year_range, mode)
    except Exception as e:
        traceback.print_exc()
        empty = go.Figure()
        empty.update_layout(title=f"Erreur : {e}", template="plotly_white")
        return [html.P(f"Erreur : {e}", style={"color":"red"})], empty, empty, empty, empty


def _update(var1, var2, year_range, mode):
    y0, y1 = int(year_range[0]), int(year_range[1])

    # Colonne réelle selon mode
    def col(v):
        return f"anom_{v}" if mode == "anom" and not v.startswith("anom_") else v

    c1 = col(var1)
    base_var1 = var1.replace("anom_","")
    cfg1 = PATHS.get(base_var1, PATHS["SST"])
    color1 = cfg1["color"]
    unit1  = ("°C anomalie" if mode=="anom" else cfg1["unit"]) if base_var1 in PATHS else ""

    dff = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()
    dff["year"]  = dff["year"].astype(int)
    dff["month"] = dff["month"].astype(int)

    # ── INDICATEURS ──────────────────────────────────────────────────────────
    def kpi(title, value, unit="", color="#1a1a2e"):
        return html.Div([
            html.P(title, style={"margin":"0 0 2px","fontSize":"10px","color":"#666"}),
            html.P(f"{value} {unit}", style={"margin":0,"fontSize":"18px",
                                              "fontWeight":"bold","color":color}),
        ], style={"background":"#f0f4ff","borderRadius":"8px",
                  "padding":"8px 14px","borderLeft":f"4px solid {color1}"})

    sub1 = dff[c1].dropna()
    indicators = []
    if len(sub1) > 0:
        indicators += [
            kpi("Moyenne",     f"{sub1.mean():.3f}", unit1),
            kpi("Maximum",     f"{sub1.max():.3f}",  unit1),
            kpi("Minimum",     f"{sub1.min():.3f}",  unit1),
        ]
        sub_yr = dff[["year", c1]].dropna()
        if len(sub_yr) >= 5:
            sl, *_ = stats.linregress(sub_yr["year"], sub_yr[c1])
            indicators.append(kpi("Tendance", f"{sl*10:+.4f}", f"{unit1}/déc",
                                  "#C73E1D" if sl > 0 else "#2E86AB"))
        last_val = dff[dff["year"]==y1][c1].mean()
        if not np.isnan(last_val):
            indicators.append(kpi(f"Dernière année ({y1})", f"{last_val:.3f}", unit1))

    # ── SÉRIE TEMPORELLE ─────────────────────────────────────────────────────
    dff_s = dff.sort_values("date")
    fig_ts = make_subplots(specs=[[{"secondary_y": var2 != "none"}]])

    # Variable 1
    fig_ts.add_trace(go.Scatter(
        x=dff_s["date"], y=dff_s[c1],
        mode="lines+markers", name=var1,
        line={"color":color1,"width":1}, marker={"size":3}, opacity=0.5
    ))
    roll1 = dff_s[c1].rolling(8, center=True, min_periods=3).mean()
    fig_ts.add_trace(go.Scatter(
        x=dff_s["date"], y=roll1,
        mode="lines", name=f"{var1} moy. mobile",
        line={"color":color1,"width":2.5}
    ))
    sub_yr = dff_s[["year",c1]].dropna()
    if len(sub_yr) >= 5:
        yr_num = sub_yr["year"] + (dff_s.loc[sub_yr.index,"month"]-1)/12
        sl, inter, *_ = stats.linregress(yr_num, sub_yr[c1])
        fig_ts.add_trace(go.Scatter(
            x=[dff_s["date"].iloc[0], dff_s["date"].iloc[-1]],
            y=[sl*yr_num.iloc[0]+inter, sl*yr_num.iloc[-1]+inter],
            mode="lines", name=f"Tendance {var1} ({sl*10:+.3f}/déc)",
            line={"color":"black","width":1.5,"dash":"dash"}
        ))

    # Variable 2 (axe secondaire)
    if var2 != "none":
        c2    = col(var2)
        bv2   = var2.replace("anom_","")
        cfg2  = PATHS.get(bv2, PATHS["SST"])
        color2 = cfg2["color"]
        fig_ts.add_trace(go.Scatter(
            x=dff_s["date"], y=dff_s[c2],
            mode="lines+markers", name=var2,
            line={"color":color2,"width":1,"dash":"dot"},
            marker={"size":3}, opacity=0.5
        ), secondary_y=True)
        roll2 = dff_s[c2].rolling(8, center=True, min_periods=3).mean()
        fig_ts.add_trace(go.Scatter(
            x=dff_s["date"], y=roll2,
            mode="lines", name=f"{var2} moy. mobile",
            line={"color":color2,"width":2}
        ), secondary_y=True)
        fig_ts.update_yaxes(title_text=f"{var2} ({cfg2['unit']})", secondary_y=True)

    fig_ts.update_layout(
        title=f"Série temporelle — {var1}" + (f" vs {var2}" if var2!="none" else ""),
        xaxis_title="Année",
        yaxis_title=f"{var1} ({unit1})",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation":"h","y":-0.2,"font":{"size":10}},
        margin={"t":50,"b":50,"l":60,"r":60},
    )

    # ── SAISONNIERS ───────────────────────────────────────────────────────────
    fig_seas = go.Figure()
    saison_order  = ["Hiver","Printemps","Été","Automne"]
    saison_colors = {"Hiver":"#4ECDC4","Printemps":"#96CEB4",
                     "Été":"#FF6B6B","Automne":"#F7DC6F"}
    for s in saison_order:
        vals = dff[dff["saison"]==s][c1].dropna()
        if len(vals) > 0:
            fig_seas.add_trace(go.Box(
                y=vals, name=s,
                marker_color=saison_colors[s], boxmean=True
            ))
    fig_seas.update_layout(
        title=f"Distribution saisonnière — {var1}",
        yaxis_title=f"{var1} ({unit1})",
        showlegend=False,
        template="plotly_white",
        margin={"t":50,"b":40,"l":50,"r":20},
    )

    # ── SCATTER var1 vs var2 ──────────────────────────────────────────────────
    fig_sc = go.Figure()
    if var2 != "none":
        c2   = col(var2)
        bv2  = var2.replace("anom_","")
        cfg2 = PATHS.get(bv2, PATHS["SST"])
        sub  = dff[[c1, c2, "month_fr"]].dropna()
        if len(sub) >= 5:
            for m in dff["month_fr"].unique():
                ms = sub[sub["month_fr"]==m]
                fig_sc.add_trace(go.Scatter(
                    x=ms[c1], y=ms[c2], mode="markers",
                    name=m, marker={"size":7,"opacity":0.7}
                ))
            sl, inter, r, p, _ = stats.linregress(sub[c1], sub[c2])
            x_line = np.linspace(sub[c1].min(), sub[c1].max(), 100)
            fig_sc.add_trace(go.Scatter(
                x=x_line, y=sl*x_line+inter,
                mode="lines", name=f"r={r:.3f} p={p:.4f}",
                line={"color":"black","dash":"dash","width":1.5}
            ))
        fig_sc.update_layout(
            title=f"{var1} vs {var2}<br><sup>r={r:.3f} | p={p:.4f}</sup>" if len(sub)>=5 else f"{var1} vs {var2}",
            xaxis_title=f"{var1} ({unit1})",
            yaxis_title=f"{var2} ({cfg2['unit']})",
            template="plotly_white",
            legend={"font":{"size":9}},
            margin={"t":60,"b":40,"l":50,"r":20},
        )
    else:
        fig_sc.update_layout(title="Sélectionnez une 2ème variable",
                             template="plotly_white")

    # ── HEATMAP année × mois ─────────────────────────────────────────────────
    pivot = dff.pivot_table(index="year", columns="month_fr",
                             values=c1, aggfunc="mean").reindex(columns=MOIS_FR)
    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Plasma",
        colorbar={"title":{"text":unit1,"side":"right"}},
        hovertemplate="Année:%{y}<br>Mois:%{x}<br>Valeur:%{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(
        title=f"Heatmap {var1} — année × mois",
        xaxis_title="Mois",
        yaxis_title="Année",
        template="plotly_white",
        margin={"t":50,"b":40,"l":60,"r":20},
    )

    return indicators, fig_ts, fig_seas, fig_sc, fig_hm

# =============================================================================
# 6. LANCEMENT
# =============================================================================

if __name__ == "__main__":
    print("\n✅ Dashboard prêt → http://127.0.0.1:8051\n")
    app.run(debug=True, port=8051)
