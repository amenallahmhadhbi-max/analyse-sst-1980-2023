# Analyse SST 1980–2023 — Migration R → Python

**Auteur :** Amenallah Mhadhbi  
**Encadrante :** Ines Abdeljaoued-Tej  
**Période :** Février–Mai 2026

---

## Question scientifique

> **La température de surface de la mer (SST) a-t-elle augmenté entre 1980
> et 2023, et si oui, à quel rythme et avec quelles variations régionales
> et saisonnières ?**

---

## Structure du projet

```
projet_climat/
│
├── data/                          ← Fichiers NetCDF SST (non inclus dans le dépôt)
│   └── catalogue_sst.csv          ← Cache généré automatiquement au 1er lancement
│
├── analyse_sst.py                 ← Pipeline complet R → Python (Étapes 1–4)
├── dashboard.py                   ← Tableau de bord interactif Dash (Étape 5)
│
├── qualite_donnees.md             ← Sources, NA, anomalies, biais
└── README.md                      ← Ce fichier
```

---

## Données

Les fichiers NetCDF sont issus du **Climate Data Store (CDS) de Copernicus** :

- **Dataset** : ESA SST CCI — GHRSST Level 4 OSTIA
- **Lien** : https://cds.climate.copernicus.eu
- **Résolution** : 0.05°/pixel, mensuel, 1980–2023
- **Format** : NetCDF-4 (`.nc`)
- Placer tous les fichiers `.nc` dans le dossier `data/`

---

## Installation

```bash
pip install xarray netCDF4 numpy pandas matplotlib seaborn scipy cartopy tqdm dash plotly statsmodels
```

---

## Lancement

### 1. Analyse complète (Étapes 1 à 4)

Génère toutes les figures statiques dans le dossier courant :

```bash
python analyse_sst.py
```

Figures produites :
- `carte_globale_sst.png`
- `boxplot_regions.png`
- `na_par_region.png`
- `heatmap_regions.png`
- `carte_region_selectionnee.png`
- `evolution_sst_region.png`
- `tendances_region.png`
- `evolution_temporelle_mensuelle.png`
- `cycle_annuel_sst.png`
- `heatmap_annuelle.png`
- `boxplots_mensuels.png`
- `tendances_mensuelles.png`
- `anomalies_mensuelles.png`
- `serie_temporelle_rolling.png`
- `comparaison_saisonniere.png`
- `carte_evolution_spatiale.png`
- `scatter_correlation_sst.png`

### 2. Tableau de bord interactif (Étape 5)

```bash
python dashboard.py
```

Puis ouvrir dans le navigateur : **http://127.0.0.1:8050**

> **Note** : Au premier lancement, le catalogue CSV est construit
> automatiquement (~2–5 min). Les lancements suivants sont instantanés.

---

## Tableau de bord — Fonctionnalités

| Contrôle | Description |
|----------|-------------|
| **Région océanique** | Sélection parmi 8 régions (Mer Rouge, Méditerranée, etc.) |
| **Période d'analyse** | Slider années 1980–2023 |
| **Variable** | SST brute (°C) ou Anomalie (°C) |

| Graphique | Description |
|-----------|-------------|
| **Série temporelle** | SST mensuelle + moyenne mobile 12 mois + tendance |
| **Boxplots saisonniers** | Distribution par saison (Hiver/Printemps/Été/Automne) |
| **Heatmap année × mois** | Évolution inter-annuelle et saisonnière |
| **Carte spatiale** | Réchauffement début vs fin de période |

| Indicateur | Description |
|------------|-------------|
| SST moyenne | Moyenne sur la période sélectionnée |
| SST dernière année | Valeur de la dernière année |
| SST max / min | Extrêmes observés |
| Tendance | °C par décennie |
| Variation totale | Différence première → dernière année |

---

## Résultats principaux

- Réchauffement **significatif** (p < 0.001) sur toutes les régions et tous
  les mois analysés
- **Mer Rouge** : tendance de +0.149 à +0.272°C/décennie selon le mois,
  avec un maximum en Octobre
- **Juillet** est le mois qui se réchauffe le plus vite à l'échelle globale
  (+0.144°C/décennie)
- Réchauffement spatial moyen entre 1980–1985 et 2018–2023 : **+1.0 à +1.5°C**
  selon la région

---

## Correspondance R → Python

| R | Python |
|---|--------|
| `ncdf4` + `raster` | `xarray` + `numpy` |
| `dplyr` / `tidyr` | `pandas` |
| `ggplot2` | `matplotlib` + `seaborn` |
| `lubridate` | `pandas datetime` |
| `lm()` | `scipy.stats.linregress` |
| `rollmean()` (zoo) | `pandas.Series.rolling()` |
| `loess` | `statsmodels.nonparametric.lowess` |
| Tableau de bord Shiny | **Dash (Plotly)** |
