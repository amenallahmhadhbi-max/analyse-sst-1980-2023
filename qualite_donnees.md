# Qualité des Données — Analyse SST 1980-2023

## 1. Source des données

| Paramètre | Valeur |
|-----------|--------|
| **Dataset** | ESA SST CCI & C3S — GHRSST Level 4 OSTIA |
| **Identifiant CDS** | `satellite-sea-surface-temperature` |
| **Version** | CDR 3.0 (1980–2021) + ICDR 3.0 (2022–2023) |
| **Résolution spatiale** | 0.05° par pixel (~5 km) |
| **Résolution temporelle** | Mensuelle (1 fichier/mois) |
| **Période couverte** | Janvier 1980 – Octobre 2023 |
| **Format** | NetCDF-4 (.nc) |
| **Variable principale** | `analysed_sst` (en Kelvin, convertie en °C) |
| **Nombre de fichiers** | 176 fichiers NetCDF |
| **Dimensions spatiales** | 7200 × 3600 pixels (longitude × latitude) |

---

## 2. Taux de données manquantes (NA)

### 2.1 NA globaux (fichier entier)

Les valeurs manquantes sont **structurelles** : elles correspondent aux terres
émergées et aux calottes glaciaires, non aux océans.

| Métrique | Valeur |
|----------|--------|
| Cellules totales par fichier | 25 920 000 |
| Cellules valides (océan) | ~17 195 230 (66.3%) |
| Cellules NA (terres/glaces) | ~8 724 770 (33.7%) |
| Cohérence inter-fichiers | ✅ Stable à 33.7% sur tous les 176 fichiers |

### 2.2 NA par région océanique

Les pourcentages sont calculés par rapport au **total global** du fichier
(25 920 000 cellules), conformément à la méthodologie du code R de référence.

| Région | NA moyen (%) | Interprétation |
|--------|-------------|----------------|
| Mer Rouge | ~0.0% | Mer entièrement océanique, bbox très ciblée |
| Atlantique Nord | ~22.7% | Côtes et quelques terres incluses |
| Atlantique Sud | ~27.2% | Amérique du Sud partiellement incluse |
| Pacifique Sud | ~35.6% | Grande surface, zones côtières |
| Pacifique Nord | ~39.6% | Asie de l'Est, Alaska partiellement inclus |
| Méditerranée | ~43.5% | Péninsules méditerranéennes |
| Mer Noire | ~46.9% | Petite mer, beaucoup de côtes |
| Océan Indien | ~59.2% | Sous-continent indien et Afrique inclus |

> **Note** : Les NA des régions Pacifique (bbox > 180°) sont calculés via un
> masque 2D pour éviter l'inclusion des terres entre les deux bandes de
> longitudes (Amérique du Sud).

---

## 3. Anomalies détectées et traitement

### 3.1 Valeurs aberrantes

| Type | Détection | Traitement appliqué |
|------|-----------|---------------------|
| Températures < −2°C | Physiquement impossible (gel) | Conservées : point de congélation de l'eau salée ≈ −1.8°C |
| Températures > 34°C | Rares, zones tropicales peu profondes | Conservées : valeurs extrêmes légitimes |
| NA structurels (terres) | Présents dans tous les fichiers | Conservés, exclus des calculs de moyenne |

### 3.2 Cohérence temporelle

- **Aucune rupture de série** détectée entre CDR 3.0 et ICDR 3.0 (jonction 2021–2022)
- Les fichiers couvrent **4 mois par an** dans l'échantillon analysé
  (Janvier, Mai, Juillet, Octobre), ce qui représente les 4 saisons
- Les 176 fichiers ont tous la même résolution spatiale (7200 × 3600)

### 3.3 Biais potentiels

| Biais | Description | Impact estimé |
|-------|-------------|---------------|
| **Interpolation spatiale** | OSTIA utilise un algorithme OI (Optimal Interpolation) pour remplir les zones nuageuses | Lissage des extrêmes locaux |
| **Couverture nuageuse** | Les zones à forte couverture nuageuse sont interpolées, pas mesurées directement | Sous-estimation de la variabilité |
| **Passage satellite** | Résolution temporelle sub-mensuelle non disponible ici | Aliasing des variations rapides |
| **Changement de capteur** | Transition entre différents satellites sur 43 ans | Possible discontinuité légère pré-1990 |

---

## 4. Région d'étude approfondie : Mer Rouge

| Paramètre | Valeur |
|-----------|--------|
| Bbox | lon [32°E, 44°E] — lat [12°N, 30°N] |
| NA moyen | 0.0% |
| Température moyenne 1980–2023 | ~26.1°C |
| Tendance globale | +0.167 à +0.272 °C/décennie selon le mois |
| Mois avec la plus forte tendance | Octobre (+0.272°C/déc) |
| Significativité | p < 0.05 pour tous les mois analysés |

---

## 5. Pipeline de traitement

```
Fichiers NetCDF (Kelvin)
        ↓
Conversion °C : sst = sst_kelvin - 273.15
        ↓
Masque 2D par région (np.meshgrid + bbox)
        ↓
Calcul moyenne spatiale par fichier et par région
        ↓
Catalogue CSV (176 × 8 régions = 1408 observations)
        ↓
Calcul anomalies (référence : moyenne 1991-2020)
        ↓
Régression linéaire par mois (scipy.stats.linregress)
        ↓
Visualisations + Tableau de bord Dash
```

---

## 6. Recommandations

- **Pour une analyse plus robuste** : télécharger les données journalières ou
  hebdomadaires (actuellement mensuel) afin de capturer les événements extrêmes
- **Variable complémentaire suggérée** : vitesse du vent de surface (ERA5) pour
  analyser la corrélation avec la SST
- **Limite principale** : l'échantillonnage à 4 mois/an (Janv, Mai, Juil, Oct)
  sous-représente les mois de transition (Mars, Avril, Septembre)
