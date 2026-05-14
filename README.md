# Modélisation des Risques Financiers et Dérivés OTC au Maroc

## Description

Projet de finance quantitative portant sur :

- la modélisation des risques financiers ;
- les dérivés OTC au Maroc ;
- la gestion du risque de contrepartie ;
- les stratégies de couverture ;
- les simulations Monte Carlo.

Le projet implémente plusieurs modèles quantitatifs appliqués au contexte marocain.

---

## Contenu du Projet

### 5.3.1 — Exposition aux Risques Financiers

#### Risque de taux
- Modèle de Vasicek
- Calibration OLS / MLE
- Simulation Monte Carlo
- Stress testing BAM

#### Risque de change
- Geometric Brownian Motion (GBM)
- Parité couverte des taux (CIP)
- Simulation USD/MAD
- VaR / Expected Shortfall

---

### 5.3.2 — Stratégies de Couverture

#### Interest Rate Swap (IRS)
- Valorisation
- Réduction de variance
- Réduction VaR

#### Forward de Change
- Pricing forward FX
- Couverture importateur
- Analyse de performance

---

### 5.3.3 — Risque de Contrepartie

- EBE / ENE
- Netting ISDA
- EAD
- CVA
- Effet du collatéral

---

### 5.3.4 — Réglementation OTC Marocaine

Comparaison :

- cadre restrictif ;
- cadre flexible prudentiel.

Indicateurs :
- EAD
- CVA
- HHI
- diversification
- spreads OTC

---

## Méthodes Quantitatives

- Monte Carlo
- Vasicek
- GBM
- Value-at-Risk
- Expected Shortfall
- CVA
- Stress testing

---

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Structure du Projet

```text
src/
├── simulations/
├── hedging/
├── metrics/
├── reporting/
└── visualisation/
```

---

## Résultats Principaux

- 10 000 trajectoires simulées
- Calibration du modèle de Vasicek
- Réduction CVA grâce au collatéral
- Analyse réglementaire OTC
- Stress tests BAM

---

## Auteur

HONVOU Brivel

Master Finance Quantitative / Analyse quantitative
