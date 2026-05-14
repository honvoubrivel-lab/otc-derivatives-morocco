"""
=============================================================================
simulations/cash_flows.py — Calcul des cash flows d'exposition
=============================================================================
Calcule les cash flows non couverts (exposition brute) pour :
  1. La dette à taux variable (section 5.3.1.1)
  2. Les importations en devises (section 5.3.1.2)

Ces flux constituent la base des calculs de risque et de couverture.
=============================================================================
"""

import numpy as np
from typing import Tuple

from src.config.settings import AGENT, SIM


# =============================================================================
# SECTION 5.3.1.1 — EXPOSITION AU RISQUE DE TAUX
# =============================================================================

def calculer_charges_dette_variable(
    r_sim: np.ndarray,
    notionnel: float = AGENT.notionnel,
    spread: float = AGENT.spread_bancaire,
    dt: float = SIM.dt_mensuel,
) -> np.ndarray:
    """
    Calcule les charges financières mensuelles d'une dette à taux variable.

    Formule :
        CF_t = -N × (r_t + spread) × Δt

    Le signe négatif indique une sortie de trésorerie.
    Le spread représente la marge commerciale de la banque (+150bp).

    Parameters
    ----------
    r_sim : np.ndarray, shape (M, n_periodes+1)
        Trajectoires de taux simulés (colonne 0 = r₀ non utilisée).
    notionnel : float
        Capital emprunté (MAD).
    spread : float
        Marge commerciale en décimal (ex. 0.015 = 150bp).
    dt : float
        Pas de temps mensuel.

    Returns
    -------
    np.ndarray, shape (M, n_periodes)
        Charges mensuelles pour chaque trajectoire (valeurs négatives).
    """
    # On exclut la colonne r₀ (indice 0) car ce sont les taux futurs qui comptent
    taux_variables = r_sim[:, 1:]
    taux_effectifs = taux_variables + spread

    return -notionnel * taux_effectifs * dt


def calculer_tresorerie_taux(
    r_sim: np.ndarray,
    notionnel: float = AGENT.notionnel,
    spread: float = AGENT.spread_bancaire,
    dt: float = SIM.dt_mensuel,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les charges mensuelles et la trésorerie cumulée (risque de taux).

    Returns
    -------
    tuple
        (charges_mensuelles, tresorerie_cumulee) — shapes (M, n_periodes)
    """
    charges = calculer_charges_dette_variable(r_sim, notionnel, spread, dt)
    tresorerie_cumulee = charges.cumsum(axis=1)
    return charges, tresorerie_cumulee


# =============================================================================
# SECTION 5.3.1.2 — EXPOSITION AU RISQUE DE CHANGE
# =============================================================================

def calculer_couts_importation(
    s_sim: np.ndarray,
    montant_mensuel_usd: float = AGENT.importations_mensuelles_usd,
) -> np.ndarray:
    """
    Calcule le coût mensuel d'importation en MAD (exposition brute).

    L'importateur doit acheter X_mensuel USD chaque mois.
    En MAD, le coût est : CF_t^FX = -X_mensuel × S_t

    Parameters
    ----------
    s_sim : np.ndarray, shape (M, n_periodes+1)
        Trajectoires du taux de change simulé.
    montant_mensuel_usd : float
        Montant mensuel à couvrir en USD.

    Returns
    -------
    np.ndarray, shape (M, n_periodes)
        Coûts mensuels en MAD (valeurs négatives = sorties).
    """
    # On exclut S₀ (colonne 0) — seuls les taux futurs génèrent des flux
    return -montant_mensuel_usd * s_sim[:, 1:]


def calculer_perte_change(
    s_sim: np.ndarray,
    s0: float,
    montant_mensuel_usd: float = AGENT.importations_mensuelles_usd,
) -> np.ndarray:
    """
    Calcule la perte de change mensuelle par rapport au budget (taux S₀).

    ΔCF_t = -X_mensuel × (S_t - S₀)
    - Si S_t > S₀ → dirham déprécié → perte (ΔCF < 0)
    - Si S_t < S₀ → dirham apprécié → gain (ΔCF > 0)

    Parameters
    ----------
    s_sim : np.ndarray
        Trajectoires du taux de change.
    s0 : float
        Taux spot initial S₀ (base budgétaire).
    montant_mensuel_usd : float
        Montant mensuel en USD.

    Returns
    -------
    np.ndarray, shape (M, n_periodes)
        Perte/gain mensuel vs budget.
    """
    return -montant_mensuel_usd * (s_sim[:, 1:] - s0)


def calculer_tresorerie_fx(
    s_sim: np.ndarray,
    s0: float,
    montant_mensuel_usd: float = AGENT.importations_mensuelles_usd,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule les coûts, pertes et trésorerie cumulée (exposition FX).

    Returns
    -------
    tuple
        (couts_mensuels, pertes_change, cout_cumule) — shapes (M, n_periodes)
    """
    couts = calculer_couts_importation(s_sim, montant_mensuel_usd)
    pertes = calculer_perte_change(s_sim, s0, montant_mensuel_usd)
    cumule = couts.cumsum(axis=1)
    return couts, pertes, cumule


# =============================================================================
# SÉLECTION DES SCÉNARIOS REPRÉSENTATIFS
# =============================================================================

def identifier_scenarios_representatifs(
    valeurs_finales: np.ndarray,
) -> Tuple[int, int, int]:
    """
    Identifie les indices des trajectoires les plus proches des percentiles
    5%, 50% (médiane) et 95% de la distribution terminale.

    Utilisé pour sélectionner 3 scénarios représentatifs :
      - Central  : trajectoire proche de la médiane
      - Haussier : trajectoire proche du 95e percentile
      - Baissier : trajectoire proche du 5e percentile

    Parameters
    ----------
    valeurs_finales : np.ndarray
        Vecteur (M,) des valeurs à la date terminale.

    Returns
    -------
    tuple
        (idx_central, idx_haussier, idx_baissier)
    """
    idx_central  = int(np.argmin(np.abs(valeurs_finales - np.median(valeurs_finales))))
    idx_haussier = int(np.argmin(np.abs(valeurs_finales - np.percentile(valeurs_finales, 95))))
    idx_baissier = int(np.argmin(np.abs(valeurs_finales - np.percentile(valeurs_finales, 5))))
    return idx_central, idx_haussier, idx_baissier
