"""
=============================================================================
metrics/counterparty.py — Risque de contrepartie (Section 5.3.3)
=============================================================================
Modélise le risque de contrepartie OTC :
  - MTM simulés (Mark-to-Market)
  - EBE (Exposition Brute Espérée)
  - ENE (Exposition Nette Espérée)
  - Netting Factor
  - EAD (Exposure at Default)
  - CVA (Credit Value Adjustment)
  - Effets du collatéral CSA et du netting ISDA
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.config.settings import SIM, AGENT


# =============================================================================
# STRUCTURE DE RÉSULTATS
# =============================================================================

@dataclass
class ResultatsContrepartie:
    """Résultats complets de l'analyse du risque de contrepartie."""
    # Expositions brutes
    ebe: np.ndarray          # Exposition Brute Espérée par période
    ebe_totale: float        # EBE cumulée

    # Expositions nettes (après netting)
    ene: np.ndarray          # Exposition Nette Espérée par période
    ene_totale: float        # ENE cumulée

    # Netting Factor
    netting_factor: float    # ENE / EBE (ratio de compression)

    # Exposure at Default
    ead_brute: float         # EAD sans atténuation
    ead_nette: float         # EAD après netting
    ead_collat: float        # EAD après netting + collatéral

    # CVA
    cva: float               # Credit Value Adjustment
    cva_collat: float        # CVA après collatéralisation

    # Distributions des expositions par simulation
    dist_brute: np.ndarray   # Distribution (M,)
    dist_nette: np.ndarray   # Distribution (M,)
    dist_ead: np.ndarray     # Distribution EAD finale (M,)


# =============================================================================
# CALCUL DES MTM SIMULÉS
# =============================================================================

def calculer_mtm_irs(
    flux_swap: np.ndarray,
    taux_actualisation: float = None,
    dt: float = SIM.dt_mensuel,
) -> np.ndarray:
    """
    Calcule les valeurs Mark-to-Market de l'IRS à chaque période.

    MTM_t^IRS = Σ_{s>t} flux_swap_s × DF(t,s)

    où DF(t,s) = exp(-r × (s-t)×dt) est le facteur d'actualisation.

    Le MTM est positif si les taux ont monté au-dessus de K (in-the-money),
    négatif si les taux sont tombés en dessous (out-of-the-money).

    Parameters
    ----------
    flux_swap : np.ndarray, shape (M, n_periodes)
        Flux nets du swap pour chaque trajectoire.
    taux_actualisation : float, optional
        Taux d'actualisation (défaut : 3%).
    dt : float
        Pas de temps mensuel.

    Returns
    -------
    np.ndarray, shape (M, n_periodes)
        MTM du swap à chaque période.
    """
    if taux_actualisation is None:
        taux_actualisation = 0.03

    M, n = flux_swap.shape
    mtm = np.zeros((M, n))

    for t in range(n):
        # Flux futurs à partir de t+1
        flux_futurs = flux_swap[:, t:]
        n_futurs = flux_futurs.shape[1]
        # Facteurs d'actualisation pour les périodes futures
        horizons = np.arange(1, n_futurs + 1) * dt
        df = np.exp(-taux_actualisation * horizons)
        # MTM = valeur actualisée des flux futurs
        mtm[:, t] = (flux_futurs * df).sum(axis=1)

    return mtm


def calculer_mtm_forward(
    gain_forward: np.ndarray,
    taux_actualisation: float = None,
    dt: float = SIM.dt_mensuel,
) -> np.ndarray:
    """
    Calcule les valeurs MTM du contrat forward à chaque période.

    MTM_t^FWD = Σ_{s>t} gain_forward_s × DF(t,s)

    Parameters
    ----------
    gain_forward : np.ndarray, shape (M, n_periodes)
        Gains/pertes sur le forward à chaque mois.
    taux_actualisation : float
        Taux d'actualisation.
    dt : float
        Pas de temps.

    Returns
    -------
    np.ndarray, shape (M, n_periodes)
        MTM du forward à chaque période.
    """
    if taux_actualisation is None:
        taux_actualisation = 0.03

    M, n = gain_forward.shape
    mtm = np.zeros((M, n))

    for t in range(n):
        flux_futurs = gain_forward[:, t:]
        n_futurs = flux_futurs.shape[1]
        horizons = np.arange(1, n_futurs + 1) * dt
        df = np.exp(-taux_actualisation * horizons)
        mtm[:, t] = (flux_futurs * df).sum(axis=1)

    return mtm


# =============================================================================
# EXPOSITION BRUTE ESPÉRÉE (EBE)
# =============================================================================

def calculer_ebe(mtm: np.ndarray) -> np.ndarray:
    """
    Calcule l'Exposition Brute Espérée (EBE) à chaque période.

    EBE_t = E[max(MTM_t, 0)]

    L'EBE représente la perte potentielle si la contrepartie fait défaut
    à t et que le contrat a une valeur positive (favorable à l'entreprise).
    Sans accord de netting, chaque contrat est traité séparément.

    Parameters
    ----------
    mtm : np.ndarray, shape (M, n_periodes)
        Valeurs MTM simulées.

    Returns
    -------
    np.ndarray, shape (n_periodes,)
        EBE à chaque pas de temps.
    """
    return np.maximum(mtm, 0).mean(axis=0)


# =============================================================================
# EXPOSITION NETTE ESPÉRÉE (ENE) — NETTING ISDA
# =============================================================================

def calculer_ene(
    mtm_irs: np.ndarray,
    mtm_fwd: np.ndarray,
) -> np.ndarray:
    """
    Calcule l'Exposition Nette Espérée (ENE) après netting ISDA.

    En présence d'un accord de netting bilatéral (ISDA Master Agreement),
    les expositions positives et négatives des différents contrats avec
    une même contrepartie se compensent :

        MTM_net_t = MTM_IRS_t + MTM_FWD_t
        ENE_t = E[max(MTM_net_t, 0)]

    Le netting réduit significativement l'exposition brute.

    Parameters
    ----------
    mtm_irs : np.ndarray, shape (M, n_irs)
        MTM du swap de taux.
    mtm_fwd : np.ndarray, shape (M, n_fwd)
        MTM du forward de change.

    Returns
    -------
    np.ndarray, shape (n_commun,)
        ENE à chaque pas de temps.
    """
    # Alignement sur la longueur commune
    n_commun = min(mtm_irs.shape[1], mtm_fwd.shape[1])
    mtm_net = mtm_irs[:, :n_commun] + mtm_fwd[:, :n_commun]
    return np.maximum(mtm_net, 0).mean(axis=0)


def calculer_netting_factor(ebe_totale: float, ene_totale: float) -> float:
    """
    Calcule le Netting Factor (facteur de compression).

    NF = ENE / EBE ∈ [0, 1]

    - NF = 1 : pas de compensation (netting sans effet)
    - NF → 0 : netting très efficace (compensation quasi-totale)

    Un NF faible indique que les contrats sont corrélés négativement,
    permettant une forte compensation des expositions.

    Parameters
    ----------
    ebe_totale : float
        Exposition Brute Espérée cumulée.
    ene_totale : float
        Exposition Nette Espérée cumulée.

    Returns
    -------
    float
        Netting Factor ∈ [0, 1].
    """
    return float(ene_totale / ebe_totale) if ebe_totale > 0 else 1.0


# =============================================================================
# EXPOSURE AT DEFAULT (EAD)
# =============================================================================

def calculer_ead_complet(
    ebe_totale: float,
    ene_totale: float,
    taux_collateral: float = 0.80,
) -> Dict[str, float]:
    """
    Calcule l'EAD aux différents stades d'atténuation du risque.

    Cascade de réduction :
        EAD_brute          = EBE_totale
        EAD_nette          = ENE_totale          (après netting ISDA)
        EAD_collat         = ENE × (1 - α_c)     (après collatéral CSA)

    où α_c = taux de collatéralisation (fraction couverte).

    Parameters
    ----------
    ebe_totale : float
        EBE cumulée.
    ene_totale : float
        ENE cumulée.
    taux_collateral : float
        Taux de collatéralisation (0.80 = 80% couvert).

    Returns
    -------
    dict
        EAD aux trois niveaux et les réductions relatives.
    """
    ead_brute = ebe_totale
    ead_nette = ene_totale
    ead_collat = max(ene_totale * (1 - taux_collateral), 0.0)

    red_netting  = (ead_brute - ead_nette) / ead_brute * 100 if ead_brute > 0 else 0.0
    red_collat   = (ead_nette - ead_collat) / ead_nette * 100 if ead_nette > 0 else 0.0
    red_totale   = (ead_brute - ead_collat) / ead_brute * 100 if ead_brute > 0 else 0.0

    return {
        "ead_brute"          : ead_brute,
        "ead_nette"          : ead_nette,
        "ead_collat"         : ead_collat,
        "reduction_netting"  : red_netting,
        "reduction_collat"   : red_collat,
        "reduction_totale"   : red_totale,
    }


# =============================================================================
# CREDIT VALUE ADJUSTMENT (CVA)
# =============================================================================

def calculer_cva_complet(
    ene: np.ndarray,
    probabilite_defaut_annuelle: float = 0.02,
    taux_recouvrement: float = 0.40,
    taux_actualisation: float = 0.03,
    dt: float = SIM.dt_mensuel,
    taux_collateral: float = 0.0,
) -> Dict[str, float]:
    """
    Calcule le CVA selon la formule standard (Bâle III / BCBS) :

        CVA = LGD × Σ_t  PD_t × EPE_t × DF_t

    où :
        LGD   = 1 - R                  (Loss Given Default)
        PD_t  = P[défaut à t]         (probabilité de défaut marginale)
        EPE_t = ENE_t (ou ENE × (1-α)) (exposition positive attendue)
        DF_t  = exp(-r × t × dt)       (facteur d'actualisation sans risque)

    Deux calculs sont réalisés :
      - CVA sans collatéral  : EPE = ENE
      - CVA avec collatéral  : EPE = ENE × (1 - taux_collateral)

    Parameters
    ----------
    ene : np.ndarray, shape (n_periodes,)
        Exposition Nette Espérée par période.
    probabilite_defaut_annuelle : float
        PD annuelle de la contrepartie (ex. 0.02 = 2%).
    taux_recouvrement : float
        Taux de recouvrement R (ex. 0.40 = 40%).
    taux_actualisation : float
        Taux sans risque pour l'actualisation.
    dt : float
        Pas de temps.
    taux_collateral : float
        Taux de collatéralisation (0 = aucun collatéral).

    Returns
    -------
    dict
        CVA sans et avec collatéral, et réduction due au collatéral.
    """
    lgd = 1 - taux_recouvrement
    n   = len(ene)

    # Probabilité de défaut marginale par période
    pd_periode = 1 - (1 - probabilite_defaut_annuelle) ** dt

    # Facteurs d'actualisation
    periodes = np.arange(1, n + 1)
    df = np.exp(-taux_actualisation * periodes * dt)

    # CVA sans collatéral
    cva_sans = float(lgd * pd_periode * np.sum(ene * df))

    # CVA avec collatéral (exposition réduite)
    epe_collat = ene * (1 - taux_collateral)
    cva_avec   = float(lgd * pd_periode * np.sum(epe_collat * df))

    reduction_cva = (cva_sans - cva_avec) / cva_sans * 100 if cva_sans > 0 else 0.0

    return {
        "cva_sans_collat"   : cva_sans,
        "cva_avec_collat"   : cva_avec,
        "reduction_cva_pct" : reduction_cva,
        "lgd"               : lgd,
        "pd_periode"        : pd_periode,
    }


# =============================================================================
# ANALYSE COMPLÈTE DU RISQUE DE CONTREPARTIE
# =============================================================================

def analyser_risque_contrepartie(
    flux_swap: np.ndarray,
    gain_forward: np.ndarray,
    taux_collateral: float = 0.80,
    probabilite_defaut: float = 0.02,
    taux_recouvrement: float = 0.40,
) -> ResultatsContrepartie:
    """
    Pipeline complet d'analyse du risque de contrepartie.

    Étapes :
      1. Calcul des MTM simulés (IRS et Forward)
      2. EBE (avant netting)
      3. ENE (après netting ISDA)
      4. Netting Factor
      5. EAD aux trois niveaux
      6. CVA sans et avec collatéral
      7. Distributions des expositions

    Parameters
    ----------
    flux_swap : np.ndarray, shape (M, n_taux)
        Flux nets du swap.
    gain_forward : np.ndarray, shape (M, n_fx)
        Gains/pertes sur le forward.
    taux_collateral : float
        Taux de couverture par collatéral.
    probabilite_defaut : float
        PD annuelle de la contrepartie.
    taux_recouvrement : float
        Taux de recouvrement LGD.

    Returns
    -------
    ResultatsContrepartie
        Tous les résultats consolidés.
    """
    # ── MTM simulés ──────────────────────────────────────────────────────────
    mtm_irs = calculer_mtm_irs(flux_swap)
    mtm_fwd = calculer_mtm_forward(gain_forward)

    # ── EBE (sans netting) ────────────────────────────────────────────────────
    ebe_irs = calculer_ebe(mtm_irs)
    ebe_fwd = calculer_ebe(mtm_fwd)

    # EBE combinée : somme des EBE individuelles (sans compensation)
    n_commun = min(len(ebe_irs), len(ebe_fwd))
    ebe_total = ebe_irs[:n_commun] + ebe_fwd[:n_commun]
    ebe_totale = float(ebe_total.sum())

    # ── ENE (avec netting ISDA) ───────────────────────────────────────────────
    ene = calculer_ene(mtm_irs, mtm_fwd)
    ene_totale = float(ene.sum())

    # ── Netting Factor ────────────────────────────────────────────────────────
    nf = calculer_netting_factor(ebe_totale, ene_totale)

    # ── EAD aux trois niveaux ─────────────────────────────────────────────────
    ead_res = calculer_ead_complet(ebe_totale, ene_totale, taux_collateral)

    # ── CVA ───────────────────────────────────────────────────────────────────
    cva_res = calculer_cva_complet(
        ene,
        probabilite_defaut_annuelle = probabilite_defaut,
        taux_recouvrement           = taux_recouvrement,
        taux_collateral             = taux_collateral,
    )

    # ── Distributions des expositions par simulation ─────────────────────────
    # Exposition brute par simulation : max(MTM_IRS, 0) + max(MTM_FWD, 0)
    dist_brute = (
        np.maximum(mtm_irs[:, :n_commun], 0).sum(axis=1)
        + np.maximum(mtm_fwd[:, :n_commun], 0).sum(axis=1)
    )
    # Exposition nette par simulation : max(MTM_IRS + MTM_FWD, 0)
    mtm_net = mtm_irs[:, :n_commun] + mtm_fwd[:, :n_commun]
    dist_nette = np.maximum(mtm_net, 0).sum(axis=1)
    # EAD après collatéral
    dist_ead = np.maximum(dist_nette * (1 - taux_collateral), 0)

    return ResultatsContrepartie(
        ebe            = ebe_total,
        ebe_totale     = ebe_totale,
        ene            = ene,
        ene_totale     = ene_totale,
        netting_factor = nf,
        ead_brute      = ead_res["ead_brute"],
        ead_nette      = ead_res["ead_nette"],
        ead_collat     = ead_res["ead_collat"],
        cva            = cva_res["cva_sans_collat"],
        cva_collat     = cva_res["cva_avec_collat"],
        dist_brute     = dist_brute,
        dist_nette     = dist_nette,
        dist_ead       = dist_ead,
    )
