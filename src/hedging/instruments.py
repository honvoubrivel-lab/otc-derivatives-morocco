"""
=============================================================================
hedging/instruments.py — Instruments de couverture (IRS et Forward de change)
=============================================================================
Implémente les deux stratégies de couverture dérivées OTC :

1. SWAP DE TAUX D'INTÉRÊT (IRS) — Section 5.3.2.1
   Transforme une dette à taux variable en dette à taux fixe K.
   Mécanique : l'entreprise paie K et reçoit r_t → le coût devient -N·(K+spread)·Δt

2. CONTRAT FORWARD DE CHANGE — Section 5.3.2.2
   Fixe le taux de change futur à F₀ pour les importations en USD.
   Mécanique : gain_fwd_t = X_mensuel·(S_t - F₀) → coût = -X_mensuel·F₀

Dans les deux cas, le taux variable (r_t ou S_t) disparaît des cash flows
après couverture → variance des CF ≈ 0 (couverture parfaite).
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

from src.config.settings import AGENT, SIM
from src.simulations.vasicek import ParametresVasicek, prix_obligation_vasicek


# =============================================================================
# SECTION 5.3.2.1 — SWAP DE TAUX D'INTÉRÊT (IRS)
# =============================================================================

@dataclass
class ResultatsIRS:
    """Résultats complets de la couverture par IRS."""
    taux_fixe_K: float             # Taux fixe d'équilibre K
    annuite: float                  # Facteur d'annuité Σ(Δt·P)
    facteurs_actualisation: np.ndarray  # P(0, t_i) pour i = 1..n
    cout_irs_theorique: float       # Coût déterministe exact
    flux_swap: np.ndarray           # Flux nets du swap (M × n_periodes)
    cf_couverts: np.ndarray         # CF dette couverte (M × n_periodes)
    cout_couvert: np.ndarray        # Coût total couvert (M,)


def calculer_taux_swap_irs(
    params_vasicek: ParametresVasicek,
    n_periodes: int = None,
    dt: float = SIM.dt_mensuel,
) -> Tuple[float, np.ndarray, float]:
    """
    Calcule le taux fixe K d'un IRS par la condition d'absence d'arbitrage.

    Condition d'équilibre (NPV = 0 à l'initiation) :
        Valeur jambe variable  =  Valeur jambe fixe
        N × [1 - P(0,T)]       =  N × K × Δt × Σ P(0, t_i)

    D'où :
        K = (1 - P(0,T)) / (Δt × Σ P(0, t_i))

    Les facteurs d'actualisation P(0, t_i) sont calculés avec la formule
    analytique de Vasicek (cohérence avec le modèle de taux simulé).

    Parameters
    ----------
    params_vasicek : ParametresVasicek
        Paramètres du modèle de taux calibré.
    n_periodes : int
        Nombre de paiements (mois).
    dt : float
        Accrual factor mensuel.

    Returns
    -------
    tuple
        (K, facteurs_P, annuite) — taux fixe, courbe P, facteur d'annuité
    """
    if n_periodes is None:
        n_periodes = AGENT.n_periodes_dette

    T = AGENT.maturite_dette

    # Maturités des paiements : 1/12, 2/12, ..., 36/12
    maturites = np.arange(1, n_periodes + 1) * dt

    # Facteurs d'actualisation par la formule analytique de Vasicek
    P = prix_obligation_vasicek(maturites, params_vasicek)   # (n_periodes,)
    P_T = prix_obligation_vasicek(T, params_vasicek)          # scalaire

    # Facteur d'annuité : Σ(Δt × P(0, t_i))
    annuite = np.sum(dt * P)

    # Taux fixe d'équilibre K
    K = (1 - P_T) / annuite

    return K, P, annuite


def appliquer_couverture_irs(
    cf_dette_nc: np.ndarray,
    r_sim: np.ndarray,
    params_vasicek: ParametresVasicek,
    notionnel: float = AGENT.notionnel,
    spread: float = AGENT.spread_bancaire,
    dt: float = SIM.dt_mensuel,
) -> ResultatsIRS:
    """
    Applique la couverture IRS et calcule tous les flux couverts.

    L'IRS est un swap payeur de taux fixe :
      - L'entreprise PAIE : N × K × Δt  (fixe)
      - L'entreprise REÇOIT : N × r_t × Δt  (variable)
      - Flux net swap : N × (r_t - K) × Δt

    Après ajout du swap à la dette variable :
        CF_couvert = CF_dette_variable + Flux_swap
                   = -N(r_t + s)Δt + N(r_t - K)Δt
                   = -N(K + s)Δt   (constant, r_t disparu !)

    La variance des CF couverts est théoriquement nulle → couverture parfaite.

    Parameters
    ----------
    cf_dette_nc : np.ndarray, shape (M, n_periodes)
        Cash flows de la dette non couverte.
    r_sim : np.ndarray, shape (M, n_periodes+1)
        Taux simulés (utilisés pour les flux du swap).
    params_vasicek : ParametresVasicek
        Paramètres pour le calcul de K.
    notionnel : float
        Capital emprunté.
    spread : float
        Spread bancaire.
    dt : float
        Accrual factor mensuel.

    Returns
    -------
    ResultatsIRS
        Structure complète des résultats IRS.
    """
    n_periodes = cf_dette_nc.shape[1]

    # Calcul du taux fixe K
    K, P, annuite = calculer_taux_swap_irs(params_vasicek, n_periodes, dt)

    # Taux variables futurs (M × n_periodes)
    r_variable = r_sim[:, 1:]

    # Flux nets du swap : N × (r_t - K) × Δt
    flux_swap = notionnel * (r_variable - K) * dt

    # Cash flows couverts
    cf_couverts = cf_dette_nc + flux_swap

    # Coût total couvert (doit être quasi-constant)
    cout_couvert = cf_couverts.sum(axis=1)

    # Valeur théorique exacte (déterministe)
    cout_theorique = -notionnel * (K + spread) * dt * n_periodes

    return ResultatsIRS(
        taux_fixe_K             = K,
        annuite                 = annuite,
        facteurs_actualisation  = P,
        cout_irs_theorique      = cout_theorique,
        flux_swap               = flux_swap,
        cf_couverts             = cf_couverts,
        cout_couvert            = cout_couvert,
    )


# =============================================================================
# SECTION 5.3.2.2 — CONTRAT FORWARD DE CHANGE
# =============================================================================

@dataclass
class ResultatsForward:
    """Résultats complets de la couverture par forward de change."""
    taux_forward_F0: float          # Taux forward F₀
    prime_abs: float                # Prime forward absolue (MAD)
    prime_pct: float                # Prime forward relative (%)
    cout_budget: float              # Coût budgété (basé sur S₀)
    cout_fwd_theorique: float       # Coût déterministe couvert
    gain_forward: np.ndarray        # Gains sur le forward (M × n_periodes)
    cf_couverts: np.ndarray         # CF importations couvertes (M × n_periodes)
    cout_couvert: np.ndarray        # Coût total couvert (M,)


def calculer_taux_forward_cip(
    s0: float,
    r_d: float,
    r_f: float,
    T: float = AGENT.horizon_couverture_fx,
) -> float:
    """
    Calcule le taux forward F₀ par la Parité des Taux d'Intérêt Couverte (CIP).

    La CIP est une relation d'absence d'arbitrage :
    un investisseur est indifférent entre :
      (a) Placer 1 MAD au taux r_d pendant T
      (b) Convertir en USD (×1/S₀), placer au taux r_f, revenir en MAD (×F₀)

    Égalité des rendements :
        exp(r_d × T) = (1/S₀) × exp(r_f × T) × F₀
        ⟹  F₀ = S₀ × exp((r_d - r_f) × T)

    Note : On utilise θ calibré comme r_d (taux BAM long terme)
    pour la valorisation du forward, ce qui est cohérent avec
    le modèle de Vasicek utilisé pour les simulations.

    Parameters
    ----------
    s0 : float
        Taux spot USD/MAD.
    r_d : float
        Taux domestique MAD (θ Vasicek).
    r_f : float
        Taux étranger USD (SOFR).
    T : float
        Horizon en années.

    Returns
    -------
    float
        Taux forward F₀.
    """
    return s0 * np.exp((r_d - r_f) * T)


def appliquer_couverture_forward(
    cf_fx_nc: np.ndarray,
    s_sim: np.ndarray,
    s0: float,
    r_d: float,
    r_f: float,
    montant_mensuel_usd: float = AGENT.importations_mensuelles_usd,
    n_periodes: int = AGENT.n_periodes_fx,
) -> ResultatsForward:
    """
    Applique la couverture par forward de change et calcule les flux couverts.

    L'importateur vend les USD forward à F₀ pour chaque mois de l'année.
    À chaque date de livraison :
      - Il achète USD au marché :  -X_mensuel × S_t
      - Il reçoit du forward  :   +X_mensuel × (S_t - F₀)
      - CF net couvert         :   -X_mensuel × F₀  (constant !)

    Le gain sur le forward compense exactement la variation de S_t.

    Parameters
    ----------
    cf_fx_nc : np.ndarray, shape (M, n_periodes)
        Coûts d'importation non couverts.
    s_sim : np.ndarray, shape (M, n_periodes+1)
        Trajectoires USD/MAD simulées.
    s0 : float
        Taux spot S₀.
    r_d : float
        Taux BAM (pour le calcul de F₀).
    r_f : float
        Taux SOFR.
    montant_mensuel_usd : float
        Montant mensuel à couvrir.
    n_periodes : int
        Nombre de mois couverts.

    Returns
    -------
    ResultatsForward
        Structure complète des résultats forward.
    """
    T_fx = AGENT.horizon_couverture_fx

    # Calcul du taux forward F₀ via CIP
    F0 = calculer_taux_forward_cip(s0, r_d, r_f, T_fx)
    prime_abs = F0 - s0
    prime_pct = (F0 - s0) / s0 * 100

    # Taux de change futurs (M × n_periodes)
    s_periodes = s_sim[:, 1:]

    # Gain sur le forward à chaque période : X_mensuel × (S_t - F₀)
    gain_forward = montant_mensuel_usd * (s_periodes - F0)

    # CF couverts : CF_fx + Gain_forward = -X_mensuel × F₀ (constant)
    cf_couverts = cf_fx_nc + gain_forward
    cout_couvert = cf_couverts.sum(axis=1)

    # Coûts de référence
    cout_budget    = -montant_mensuel_usd * s0 * n_periodes
    cout_theorique = -montant_mensuel_usd * F0 * n_periodes

    return ResultatsForward(
        taux_forward_F0    = F0,
        prime_abs          = prime_abs,
        prime_pct          = prime_pct,
        cout_budget        = cout_budget,
        cout_fwd_theorique = cout_theorique,
        gain_forward       = gain_forward,
        cf_couverts        = cf_couverts,
        cout_couvert       = cout_couvert,
    )
