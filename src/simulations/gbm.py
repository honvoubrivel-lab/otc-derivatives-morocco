"""
=============================================================================
simulations/gbm.py — Modèle GBM : simulation du taux de change USD/MAD
=============================================================================
Implémente le Mouvement Brownien Géométrique (GBM / Black-Scholes 1973)
pour simuler les trajectoires du taux de change USD/MAD.

Équation différentielle stochastique :
    dS_t = μ·S_t·dt + σ_S·S_t·dW_t^S

Solution exacte (lemme d'Itô) :
    S_{t+Δt} = S_t · exp[ (μ - σ²/2)·Δt + σ·√Δt·ε_t ]

Le drift μ est calibré par la Parité des Taux Couverte (CIP) :
    μ = r_d - r_f

où r_d = taux directeur BAM (θ calibré) et r_f = SOFR (taux USD).

La correction d'Itô (−σ²/2) garantit que E[S_t] = S₀·exp(μ·t),
propriété nécessaire pour l'absence d'arbitrage.
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.config.settings import SIM, AGENT


# =============================================================================
# STRUCTURE DE DONNÉES : PARAMÈTRES GBM
# =============================================================================

@dataclass
class ParametresGBM:
    """
    Paramètres du modèle GBM pour le taux de change.

    Attributs
    ---------
    mu : float
        Drift (= r_d - r_f via CIP).
    sigma_s : float
        Volatilité annualisée du taux de change.
    s0 : float
        Taux spot USD/MAD initial.
    r_d : float
        Taux domestique (BAM).
    r_f : float
        Taux étranger (SOFR / USD).
    """
    mu: float
    sigma_s: float
    s0: float
    r_d: float
    r_f: float

    @property
    def taux_forward(self) -> float:
        """
        Taux forward 1 an via la Parité des Taux d'Intérêt Couverte (CIP) :
            F₀ = S₀ · exp((r_d - r_f) · T)
        """
        T_fx = AGENT.horizon_couverture_fx
        return self.s0 * np.exp((self.r_d - self.r_f) * T_fx)

    @property
    def prime_forward_pct(self) -> float:
        """Prime forward relative (%) par rapport au spot."""
        return (self.taux_forward - self.s0) / self.s0 * 100


# =============================================================================
# SIMULATION MONTE CARLO — SOLUTION EXACTE GBM
# =============================================================================

def simuler_taux_change_gbm(
    params: ParametresGBM,
    n_simulations: int = SIM.n_simulations,
    n_periodes: int = AGENT.n_periodes_fx,
    dt: float = SIM.dt_mensuel,
    graine: int = SIM.graine_aleatoire,
    decalage_graine: int = 10000,
) -> np.ndarray:
    """
    Simule N trajectoires du taux de change USD/MAD selon le modèle GBM.

    La solution exacte est utilisée (pas d'erreur de discrétisation) :
        S_{t+Δt} = S_t · exp[ (μ - σ²/2)·Δt + σ·√Δt·ε_t ]

    Le terme (μ - σ²/2) est la correction d'Itô. Sans cette correction,
    l'espérance serait surestimée car E[exp(X)] > exp(E[X]) pour X gaussien.

    Un décalage de graine (decalage_graine) permet d'utiliser des chocs
    indépendants de ceux utilisés pour le modèle de taux Vasicek, tout en
    conservant la reproductibilité globale.

    Parameters
    ----------
    params : ParametresGBM
        Paramètres μ, σ_S, S₀.
    n_simulations : int
        Nombre de trajectoires.
    n_periodes : int
        Nombre de périodes (mois).
    dt : float
        Pas de temps (1/12 pour mensuel).
    graine : int
        Graine de base.
    decalage_graine : int
        Décalage pour indépendance avec les chocs de taux.

    Returns
    -------
    np.ndarray, shape (n_simulations, n_periodes + 1)
        Trajectoires USD/MAD. Colonne 0 = S₀.
    """
    np.random.seed(graine + decalage_graine)

    # Terme de dérive ajusté par la correction d'Itô
    derive_ajustee = (params.mu - 0.5 * params.sigma_s**2) * dt

    # Chocs gaussiens (M × n_periodes)
    chocs = np.random.normal(0, 1, (n_simulations, n_periodes))

    S_sim = np.zeros((n_simulations, n_periodes + 1))
    S_sim[:, 0] = params.s0

    for t in range(n_periodes):
        # Solution exacte du GBM : multiplication par exponentielle
        S_sim[:, t + 1] = S_sim[:, t] * np.exp(
            derive_ajustee + params.sigma_s * np.sqrt(dt) * chocs[:, t]
        )

    return S_sim


# =============================================================================
# SIMULATION DE SCÉNARIOS DE STRESS (CHANGE)
# =============================================================================

def simuler_scenarios_stress_fx(
    params: ParametresGBM,
    multiplicateurs_s0: Dict[str, float],
    n_simulations: int = SIM.n_simulations,
    n_periodes: int = AGENT.n_periodes_fx,
    dt: float = SIM.dt_mensuel,
    graine: int = SIM.graine_aleatoire,
    decalage_graine: int = 10000,
) -> Dict[str, np.ndarray]:
    """
    Simule des scénarios de stress en multipliant S₀ par un facteur.

    Ex. multiplicateur 1.10 → MAD se déprécie de 10% (USD plus cher).
    Les mêmes chocs aléatoires sont utilisés dans tous les scénarios
    pour isoler l'effet du niveau initial.

    Parameters
    ----------
    params : ParametresGBM
        Paramètres GBM de base.
    multiplicateurs_s0 : dict
        {nom: multiplicateur} — ex. {"Base": 1.0, "Dépréc. -10%": 1.10}
    n_simulations, n_periodes, dt, graine, decalage_graine : cf. `simuler_taux_change_gbm`.

    Returns
    -------
    dict
        {nom_scenario: tableau (n_simulations, n_periodes+1)}
    """
    np.random.seed(graine + decalage_graine)
    derive_ajustee = (params.mu - 0.5 * params.sigma_s**2) * dt
    chocs = np.random.normal(0, 1, (n_simulations, n_periodes))

    resultats = {}
    for nom, mult in multiplicateurs_s0.items():
        s0_stress = params.s0 * mult
        S_stress = np.zeros((n_simulations, n_periodes + 1))
        S_stress[:, 0] = s0_stress

        for t in range(n_periodes):
            S_stress[:, t + 1] = S_stress[:, t] * np.exp(
                derive_ajustee + params.sigma_s * np.sqrt(dt) * chocs[:, t]
            )
        resultats[nom] = S_stress

    return resultats
