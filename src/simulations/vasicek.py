"""
=============================================================================
simulations/vasicek.py — Modèle de Vasicek : calibration et simulation
=============================================================================
Implémente le modèle de taux de Vasicek (1977) pour simuler les trajectoires
du taux directeur BAM sur l'horizon de la dette.

Équation différentielle stochastique (EDS) :
    dr_t = κ·(θ - r_t)·dt + σ·dW_t

Propriétés clés :
  - Retour à la moyenne : κ contrôle la vitesse de convergence vers θ
  - Demi-vie           : ln(2) / κ (temps pour combler la moitié de l'écart)
  - Distribution gaussienne conditionnelle → peut générer des taux négatifs
    (limitation connue du modèle, corrigée par clip à 0)

Calibration disponible :
  1. OLS (Moindres Carrés Ordinaires) — estimation rapide et robuste
  2. MLE (Maximum de Vraisemblance)   — estimation précise, utilisée par défaut
=============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, Any

from src.config.settings import SIM


# =============================================================================
# STRUCTURE DE DONNÉES : RÉSULTATS DE CALIBRATION
# =============================================================================

@dataclass
class ParametresVasicek:
    """
    Paramètres structurels du modèle de Vasicek.

    Attributs
    ---------
    kappa : float
        Vitesse de retour à la moyenne (κ > 0 obligatoire).
    theta : float
        Taux d'équilibre long terme (taux cible du taux directeur).
    sigma : float
        Volatilité instantanée du taux (σ > 0 obligatoire).
    r0 : float
        Taux initial (dernier taux BAM observé, point de départ des simulations).
    log_vraisemblance : float
        Log-vraisemblance maximisée (MLE uniquement).
    r_squared_ols : float
        R² de la régression OLS.
    """
    kappa: float
    theta: float
    sigma: float
    r0: float
    log_vraisemblance: float = 0.0
    r_squared_ols: float = 0.0

    @property
    def demi_vie(self) -> float:
        """Demi-vie du processus (années) : temps pour réduire l'écart de moitié."""
        return np.log(2) / self.kappa


# =============================================================================
# CALIBRATION : MÉTHODE OLS
# =============================================================================

def calibrer_vasicek_ols(
    serie_taux: pd.Series,
    dt: float = SIM.dt_mensuel,
) -> Dict[str, Any]:
    """
    Calibre le modèle de Vasicek par régression OLS sur les variations de taux.

    Discrétisation d'Euler du modèle :
        Δr_t = κ·θ·Δt  +  (-κ·Δt)·r_t  +  ε_t
             =    A    +        B·r_t   +  ε_t

    On régresse Δr_t sur r_t pour estimer A et B,
    puis on récupère κ = -B/Δt et θ = A/(κ·Δt).

    Parameters
    ----------
    serie_taux : pd.Series
        Série temporelle des taux directeurs observés (décimal).
    dt : float
        Pas de temps de la régression (mensuel = 1/12).

    Returns
    -------
    dict
        Paramètres OLS : kappa, theta, sigma, A_hat, B_hat, r_squared, residus.
    """
    r_obs = serie_taux.values
    r_t   = r_obs[:-1]
    r_tp1 = r_obs[1:]
    delta_r = r_tp1 - r_t

    X_reg = sm.add_constant(r_t)
    ols = sm.OLS(delta_r, X_reg).fit()

    A_hat = ols.params[0]
    B_hat = ols.params[1]

    kappa_ols = -B_hat / dt
    theta_ols = A_hat / (kappa_ols * dt) if kappa_ols > 0 else 0.0
    sigma_ols = np.std(ols.resid) / np.sqrt(dt)

    return {
        "kappa"     : kappa_ols,
        "theta"     : theta_ols,
        "sigma"     : sigma_ols,
        "A_hat"     : A_hat,
        "B_hat"     : B_hat,
        "r_squared" : ols.rsquared,
        "residus"   : ols.resid,
        "ols_result": ols,
    }


# =============================================================================
# CALIBRATION : MÉTHODE MLE
# =============================================================================

def calibrer_vasicek_mle(
    serie_taux: pd.Series,
    dt: float = SIM.dt_mensuel,
    params_init: Dict[str, float] = None,
) -> ParametresVasicek:
    """
    Calibre le modèle de Vasicek par Maximum de Vraisemblance (MLE).

    La solution exacte du modèle donne une distribution conditionnelle gaussienne :
        r_{t+Δt} | r_t ~ N(μ_cond, σ²_cond)

    où :
        μ_cond  = θ + (r_t - θ)·exp(-κ·Δt)
        σ²_cond = σ²/(2κ) · (1 - exp(-2κ·Δt))

    On maximise la log-vraisemblance :
        ℓ(κ,θ,σ) = Σ_t [ -½ ln(2πσ²_cond) - (r_{t+1} - μ_cond)² / (2σ²_cond) ]

    Le MLE est plus efficace que l'OLS car il exploite la distribution
    exacte (pas seulement la moyenne conditionnelle).

    Parameters
    ----------
    serie_taux : pd.Series
        Série temporelle des taux observés.
    dt : float
        Pas de temps.
    params_init : dict, optional
        Point de départ de l'optimisation (défaut : résultats OLS).

    Returns
    -------
    ParametresVasicek
        Paramètres calibrés par MLE.
    """
    r_obs = serie_taux.values
    r_t   = r_obs[:-1]
    r_tp1 = r_obs[1:]
    r0    = r_obs[-1]

    # Point de départ : calibration OLS
    ols_res = calibrer_vasicek_ols(serie_taux, dt)
    if params_init is None:
        params_init = [ols_res["kappa"], ols_res["theta"], ols_res["sigma"]]

    def neg_log_vraisemblance(p: np.ndarray) -> float:
        """Fonction objectif : log-vraisemblance négative à minimiser."""
        kappa, theta, sigma = p
        if kappa <= 0 or sigma <= 0:
            return 1e10

        e_k    = np.exp(-kappa * dt)
        mu_c   = theta + (r_t - theta) * e_k
        var_c  = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))

        if var_c <= 0:
            return 1e10

        ll = -0.5 * np.sum(
            np.log(2 * np.pi * var_c) + (r_tp1 - mu_c)**2 / var_c
        )
        return -ll

    resultat = minimize(
        neg_log_vraisemblance,
        x0     = params_init,
        method = "L-BFGS-B",
        bounds = [(1e-6, None), (1e-6, 0.20), (1e-6, 0.50)],
    )

    kappa_mle, theta_mle, sigma_mle = resultat.x

    return ParametresVasicek(
        kappa              = kappa_mle,
        theta              = theta_mle,
        sigma              = sigma_mle,
        r0                 = r0,
        log_vraisemblance  = -resultat.fun,
        r_squared_ols      = ols_res["r_squared"],
    )


# =============================================================================
# SIMULATION MONTE CARLO — SCHÉMA D'EULER-MARUYAMA
# =============================================================================

def simuler_taux_vasicek(
    params: ParametresVasicek,
    n_simulations: int = SIM.n_simulations,
    n_periodes: int = None,
    dt: float = SIM.dt_mensuel,
    graine: int = SIM.graine_aleatoire,
) -> np.ndarray:
    """
    Simule N trajectoires de taux avec le schéma d'Euler-Maruyama.

    Schéma de discrétisation :
        r_{t+Δt} = r_t + κ·(θ - r_t)·Δt + σ·√Δt·ε_t
        ε_t ~ N(0,1) i.i.d.

    Les taux négatifs éventuels (limitation connue de Vasicek)
    sont clippés à 0 pour rester économiquement cohérents.

    Note : On génère toute la matrice de chocs aléatoires d'un coup
    (M × n_periodes) pour vectoriser le calcul et éviter les boucles Python
    sur les trajectoires.

    Parameters
    ----------
    params : ParametresVasicek
        Paramètres calibrés (κ, θ, σ, r₀).
    n_simulations : int
        Nombre de trajectoires Monte Carlo.
    n_periodes : int
        Nombre de pas de temps. Défaut : depuis les paramètres globaux.
    dt : float
        Pas de temps (mensuel = 1/12).
    graine : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    np.ndarray, shape (n_simulations, n_periodes + 1)
        Matrice des trajectoires de taux simulés.
        Colonne 0 = r₀ (état initial), colonnes 1..n_periodes = taux simulés.
    """
    from src.config.settings import AGENT
    if n_periodes is None:
        n_periodes = AGENT.n_periodes_dette

    np.random.seed(graine)

    r_sim = np.zeros((n_simulations, n_periodes + 1))
    r_sim[:, 0] = params.r0

    # Génération vectorisée de tous les chocs gaussiens
    chocs = np.random.normal(0, 1, (n_simulations, n_periodes))

    for t in range(n_periodes):
        derive    = params.kappa * (params.theta - r_sim[:, t]) * dt
        diffusion = params.sigma * np.sqrt(dt) * chocs[:, t]
        r_sim[:, t + 1] = r_sim[:, t] + derive + diffusion

    # Clip des taux négatifs (cohérence économique)
    r_sim = np.maximum(r_sim, 0.0)

    return r_sim


# =============================================================================
# PRIX D'OBLIGATIONS ZÉRO-COUPON (FORMULE ANALYTIQUE VASICEK)
# =============================================================================

def prix_obligation_vasicek(
    tau: float,
    params: ParametresVasicek,
) -> float:
    """
    Calcule le prix d'une obligation zéro-coupon P(0, τ) selon Vasicek.

    Formule analytique (solution exacte) :
        P(0, τ) = A(τ) · exp(-B(τ) · r₀)

    où :
        B(τ) = (1 - exp(-κ·τ)) / κ
        A(τ) = exp[ (θ - σ²/(2κ²)) · (B(τ) - τ) - σ²/(4κ) · B²(τ) ]

    Cette formule dérive de la solution de l'EDP de Feynman-Kac associée
    au modèle de Vasicek (Hull, 2018, ch. 31).

    Parameters
    ----------
    tau : float or np.ndarray
        Maturité(s) en années.
    params : ParametresVasicek
        Paramètres du modèle.

    Returns
    -------
    float or np.ndarray
        Prix P(0, τ).
    """
    kappa, theta, sigma, r0 = params.kappa, params.theta, params.sigma, params.r0

    B = (1 - np.exp(-kappa * tau)) / kappa
    A = np.exp(
        (theta - sigma**2 / (2 * kappa**2)) * (B - tau)
        - (sigma**2 / (4 * kappa)) * B**2
    )
    return A * np.exp(-B * r0)


# =============================================================================
# SIMULATION DE SCÉNARIOS DE STRESS (TAUX)
# =============================================================================

def simuler_scenarios_stress_taux(
    params: ParametresVasicek,
    chocs_bp: Dict[str, float],
    n_simulations: int = SIM.n_simulations,
    n_periodes: int = None,
    dt: float = SIM.dt_mensuel,
    graine: int = SIM.graine_aleatoire,
) -> Dict[str, np.ndarray]:
    """
    Simule des scénarios de stress en modifiant le taux initial r₀.

    Chaque scénario part du même r₀ majoré d'un choc instantané (en pb),
    mais utilise exactement les mêmes chocs aléatoires. Cela permet d'isoler
    l'effet du niveau initial sur la distribution des coûts.

    Parameters
    ----------
    params : ParametresVasicek
        Paramètres de base.
    chocs_bp : dict
        Dictionnaire {nom_scenario: choc_en_decimale}.
        Ex. {"Base": 0.0, "+100bp": 0.01, "+200bp": 0.02}
    n_simulations, n_periodes, dt, graine : cf. `simuler_taux_vasicek`.

    Returns
    -------
    dict
        {nom_scenario: tableau (n_simulations, n_periodes+1)}
    """
    from src.config.settings import AGENT
    if n_periodes is None:
        n_periodes = AGENT.n_periodes_dette

    # Génération unique des chocs (identiques pour tous les scénarios)
    np.random.seed(graine)
    chocs = np.random.normal(0, 1, (n_simulations, n_periodes))

    resultats = {}
    for nom, choc in chocs_bp.items():
        r0_stress = params.r0 + choc
        r_stress = np.zeros((n_simulations, n_periodes + 1))
        r_stress[:, 0] = r0_stress

        for t in range(n_periodes):
            derive    = params.kappa * (params.theta - r_stress[:, t]) * dt
            diffusion = params.sigma * np.sqrt(dt) * chocs[:, t]
            r_stress[:, t + 1] = r_stress[:, t] + derive + diffusion

        resultats[nom] = np.maximum(r_stress, 0.0)

    return resultats
