"""
=============================================================================
metrics/risk.py — Métriques de risque quantitatives
=============================================================================
Calcule les métriques de gestion des risques pour l'évaluation des stratégies
de couverture (section 5.3.2.3) et du risque de contrepartie (section 5.3.3).

Métriques implémentées :
  1. VaR (Value-at-Risk) à 95% et 99%
  2. ES (Expected Shortfall / CVaR) à 99%
  3. η (réduction de variance) — efficacité de la couverture
  4. ΔVaR% — réduction relative de la VaR
  5. σ_CF — écart-type des cash flows (stabilité de trésorerie)
  6. Ratio de Sharpe de couverture — rapport rendement/risque
  7. Exposition positive attendue (EPE) — risque de contrepartie
  8. Exposure at Default (EAD) après netting et collatéral
=============================================================================
"""

import numpy as np
from dataclasses import dataclass
from scipy import stats
from typing import Dict, Tuple, Optional


# =============================================================================
# MÉTRIQUES DE BASE
# =============================================================================

def var_percentile(couts: np.ndarray, niveau: float = 0.99) -> float:
    """
    Calcule la Value-at-Risk (VaR) au niveau de confiance donné.

    VaR_α = -Quantile_{1-α}(coûts)

    Les coûts sont des sorties (négatifs), donc la VaR est positive.
    On prend l'opposé du quantile bas de la distribution des coûts.

    Parameters
    ----------
    couts : np.ndarray
        Distribution des coûts totaux simulés (valeurs négatives).
    niveau : float
        Niveau de confiance (0.99 = VaR 99%).

    Returns
    -------
    float
        VaR positive (montant de perte maximal au niveau considéré).
    """
    q = (1 - niveau) * 100   # percentile = 1% pour VaR 99%
    return float(-np.percentile(couts, q))


def expected_shortfall(couts: np.ndarray, niveau: float = 0.99) -> float:
    """
    Calcule l'Expected Shortfall (ES / CVaR) au niveau de confiance donné.

    ES_α = E[ -coût | coût < VaR_seuil ]
         = E[ perte | perte > VaR_α ]

    L'ES est la perte moyenne dans les (1-α)% pires scénarios.
    Elle est sous-additive (contrairement à la VaR) et cohérente au sens
    d'Artzner et al. (1999), ce qui en fait une mesure de risque supérieure.

    Parameters
    ----------
    couts : np.ndarray
        Distribution des coûts totaux simulés.
    niveau : float
        Niveau de confiance.

    Returns
    -------
    float
        Expected Shortfall positif.
    """
    seuil = np.percentile(couts, (1 - niveau) * 100)
    pertes_extremes = couts[couts <= seuil]
    return float(-pertes_extremes.mean())


def reduction_variance(couts_nc: np.ndarray, couts_couvert: np.ndarray) -> float:
    """
    Calcule le ratio de réduction de variance η (efficacité de couverture).

    η = 1 - Var(CF_couvert) / Var(CF_non_couvert)

    Interprétation :
      η = 1.0  → couverture parfaite (variance entièrement éliminée)
      η = 0.0  → couverture inefficace
      η < 0    → couverture aggrave le risque

    Parameters
    ----------
    couts_nc : np.ndarray
        Distribution des coûts non couverts.
    couts_couvert : np.ndarray
        Distribution des coûts après couverture.

    Returns
    -------
    float
        Ratio η ∈ [0, 1] (théoriquement).
    """
    var_nc = np.var(couts_nc)
    var_c  = np.var(couts_couvert)
    return float(1.0 - var_c / var_nc) if var_nc > 0 else 0.0


def delta_var_pct(var_avant: float, var_apres: float) -> float:
    """
    Calcule la réduction relative de VaR en pourcentage.

    ΔVaR% = (VaR_avant - VaR_après) / VaR_avant × 100

    Parameters
    ----------
    var_avant : float
        VaR avant couverture.
    var_apres : float
        VaR après couverture.

    Returns
    -------
    float
        Réduction en pourcentage (valeur positive = amélioration).
    """
    return float((var_avant - var_apres) / var_avant * 100) if var_avant > 0 else 0.0


def ratio_sharpe_couverture(
    couts_nc: np.ndarray,
    couts_couvert: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Calcule le ratio de Sharpe adapté à la couverture.

    Mesure l'efficience de la couverture : combien d'unités de réduction
    de coût moyen obtient-on par unité de risque (σ) réduit ?

    Sharpe_couv = ΔMoyenne / Δσ
                = (μ_nc - μ_c) / (σ_nc - σ_c)

    Un ratio élevé → couverture très efficiente.

    Parameters
    ----------
    couts_nc, couts_couvert : np.ndarray
        Distributions de coûts.
    epsilon : float
        Plancher pour éviter la division par zéro.

    Returns
    -------
    float
        Ratio de Sharpe de couverture.
    """
    delta_moy   = couts_nc.mean() - couts_couvert.mean()
    delta_sigma = np.std(couts_nc) - np.std(couts_couvert)
    return float(delta_moy / max(delta_sigma, epsilon))


# =============================================================================
# STRUCTURE DE RÉSULTATS CONSOLIDÉS
# =============================================================================

@dataclass
class MetriquesRisque:
    """Métriques de risque complètes pour une position (avant et après couverture)."""
    # ── Avant couverture ──────────────────────────────────────────────────────
    moyenne_nc: float
    sigma_nc: float
    var95_nc: float
    var99_nc: float
    es99_nc: float

    # ── Après couverture ──────────────────────────────────────────────────────
    moyenne_c: float
    sigma_c: float
    var95_c: float
    var99_c: float
    es99_c: float

    # ── Métriques d'efficacité ────────────────────────────────────────────────
    eta: float               # Réduction de variance
    delta_var99_pct: float   # Réduction de VaR 99% (%)
    delta_var95_pct: float   # Réduction de VaR 95% (%)
    sharpe_couverture: float # Ratio de Sharpe de couverture


def calculer_metriques_risque(
    couts_nc: np.ndarray,
    couts_couvert: np.ndarray,
) -> MetriquesRisque:
    """
    Calcule toutes les métriques de risque avant et après couverture.

    Parameters
    ----------
    couts_nc : np.ndarray, shape (M,)
        Coûts totaux non couverts.
    couts_couvert : np.ndarray, shape (M,)
        Coûts totaux après couverture.

    Returns
    -------
    MetriquesRisque
        Objet contenant toutes les métriques.
    """
    var95_nc = var_percentile(couts_nc, 0.95)
    var99_nc = var_percentile(couts_nc, 0.99)
    var95_c  = var_percentile(couts_couvert, 0.95)
    var99_c  = var_percentile(couts_couvert, 0.99)

    return MetriquesRisque(
        # Avant
        moyenne_nc = float(couts_nc.mean()),
        sigma_nc   = float(np.std(couts_nc)),
        var95_nc   = var95_nc,
        var99_nc   = var99_nc,
        es99_nc    = expected_shortfall(couts_nc, 0.99),
        # Après
        moyenne_c  = float(couts_couvert.mean()),
        sigma_c    = float(np.std(couts_couvert)),
        var95_c    = var95_c,
        var99_c    = var99_c,
        es99_c     = expected_shortfall(couts_couvert, 0.99),
        # Efficacité
        eta                = reduction_variance(couts_nc, couts_couvert),
        delta_var99_pct    = delta_var_pct(var99_nc, var99_c),
        delta_var95_pct    = delta_var_pct(var95_nc, var95_c),
        sharpe_couverture  = ratio_sharpe_couverture(couts_nc, couts_couvert),
    )


# =============================================================================
# TESTS STATISTIQUES
# =============================================================================

def test_jarque_bera(couts: np.ndarray) -> Tuple[float, float, str]:
    """
    Teste la normalité de la distribution des coûts (test de Jarque-Bera).

    H₀ : la distribution est gaussienne (skewness = 0, kurtosis = 3).

    Parameters
    ----------
    couts : np.ndarray
        Distribution des coûts.

    Returns
    -------
    tuple
        (statistique_JB, p_valeur, interpretation)
    """
    jb_stat, jb_pval = stats.jarque_bera(couts)
    interpretation = (
        "Distribution gaussienne (H₀ non rejetée, α=5%)"
        if jb_pval > 0.05
        else "Distribution non gaussienne (H₀ rejetée, α=5%)"
    )
    return float(jb_stat), float(jb_pval), interpretation


def test_ks_lognormal(s_sim: np.ndarray, mu_ln: float, sig_ln: float) -> Tuple[float, float, str]:
    """
    Vérifie que les rendements log-normaux du GBM suivent bien N(0,1) après standardisation.

    Parameters
    ----------
    s_sim : np.ndarray, shape (M, n_periodes+1)
        Trajectoires USD/MAD.
    mu_ln : float
        Moyenne théorique de ln(S_T/S₀).
    sig_ln : float
        Écart-type théorique.

    Returns
    -------
    tuple
        (stat_KS, p_valeur, interpretation)
    """
    s0 = s_sim[:, 0]
    log_ret = np.log(s_sim[:, -1] / s0)
    log_ret_std = (log_ret - mu_ln) / sig_ln

    ks_stat, ks_pval = stats.kstest(log_ret_std, "norm")
    interpretation = (
        "GBM validé : ln(S_T/S₀) ∼ N(μ, σ²) ✓"
        if ks_pval > 0.05
        else "⚠ Écart avec la loi log-normale attendue"
    )
    return float(ks_stat), float(ks_pval), interpretation


# =============================================================================
# SECTION 5.3.3 — RISQUE DE CONTREPARTIE
# =============================================================================

def calculer_exposition_positive_attendue(
    valeurs_mark_to_market: np.ndarray,
) -> np.ndarray:
    """
    Calcule l'Exposition Positive Attendue (EPE) à chaque période.

    EPE_t = E[max(V_t, 0)]

    Représente la perte potentielle si la contrepartie fait défaut
    à la date t et que le contrat a une valeur positive (in-the-money).

    Parameters
    ----------
    valeurs_mark_to_market : np.ndarray, shape (M, n_periodes)
        Valeurs mark-to-market du dérivé à chaque période.

    Returns
    -------
    np.ndarray, shape (n_periodes,)
        EPE à chaque pas de temps.
    """
    # Seules les valeurs positives créent un risque de contrepartie
    expositions_positives = np.maximum(valeurs_mark_to_market, 0)
    return expositions_positives.mean(axis=0)


def calculer_exposition_brute(
    flux_swap: np.ndarray,
    flux_forward: np.ndarray,
) -> np.ndarray:
    """
    Calcule l'exposition brute agrégée (IRS + Forward) par simulation.

    Exposition brute = somme des valeurs positives avant tout accord de netting.
    Correspond à la perte maximale théorique si chaque contrat est
    traité séparément (sans accord de compensation).

    Parameters
    ----------
    flux_swap : np.ndarray, shape (M, n_taux)
        Flux nets du swap de taux.
    flux_forward : np.ndarray, shape (M, n_fx)
        Gains/pertes sur le forward de change.

    Returns
    -------
    np.ndarray, shape (M,)
        Exposition brute cumulée par simulation.
    """
    # Somme des flux positifs du swap (in-the-money pour l'entreprise)
    exposition_irs = np.maximum(flux_swap, 0).sum(axis=1)

    # Somme des flux positifs du forward
    exposition_fwd = np.maximum(flux_forward, 0).sum(axis=1)

    return exposition_irs + exposition_fwd


def calculer_exposition_nette(
    flux_swap: np.ndarray,
    flux_forward: np.ndarray,
) -> np.ndarray:
    """
    Calcule l'exposition nette après accord de netting (ISDA Master Agreement).

    Le netting bilatéral permet de compenser les gains et pertes sur
    l'ensemble du portefeuille avec une même contrepartie :
        V_net_t = V_IRS_t + V_FWD_t  (compensation algébrique)
        Exposition_nette = max(V_net_t, 0)

    Cela réduit significativement l'exposition brute (effet de compensation).

    Parameters
    ----------
    flux_swap : np.ndarray, shape (M, n_taux)
        Flux du swap.
    flux_forward : np.ndarray, shape (M, n_fx)
        Flux du forward.

    Returns
    -------
    np.ndarray, shape (M,)
        Exposition nette cumulée après netting.
    """
    # Le swap a 36 périodes et le forward 12 → on aligne sur la longueur commune
    n_commun = min(flux_swap.shape[1], flux_forward.shape[1])

    flux_net = flux_swap[:, :n_commun] + flux_forward[:, :n_commun]
    exposition_nette = np.maximum(flux_net, 0).sum(axis=1)

    return exposition_nette


def calculer_ead_avec_collateral(
    exposition_brute: np.ndarray,
    exposition_nette: np.ndarray,
    taux_collateral: float = 0.80,
) -> Dict[str, np.ndarray]:
    """
    Calcule l'Exposure at Default (EAD) avec et sans collatéral.

    Le collatéral (sûreté) réduit l'exposition résiduelle :
        EAD = max(Exposition_nette × (1 - taux_collateral), 0)

    Mécanismes de réduction d'exposition :
      1. Netting bilatéral (ISDA) : réduit l'exposition brute → nette
      2. Collatéralisation (CSA)  : réduit l'exposition nette → EAD

    Parameters
    ----------
    exposition_brute : np.ndarray, shape (M,)
        Exposition avant netting.
    exposition_nette : np.ndarray, shape (M,)
        Exposition après netting.
    taux_collateral : float
        Fraction couverte par le collatéral (0.80 = 80%).

    Returns
    -------
    dict
        {
            'ead_sans_collateral': np.ndarray,  # = exposition nette
            'ead_avec_collateral': np.ndarray,  # après application du collatéral
            'reduction_netting_pct': float,     # réduction due au netting (%)
            'reduction_collateral_pct': float,  # réduction due au collatéral (%)
        }
    """
    ead_sans_collateral = exposition_nette
    ead_avec_collateral = np.maximum(exposition_nette * (1 - taux_collateral), 0)

    moy_brute = exposition_brute.mean()
    moy_nette = exposition_nette.mean()
    moy_ead   = ead_avec_collateral.mean()

    reduction_netting_pct = float(
        (moy_brute - moy_nette) / moy_brute * 100
    ) if moy_brute > 0 else 0.0

    reduction_collateral_pct = float(
        (moy_nette - moy_ead) / moy_nette * 100
    ) if moy_nette > 0 else 0.0

    return {
        "ead_sans_collateral"    : ead_sans_collateral,
        "ead_avec_collateral"    : ead_avec_collateral,
        "reduction_netting_pct"  : reduction_netting_pct,
        "reduction_collateral_pct": reduction_collateral_pct,
    }


def calculer_credit_value_adjustment(
    exposition_positive_attendue: np.ndarray,
    probabilite_defaut: float,
    taux_recouvrement: float,
    dt: float,
) -> float:
    """
    Calcule le Credit Value Adjustment (CVA) simplifié.

    Le CVA représente la valeur attendue de la perte due au risque
    de défaut de la contrepartie :

        CVA ≈ (1 - R) × PD × Σ EPE_t × DF_t

    où R = taux de recouvrement, PD = probabilité de défaut,
    EPE_t = exposition positive attendue à t, DF_t = facteur d'actualisation.

    Note : il s'agit d'une approximation mono-période.

    Parameters
    ----------
    exposition_positive_attendue : np.ndarray
        EPE à chaque période.
    probabilite_defaut : float
        Probabilité de défaut annuelle de la contrepartie.
    taux_recouvrement : float
        Fraction de la valeur récupérée en cas de défaut (LGD = 1 - R).
    dt : float
        Pas de temps (pour actualiser).

    Returns
    -------
    float
        CVA estimé.
    """
    lgd = 1 - taux_recouvrement
    n   = len(exposition_positive_attendue)

    # Probabilité de défaut sur chaque sous-période
    pd_periode = 1 - (1 - probabilite_defaut) ** dt

    # Facteurs d'actualisation approximatifs (taux plat supposé)
    periodes = np.arange(1, n + 1)
    df = np.exp(-0.03 * periodes * dt)   # 3% : taux d'actualisation sans risque

    cva = lgd * pd_periode * np.sum(exposition_positive_attendue * df)
    return float(cva)
