"""
=============================================================================
analytics/regulatory.py — Section 5.3.4 : Scénario réglementaire comparatif
=============================================================================
Compare deux cadres réglementaires pour le marché dérivé OTC marocain :

  CADRE A — Restrictif (hedging uniquement)
    - Peu de contrats, forte corrélation, collatéral partiel
    - Pas de reconnaissance légale du netting ISDA
    - Spreads élevés, une seule contrepartie

  CADRE B — Flexible et prudentiel
    - Portefeuille diversifié, faible corrélation
    - Netting ISDA reconnu + CSA obligatoire
    - Supervision Bâle III / EMIR-like, spreads réduits

Toutes les simulations réutilisent les flux IRS et Forward déjà calculés
dans les sections 5.3.2 et 5.3.3 (cohérence du modèle).
=============================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from src.config.settings import SIM


# =============================================================================
# PARAMÈTRES DES DEUX CADRES RÉGLEMENTAIRES
# =============================================================================

@dataclass(frozen=True)
class CadreReglementaire:
    """
    Paramètres définissant un cadre réglementaire OTC.

    Attributs
    ---------
    nom : str
        Nom du cadre ("Restrictif" ou "Flexible").
    n_contrats : int
        Nombre de contrats OTC dans le portefeuille.
    correlation : float
        Corrélation moyenne entre les MTM des contrats (ρ).
    taux_collateral : float
        Fraction de l'exposition couverte par collatéral.
    netting_reconnu : bool
        Reconnaissance légale du close-out netting ISDA.
    spread_bp : float
        Spread bid-ask en points de base.
    n_contreparties : int
        Nombre de contreparties bancaires distinctes.
    pd_annuelle : float
        Probabilité de défaut annuelle de la contrepartie.
    taux_recouvrement : float
        Taux de recouvrement en cas de défaut (LGD = 1 - R).
    """
    nom: str
    n_contrats: int
    correlation: float
    taux_collateral: float
    netting_reconnu: bool
    spread_bp: float
    n_contreparties: int
    pd_annuelle: float
    taux_recouvrement: float
    couleur: str = "#FF8A65"
    label: str = ""


# Instances des deux cadres
CADRE_RESTRICTIF = CadreReglementaire(
    nom               = "Cadre Restrictif",
    n_contrats        = 2,
    correlation       = 0.80,
    taux_collateral   = 0.30,
    netting_reconnu   = False,
    spread_bp         = 50.0,
    n_contreparties   = 1,
    pd_annuelle       = 0.03,
    taux_recouvrement = 0.30,
     couleur           = "#FF8A65",
    label             = "Restrictif",
)

CADRE_FLEXIBLE = CadreReglementaire(
    nom               = "Cadre Flexible",
    n_contrats        = 10,
    correlation       = 0.20,
    taux_collateral   = 0.80,
    netting_reconnu   = True,
    spread_bp         = 15.0,
    n_contreparties   = 4,
    pd_annuelle       = 0.015,
    taux_recouvrement = 0.50,
    couleur           = "#66BB6A",
    label             = "Flexible",
)


# =============================================================================
# STRUCTURE DE RÉSULTATS
# =============================================================================

@dataclass
class ResultatsCadre:
    """Résultats complets de simulation pour un cadre réglementaire."""
    cadre: CadreReglementaire

    # Expositions
    ebe_moyen: float          # Exposition Brute Espérée (moyenne)
    ene_moyen: float          # Exposition Nette Espérée (moyenne)
    netting_factor: float     # ENE / EBE
    ead_moyen: float          # Exposition After Default (après collatéral)

    # CVA
    cva_total: float          # Credit Value Adjustment total

    # Coûts
    cout_spread_total: float  # Coût total des spreads sur toute la durée

    # Stabilité
    hhi: float                # Indice de Herfindahl-Hirschman (concentration)
    ratio_diversification: float  # 1 - HHI normalisé

    # Distributions (pour les figures)
    dist_ebe: np.ndarray      # Distribution Monte Carlo de l'EBE
    dist_ene: np.ndarray      # Distribution Monte Carlo de l'ENE
    dist_ead: np.ndarray      # Distribution Monte Carlo de l'EAD
    epe_profil: np.ndarray    # Profil EPE par période


# =============================================================================
# SIMULATION MONTE CARLO DU PORTEFEUILLE DE CONTRATS
# =============================================================================

def simuler_portefeuille_contrats(
    flux_irs: np.ndarray,
    flux_fwd: np.ndarray,
    cadre: CadreReglementaire,
    n_simulations: int = SIM.n_simulations,
    graine: int = SIM.graine_aleatoire + 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simule un portefeuille de n_contrats OTC en répliquant et perturbant
    les flux IRS et Forward déjà calculés.

    Méthode de Cholesky pour introduire la corrélation ρ entre contrats :
      - On génère n_contrats séries de flux corrélées à partir des flux de base
      - La matrice de corrélation est Σ_ij = ρ si i≠j, 1 si i=j
      - On décompose par Cholesky : L tel que L·Lᵀ = Σ
      - Les chocs corrélés : ε_corr = L · ε_indep

    Parameters
    ----------
    flux_irs : np.ndarray, shape (M, n_periodes_irs)
        Flux nets IRS simulés (section 5.3.2).
    flux_fwd : np.ndarray, shape (M, n_periodes_fwd)
        Gains/pertes Forward simulés (section 5.3.2).
    cadre : CadreReglementaire
        Paramètres du cadre réglementaire.
    n_simulations : int
        Nombre de trajectoires Monte Carlo.
    graine : int
        Graine pour la reproductibilité.

    Returns
    -------
    tuple
        (mtm_portefeuille, mtm_par_contrat)
        - mtm_portefeuille : (M, n_periodes) — valeur nette du portefeuille
        - mtm_par_contrat  : (M, n_contrats, n_periodes) — par contrat
    """
    np.random.seed(graine)

    n_contrats = cadre.n_contrats
    rho        = cadre.correlation

    # Aligner les deux flux sur la période commune (12 mois)
    n_periodes = min(flux_irs.shape[1], flux_fwd.shape[1])
    flux_base  = flux_irs[:, :n_periodes]  # Flux de référence

    M = flux_base.shape[0]

    # ── Matrice de corrélation (structure équicorrélée) ───────────────────────
    Sigma = np.full((n_contrats, n_contrats), rho)
    np.fill_diagonal(Sigma, 1.0)

    # Décomposition de Cholesky
    # Régularisation pour garantir la définition positive
    # ρ = 0 ou ρ = 1 rendent la matrice singulière → on clip légèrement
    rho_clip = np.clip(rho, 0.001, 0.999)
    Sigma = np.full((n_contrats, n_contrats), rho_clip)
    np.fill_diagonal(Sigma, 1.0)

    # Régularisation numérique supplémentaire (epsilon sur la diagonale)
    Sigma += np.eye(n_contrats) * 1e-8

    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # Fallback : corrélation nulle (contrats indépendants)
        L = np.eye(n_contrats)

    # ── Génération des chocs corrélés ─────────────────────────────────────────
    # Chocs indépendants : (M, n_contrats, n_periodes)
    eps_indep = np.random.normal(0, 1, (M, n_contrats, n_periodes))

    # Application de la corrélation via Cholesky
    # Pour chaque simulation m et période t : ε_corr = L · ε_indep
    eps_corr = np.einsum('ij,mjt->mit', L, eps_indep)

    # ── Construction des MTM par contrat ─────────────────────────────────────
    # Chaque contrat = flux de base × (1 + bruit relatif calibré)
    vol_relative = 0.15   # 15% de bruit autour du flux de base
    mtm_par_contrat = np.zeros((M, n_contrats, n_periodes))

    for k in range(n_contrats):
        # Alternance IRS / Forward selon la parité
        if k % 2 == 0:
            base = flux_irs[:n_simulations, :n_periodes]
        else:
            base = flux_fwd[:n_simulations, :n_periodes]

        # Perturbation corrélée autour du flux de base
        mtm_par_contrat[:, k, :] = base * (1 + vol_relative * eps_corr[:, k, :])

    # MTM total du portefeuille (somme algébrique)
    mtm_portefeuille = mtm_par_contrat.sum(axis=1)

    return mtm_portefeuille, mtm_par_contrat


# =============================================================================
# CALCUL DES EXPOSITIONS DU PORTEFEUILLE
# =============================================================================

def calculer_exposition_portefeuille(
    mtm_par_contrat: np.ndarray,
    cadre: CadreReglementaire,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule EBE, ENE et EAD pour le portefeuille selon le cadre réglementaire.

    SANS netting (cadre restrictif) :
        EBE_t = Σ_k max(V_k_t, 0)   — somme des expositions positives par contrat
        ENE_t = EBE_t                — pas de compensation

    AVEC netting ISDA (cadre flexible) :
        ENE_t = max(Σ_k V_k_t, 0)   — compensation algébrique, puis floor à 0
        EBE_t reste calculée contrat par contrat (référence)

    Netting Factor = ENE / EBE  (plus c'est bas, plus le netting est efficace)

    EAD = max(ENE × (1 - taux_collateral), 0)

    Parameters
    ----------
    mtm_par_contrat : np.ndarray, shape (M, n_contrats, n_periodes)
        MTM de chaque contrat à chaque période.
    cadre : CadreReglementaire
        Détermine si le netting est reconnu.

    Returns
    -------
    tuple
        (dist_ebe, dist_ene, dist_ead) — distributions sur M simulations (M,)
    """
    # EBE : somme des valeurs positives contrat par contrat, cumulée
    exposition_positive_par_contrat = np.maximum(mtm_par_contrat, 0)
    ebe_cumule = exposition_positive_par_contrat.sum(axis=(1, 2))   # (M,)

    if cadre.netting_reconnu:
        # Netting ISDA : compensation algébrique au niveau du portefeuille
        mtm_net = mtm_par_contrat.sum(axis=1)                        # (M, n_periodes)
        ene_cumule = np.maximum(mtm_net, 0).sum(axis=1)              # (M,)
    else:
        # Sans netting : ENE = EBE
        ene_cumule = ebe_cumule.copy()

    # EAD après collatéral
    ead_cumule = np.maximum(ene_cumule * (1 - cadre.taux_collateral), 0)

    return ebe_cumule, ene_cumule, ead_cumule


# =============================================================================
# CVA DU PORTEFEUILLE
# =============================================================================

def calculer_cva_portefeuille(
    mtm_par_contrat: np.ndarray,
    cadre: CadreReglementaire,
    dt: float = SIM.dt_mensuel,
) -> float:
    """
    Calcule le CVA total du portefeuille.

    CVA = (1 - R) × PD_periode × Σ_t EPE_t × DF_t

    Avec netting, l'EPE est calculée sur le portefeuille net.
    Sans netting, l'EPE est calculée contrat par contrat (plus élevée).

    Parameters
    ----------
    mtm_par_contrat : np.ndarray, shape (M, n_contrats, n_periodes)
    cadre : CadreReglementaire

    Returns
    -------
    float
        CVA total (MAD).
    """
    n_periodes = mtm_par_contrat.shape[2]
    lgd = 1 - cadre.taux_recouvrement
    pd_periode = 1 - (1 - cadre.pd_annuelle) ** dt

    periodes = np.arange(1, n_periodes + 1)
    df = np.exp(-0.03 * periodes * dt)

    if cadre.netting_reconnu:
        # EPE sur portefeuille net
        mtm_net = mtm_par_contrat.sum(axis=1)                    # (M, n_periodes)
        epe = np.maximum(mtm_net, 0).mean(axis=0)                # (n_periodes,)
    else:
        # EPE somme des expositions positives par contrat
        epe = np.maximum(mtm_par_contrat, 0).mean(axis=0).sum(axis=0)  # (n_periodes,)

    cva = lgd * pd_periode * np.sum(epe * df)
    return float(cva)


# =============================================================================
# COÛT DES SPREADS
# =============================================================================

def calculer_cout_spread(
    notionnel: float,
    cadre: CadreReglementaire,
    n_periodes: int,
    dt: float = SIM.dt_mensuel,
) -> float:
    """
    Calcule le coût total des spreads bid-ask sur la durée du portefeuille.

    Coût_spread = N × (spread_bp / 10000) × Δt × n_periodes × n_contrats

    Un spread élevé (cadre restrictif) représente un coût de transaction
    significatif qui s'ajoute aux charges financières de couverture.

    Parameters
    ----------
    notionnel : float
        Notionnel moyen par contrat (MAD).
    cadre : CadreReglementaire
    n_periodes : int
    dt : float

    Returns
    -------
    float
        Coût total des spreads (MAD).
    """
    spread_decimal = cadre.spread_bp / 10000
    return notionnel * spread_decimal * dt * n_periodes * cadre.n_contrats


# =============================================================================
# INDICE DE HERFINDAHL-HIRSCHMAN (CONCENTRATION DU RISQUE)
# =============================================================================

def calculer_hhi(
    mtm_par_contrat: np.ndarray,
    cadre: CadreReglementaire,
) -> Tuple[float, float]:
    """
    Calcule l'indice de Herfindahl-Hirschman (HHI) de concentration du risque.

    HHI = Σ_k (s_k)²   où s_k = exposition_k / exposition_totale

    HHI = 1    → risque concentré sur un seul contrat / contrepartie
    HHI = 1/n  → risque parfaitement diversifié entre n contrats

    Le ratio de diversification = 1 - HHI_normalisé
    HHI_normalisé = (HHI - 1/n) / (1 - 1/n)

    Parameters
    ----------
    mtm_par_contrat : np.ndarray, shape (M, n_contrats, n_periodes)
    cadre : CadreReglementaire

    Returns
    -------
    tuple
        (hhi, ratio_diversification)
    """
    n = cadre.n_contrats

    # Exposition moyenne par contrat (valeurs positives seulement)
    expo_par_contrat = np.maximum(mtm_par_contrat, 0).mean(axis=(0, 2))  # (n_contrats,)
    expo_totale = expo_par_contrat.sum()

    if expo_totale < 1e-10:
        return 1.0 / n, 1.0

    parts = expo_par_contrat / expo_totale
    hhi   = float(np.sum(parts**2))

    # HHI normalisé entre 0 (parfait) et 1 (monopole)
    hhi_min = 1.0 / n
    hhi_norm = (hhi - hhi_min) / (1 - hhi_min) if n > 1 else 1.0
    ratio_div = 1.0 - hhi_norm

    return hhi, ratio_div


# =============================================================================
# PROFIL EPE PAR PÉRIODE
# =============================================================================

def calculer_profil_epe(
    mtm_par_contrat: np.ndarray,
    cadre: CadreReglementaire,
) -> np.ndarray:
    """
    Calcule le profil d'Exposition Positive Attendue (EPE) par période.

    EPE_t = E[max(V_net_t, 0)]  avec ou sans netting selon le cadre.

    Parameters
    ----------
    mtm_par_contrat : np.ndarray, shape (M, n_contrats, n_periodes)
    cadre : CadreReglementaire

    Returns
    -------
    np.ndarray, shape (n_periodes,)
        EPE à chaque pas de temps.
    """
    if cadre.netting_reconnu:
        mtm_net = mtm_par_contrat.sum(axis=1)           # (M, n_periodes)
        epe = np.maximum(mtm_net, 0).mean(axis=0)
    else:
        epe = np.maximum(mtm_par_contrat, 0).mean(axis=0).sum(axis=0)

    return epe


# =============================================================================
# ANALYSE DE SENSIBILITÉ
# =============================================================================

def analyse_sensibilite_netting(
    flux_irs: np.ndarray,
    flux_fwd: np.ndarray,
    valeurs_n: List[int] = None,
    valeurs_rho: List[float] = None,
    valeurs_collateral: List[float] = None,
    n_simulations: int = SIM.n_simulations,
) -> Dict[str, np.ndarray]:
    """
    Analyse de sensibilité du Netting Factor et de l'EAD aux paramètres clés.

    Fait varier indépendamment :
      1. n (nombre de contrats) : plus n est élevé, plus le netting est efficace
      2. ρ (corrélation)       : plus ρ est faible, plus la diversification joue
      3. taux de collatéral    : impact direct sur l'EAD résiduelle

    Parameters
    ----------
    flux_irs, flux_fwd : np.ndarray
        Flux de base pour la simulation.
    valeurs_n : list
        Valeurs de n à tester.
    valeurs_rho : list
        Valeurs de ρ à tester.
    valeurs_collateral : list
        Taux de collatéral à tester.
    n_simulations : int

    Returns
    -------
    dict
        {
            'nf_vs_n'         : Netting Factor en fonction de n,
            'nf_vs_rho'       : Netting Factor en fonction de ρ,
            'ead_vs_collat'   : EAD moyenne en fonction du collatéral,
            'valeurs_n'       : valeurs de n testées,
            'valeurs_rho'     : valeurs de ρ testées,
            'valeurs_collat'  : valeurs de collatéral testées,
        }
    """
    if valeurs_n is None:
        valeurs_n = [1, 2, 3, 5, 8, 10, 15, 20]
    if valeurs_rho is None:
        valeurs_rho = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99]

    if valeurs_collateral is None:
        valeurs_collateral = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    nf_vs_n   = np.zeros(len(valeurs_n))
    nf_vs_rho = np.zeros(len(valeurs_rho))
    ead_vs_collat = np.zeros(len(valeurs_collateral))

    n_periodes = min(flux_irs.shape[1], flux_fwd.shape[1])

    # ── Sensibilité à n (avec netting, ρ = 0.3 fixé) ─────────────────────────
    for i, n in enumerate(valeurs_n):
        cadre_test = CadreReglementaire(
            nom="test", n_contrats=n, correlation=0.30,
            taux_collateral=0.80, netting_reconnu=True,
            spread_bp=15, n_contreparties=4,
            pd_annuelle=0.015, taux_recouvrement=0.50,
        )
        _, mtm_c = simuler_portefeuille_contrats(flux_irs, flux_fwd, cadre_test, n_simulations)
        ebe, ene, _ = calculer_exposition_portefeuille(mtm_c, cadre_test)
        nf_vs_n[i] = ene.mean() / ebe.mean() if ebe.mean() > 0 else 0.0

    # ── Sensibilité à ρ (n = 10 fixé, avec netting) ──────────────────────────
    for i, rho in enumerate(valeurs_rho):
        cadre_test = CadreReglementaire(
            nom="test", n_contrats=10, correlation=rho,
            taux_collateral=0.80, netting_reconnu=True,
            spread_bp=15, n_contreparties=4,
            pd_annuelle=0.015, taux_recouvrement=0.50,
        )
        _, mtm_c = simuler_portefeuille_contrats(flux_irs, flux_fwd, cadre_test, n_simulations)
        ebe, ene, _ = calculer_exposition_portefeuille(mtm_c, cadre_test)
        nf_vs_rho[i] = ene.mean() / ebe.mean() if ebe.mean() > 0 else 0.0

    # ── Sensibilité au collatéral (n=10, ρ=0.2, avec netting) ────────────────
    cadre_base = CadreReglementaire(
        nom="test", n_contrats=10, correlation=0.20,
        taux_collateral=0.80, netting_reconnu=True,
        spread_bp=15, n_contreparties=4,
        pd_annuelle=0.015, taux_recouvrement=0.50,
    )
    _, mtm_base = simuler_portefeuille_contrats(flux_irs, flux_fwd, cadre_base, n_simulations)
    _, ene_base, _ = calculer_exposition_portefeuille(mtm_base, cadre_base)

    for i, tc in enumerate(valeurs_collateral):
        ead_vs_collat[i] = np.maximum(ene_base * (1 - tc), 0).mean()

    return {
        "nf_vs_n"        : nf_vs_n,
        "nf_vs_rho"      : nf_vs_rho,
        "ead_vs_collat"  : ead_vs_collat,
        "valeurs_n"      : np.array(valeurs_n),
        "valeurs_rho"    : np.array(valeurs_rho),
        "valeurs_collat" : np.array(valeurs_collateral),
    }


# =============================================================================
# PIPELINE PRINCIPAL : SIMULER ET COMPARER LES DEUX CADRES
# =============================================================================

def comparer_cadres_reglementaires(
    flux_irs: np.ndarray,
    flux_fwd: np.ndarray,
    notionnel: float = 100_000_000,
    n_simulations: int = SIM.n_simulations,
) -> Tuple[ResultatsCadre, ResultatsCadre]:
    """
    Lance la simulation complète pour les deux cadres et retourne les résultats.

    Parameters
    ----------
    flux_irs : np.ndarray, shape (M, n_periodes_irs)
        Flux nets IRS (depuis appliquer_couverture_irs).
    flux_fwd : np.ndarray, shape (M, n_periodes_fwd)
        Gains Forward (depuis appliquer_couverture_forward).
    notionnel : float
        Notionnel de référence pour le calcul des spreads.
    n_simulations : int

    Returns
    -------
    tuple
        (resultats_restrictif, resultats_flexible)
    """
    n_periodes = min(flux_irs.shape[1], flux_fwd.shape[1])
    resultats = []

    for cadre in [CADRE_RESTRICTIF, CADRE_FLEXIBLE]:
        # Simulation du portefeuille
        _, mtm_c = simuler_portefeuille_contrats(
            flux_irs, flux_fwd, cadre, n_simulations
        )

        # Expositions
        ebe, ene, ead = calculer_exposition_portefeuille(mtm_c, cadre)
        nf = float(ene.mean() / ebe.mean()) if ebe.mean() > 0 else 1.0

        # CVA
        cva = calculer_cva_portefeuille(mtm_c, cadre)

        # Spread
        cout_spread = calculer_cout_spread(notionnel, cadre, n_periodes)

        # HHI
        hhi, ratio_div = calculer_hhi(mtm_c, cadre)

        # Profil EPE
        epe_profil = calculer_profil_epe(mtm_c, cadre)

        resultats.append(ResultatsCadre(
            cadre               = cadre,
            ebe_moyen           = float(ebe.mean()),
            ene_moyen           = float(ene.mean()),
            netting_factor      = nf,
            ead_moyen           = float(ead.mean()),
            cva_total           = cva,
            cout_spread_total   = cout_spread,
            hhi                 = hhi,
            ratio_diversification = ratio_div,
            dist_ebe            = ebe,
            dist_ene            = ene,
            dist_ead            = ead,
            epe_profil          = epe_profil,
        ))

    return resultats[0], resultats[1]


# =============================================================================
# AFFICHAGE CONSOLE
# =============================================================================

def afficher_comparaison_cadres(
    res_a: ResultatsCadre,
    res_b: ResultatsCadre,
) -> None:
    """Affiche un tableau comparatif des deux cadres dans la console."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  {'COMPARAISON RÉGLEMENTAIRE':^68}")
    print(sep)
    print(f"  {'Indicateur':<35} {'Restrictif':>14} {'Flexible':>14}")
    print(sep)

    def ligne(nom, va, vb, fmt=".4f", suffixe=""):
        print(f"  {nom:<35} {va:>14{fmt}} {vb:>14{fmt}}{suffixe}")

    ligne("Nombre de contrats",
          res_a.cadre.n_contrats, res_b.cadre.n_contrats, fmt=".0f")
    ligne("Corrélation ρ",
          res_a.cadre.correlation, res_b.cadre.correlation)
    ligne("Taux de collatéral",
          res_a.cadre.taux_collateral, res_b.cadre.taux_collateral)
    ligne("Spread (bp)",
          res_a.cadre.spread_bp, res_b.cadre.spread_bp, fmt=".1f")
    print(f"  {'─'*68}")
    ligne("EBE moyen (M MAD)",
          res_a.ebe_moyen/1e6, res_b.ebe_moyen/1e6)
    ligne("ENE moyen (M MAD)",
          res_a.ene_moyen/1e6, res_b.ene_moyen/1e6)
    ligne("Netting Factor",
          res_a.netting_factor, res_b.netting_factor)
    ligne("EAD moyen (M MAD)",
          res_a.ead_moyen/1e6, res_b.ead_moyen/1e6)
    ligne("CVA total (K MAD)",
          res_a.cva_total/1e3, res_b.cva_total/1e3)
    ligne("Coût spreads (K MAD)",
          res_a.cout_spread_total/1e3, res_b.cout_spread_total/1e3)
    print(f"  {'─'*68}")
    ligne("HHI (concentration)",
          res_a.hhi, res_b.hhi)
    ligne("Ratio diversification",
          res_a.ratio_diversification, res_b.ratio_diversification)
    print(sep)

    # Réductions
    red_ead = (res_a.ead_moyen - res_b.ead_moyen) / res_a.ead_moyen * 100
    red_cva = (res_a.cva_total - res_b.cva_total) / res_a.cva_total * 100
    red_nf  = (res_a.netting_factor - res_b.netting_factor) / res_a.netting_factor * 100
    print(f"\n  Réduction EAD  (Flex vs Restr) : {red_ead:+.2f}%")
    print(f"  Réduction CVA  (Flex vs Restr) : {red_cva:+.2f}%")
    print(f"  Réduction NF   (Flex vs Restr) : {red_nf:+.2f}%")
    print(sep)

# =============================================================================
# NETTING FACTOR ANALYTIQUE
# =============================================================================

def netting_factor_analytique(
    n: int,
    rho: float,
) -> float:
    """
    Calcule le Netting Factor analytique selon la formule de Gibson (2002).

    NF(n, ρ) = sqrt[ (1 + (n-1)·ρ) / n ]

    Propriétés :
      - Si ρ = 1  → NF = 1     (pas de bénéfice, contrats parfaitement corrélés)
      - Si ρ = 0  → NF = 1/√n  (bénéfice maximum, contrats indépendants)
      - Si n = 1  → NF = 1     (un seul contrat, pas de netting possible)

    Parameters
    ----------
    n : int
        Nombre de contrats dans le portefeuille.
    rho : float
        Corrélation moyenne entre les MTM des contrats.

    Returns
    -------
    float
        Netting Factor entre 0 et 1.
    """
    return float(np.sqrt((1 + (n - 1) * rho) / n))

def matrice_netting_factors(
    valeurs_n: list = None,
    valeurs_rho: list = None,
) -> np.ndarray:
    """
    Calcule une matrice de Netting Factors pour différentes combinaisons
    de n (nombre de contrats) et ρ (corrélation).

    Utilise la formule analytique de Gibson (2002) :
        NF(n, ρ) = sqrt[ (1 + (n-1)·ρ) / n ]

    Parameters
    ----------
    valeurs_n : list
        Valeurs de n à tester (défaut : [1, 2, 5, 10, 20]).
    valeurs_rho : list
        Valeurs de ρ à tester (défaut : [0.0, 0.2, 0.5, 0.8, 1.0]).

    Returns
    -------
    np.ndarray, shape (len(valeurs_n), len(valeurs_rho))
        Matrice des Netting Factors.
    """
    if valeurs_n is None:
        valeurs_n = [1, 2, 5, 10, 20]
    if valeurs_rho is None:
        valeurs_rho = [0.0, 0.2, 0.5, 0.8, 1.0]

    matrice = np.zeros((len(valeurs_n), len(valeurs_rho)))
    for i, n in enumerate(valeurs_n):
        for j, rho in enumerate(valeurs_rho):
            matrice[i, j] = netting_factor_analytique(n, rho)

    return matrice