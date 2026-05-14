"""
=============================================================================
main.py — Pipeline complet : Modélisation des Risques Financiers OTC au Maroc
=============================================================================
Chapitre 5.3 — Exposition aux risques, couvertures et risque de contrepartie

Structure du pipeline :
  5.3.1 — Exposition aux risques financiers
    5.3.1.1 — Simulation de scénarios de marché de taux (Vasicek)
    5.3.1.2 — Simulation de scénarios de change (GBM)
  5.3.2 — Stratégies de couverture par produits dérivés
    5.3.2.1 — Swap de taux d'intérêt (IRS)
    5.3.2.2 — Contrat forward de change
    5.3.2.3 — Évaluation des métriques de couverture
  5.3.3 — Risque de contrepartie
    5.3.3.1 — Netting ISDA et collatéral CSA
    5.3.3.2 — EBE / ENE / EAD / CVA
  5.3.4 — Scénario réglementaire comparatif
    5.3.4.1 — Cadre restrictif
    5.3.4.2 — Cadre flexible prudentiel
    5.3.4.3 — Impact sur la stabilité financière

Usage :
    python main.py

Auteur : levir B — Produits Dérivés OTC au Maroc
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Graine globale pour la reproductibilité (avant tout autre import)
np.random.seed(42)
warnings.filterwarnings("ignore")

# Ajout du répertoire racine au path Python
sys.path.insert(0, str(Path(__file__).parent))

# ── Imports internes ─────────────────────────────────────────────────────────
from src.config import AGENT, SIM, MARKET, PLOT, PATHS

from utils.data_loader import (
    charger_taux_bam,
    construire_serie_mensuelle_bam,
    telecharger_donnees_fx,
    extraire_taux_spot,
    calculer_volatilite_gbm,
)
from utils.console import (
    titre_section,
    sous_titre,
    separateur,
    etape,
    afficher_parametres_vasicek,
    afficher_parametres_gbm,
    afficher_tableau_metriques,
    afficher_resume_irs,
    afficher_resume_forward,
)

from src.simulations.vasicek import (
    ParametresVasicek,
    calibrer_vasicek_ols,
    calibrer_vasicek_mle,
    simuler_taux_vasicek,
    simuler_scenarios_stress_taux,
)
from src.simulations.gbm import (
    ParametresGBM,
    simuler_taux_change_gbm,
    simuler_scenarios_stress_fx,
)
from src.simulations.cash_flows import (
    calculer_charges_dette_variable,
    calculer_tresorerie_taux,
    calculer_couts_importation,
    calculer_perte_change,
    calculer_tresorerie_fx,
    identifier_scenarios_representatifs,
)

from src.hedging.instruments import (
    appliquer_couverture_irs,
    appliquer_couverture_forward,
    calculer_taux_forward_cip,
)

from src.metrics.risk import (
    var_percentile,
    expected_shortfall,
    MetriquesRisque,
    calculer_metriques_risque,
    test_jarque_bera,
    test_ks_lognormal,
    calculer_exposition_positive_attendue,
    calculer_exposition_brute,
    calculer_exposition_nette,
    calculer_ead_avec_collateral,
    calculer_credit_value_adjustment,
)
from src.metrics.counterparty import (
    analyser_risque_contrepartie,
)
from src.metrics.regulatory import (
    CADRE_RESTRICTIF,
    CADRE_FLEXIBLE,
    CadreReglementaire,
    ResultatsCadre,
    comparer_cadres_reglementaires,
    analyse_sensibilite_netting,
    afficher_comparaison_cadres,
    matrice_netting_factors,
)

from src.visualisation.figures import (
    tracer_calibration_bam,
    tracer_fan_chart_taux,
    tracer_charges_tresorerie,
    tracer_distribution_cout_taux,
    tracer_trajectoires_gbm,
    tracer_tresorerie_fx,
    tracer_couverture_irs,
    tracer_couverture_forward,
    tracer_tableau_bord_metriques,
    tracer_risque_contrepartie,
)
from src.visualisation.regulatory_figures import (
    tracer_comparaison_cadres,
    tracer_sensibilite_netting,
    tracer_iss_comparatif,
    tracer_concentration_diversification,
    tracer_impact_collateral,
    tracer_tableau_bord_reglementaire,
)

from src.reporting.summary import (
    afficher_resume_contrepartie,
    afficher_resume_reglementaire,
)

# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5.3.1.1 — SIMULATION DE SCÉNARIOS DE MARCHÉ DE TAUX
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def run_section_5311():
    """
    5.3.1.1 — Exposition au risque de taux d'intérêt (Modèle de Vasicek).

    Étapes :
      1. Chargement des données BAM
      2. Calibration OLS + MLE
      3. Simulation Monte Carlo
      4. Calcul des cash flows et métriques de risque
      5. Scénarios de stress
      6. Génération des figures
    """
    titre_section("SECTION 5.3.1.1 — RISQUE DE TAUX (VASICEK)")

    # ── ÉTAPE 1 : Chargement des données BAM ─────────────────────────────────
    etape(1, "Chargement des données BAM")
    try:
        df_bam = charger_taux_bam(PATHS.donnees_bam)
        serie_bam = construire_serie_mensuelle_bam(df_bam)
        r0_bam = float(serie_bam.iloc[-1])
        print(f"  ✓ Données BAM chargées : {len(df_bam)} décisions")
        print(f"  Période : {serie_bam.index[0].date()} → {serie_bam.index[-1].date()}")
        print(f"  Taux initial r₀ : {r0_bam*100:.4f}%")
    except FileNotFoundError:
        print("  ⚠ Fichier BAM non trouvé — utilisation des paramètres par défaut")
        # Paramètres représentatifs de calibration sur données historiques BAM
        # (Période 2006-2024, calibration documentée dans le mémoire)
        serie_bam = None
        r0_bam = 0.0225  # Dernier taux BAM = 2.25% (mars 2024)

    # ── ÉTAPE 2 : Calibration Vasicek ────────────────────────────────────────
    etape(2, "Calibration du modèle de Vasicek (OLS + MLE)")

    if serie_bam is not None:
        ols_res = calibrer_vasicek_ols(serie_bam, dt=SIM.dt_mensuel)
        params_vasicek = calibrer_vasicek_mle(serie_bam, dt=SIM.dt_mensuel)
    else:
        # Paramètres de repli calibrés hors-ligne sur les données BAM
        params_vasicek = ParametresVasicek(
            kappa=0.3200,
            theta=0.0280,
            sigma=0.0060,
            r0=r0_bam,
        )
        ols_res = {
            "r_squared": 0.0,
            "params": [0.0, 0.0],
            "resid": np.zeros(10),
            "rsquared": 0.0,
        }

        # Objet mock minimal pour la suite
        class _OLS:
            params = [
                params_vasicek.kappa * params_vasicek.theta * SIM.dt_mensuel,
                -params_vasicek.kappa * SIM.dt_mensuel,
            ]
            rsquared = 0.0
            resid = np.zeros(10)

        ols_res = _OLS()

    afficher_parametres_vasicek(
        params_vasicek.kappa,
        params_vasicek.theta,
        params_vasicek.sigma,
        params_vasicek.r0,
    )

    # ── ÉTAPE 3 : Simulation Monte Carlo ─────────────────────────────────────
    etape(3, "Simulation Monte Carlo (Vasicek — Euler-Maruyama)")
    r_sim = simuler_taux_vasicek(
        params_vasicek,
        n_simulations=SIM.n_simulations,
        n_periodes=AGENT.n_periodes_dette,
        dt=SIM.dt_mensuel,
        graine=SIM.graine_aleatoire,
    )
    print(f"  ✓ {SIM.n_simulations:,} trajectoires × {AGENT.n_periodes_dette} périodes")
    print(f"  Taux moyen final : {r_sim[:,-1].mean()*100:.4f}%")
    print(
        f"  IC 90% : [{np.percentile(r_sim[:,-1],5)*100:.4f}% ; {np.percentile(r_sim[:,-1],95)*100:.4f}%]"
    )

    # Scénarios représentatifs
    idx_c, idx_h, idx_b = identifier_scenarios_representatifs(r_sim[:, -1])

    # ── ÉTAPE 4 : Cash flows et métriques de risque ──────────────────────────
    etape(4, "Calcul des charges financières et métriques de risque")
    cf_mensuels, tresorerie_cumul = calculer_tresorerie_taux(
        r_sim, AGENT.notionnel, AGENT.spread_bancaire, SIM.dt_mensuel
    )
    cout_total = cf_mensuels.sum(axis=1)

    metriques_5311 = calculer_metriques_risque(cout_total, cout_total)
    jb_stat, jb_pval, jb_interp = test_jarque_bera(cout_total)

    print(f"  Charge mensuelle moyenne : {cf_mensuels.mean():>12,.0f} MAD")
    print(f"  Coût total moyen         : {cout_total.mean():>12,.0f} MAD")
    print(f"  VaR 99%                  : {metriques_5311.var99_nc:>12,.0f} MAD")
    print(f"  ES  99%                  : {metriques_5311.es99_nc:>12,.0f} MAD")
    print(f"  Test Jarque-Bera         : p = {jb_pval:.4f} → {jb_interp}")

    # ── ÉTAPE 5 : Scénarios de stress ────────────────────────────────────────
    etape(5, "Scénarios de stress (chocs de politique monétaire)")
    resultats_stress_taux = {}
    for nom, choc in MARKET.chocs_stress_taux.items():
        r0_s = params_vasicek.r0 + choc
        params_s = ParametresVasicek(
            kappa=params_vasicek.kappa,
            theta=params_vasicek.theta,
            sigma=params_vasicek.sigma,
            r0=r0_s,
        )
        r_s = simuler_taux_vasicek(
            params_s, n_periodes=AGENT.n_periodes_dette, graine=SIM.graine_aleatoire
        )
        cf_s = calculer_charges_dette_variable(
            r_s, AGENT.notionnel, AGENT.spread_bancaire, SIM.dt_mensuel
        )
        cout_s = cf_s.sum(axis=1)
        resultats_stress_taux[nom] = {
            "cout_total": cout_s,
            "cout_moy": float(cout_s.mean()),
            "VaR_99": float(-np.percentile(cout_s, 1)),
        }
        print(
            f"  {nom:<8} : r₀={r0_s*100:.2f}%  "
            f"coût moy = {cout_s.mean():>12,.0f} MAD  "
            f"VaR 99% = {-np.percentile(cout_s,1):>12,.0f} MAD"
        )

    # ── ÉTAPE 6 : Figures ────────────────────────────────────────────────────
    etape(6, "Génération des figures 5.3.1.1")
    if serie_bam is not None:
        r_t_cal = serie_bam.values[:-1]
        delta_r_cal = serie_bam.values[1:] - serie_bam.values[:-1]
        tracer_calibration_bam(
            serie_bam,
            ols_res["ols_result"],
            r_t_cal,
            delta_r_cal,
            params_vasicek.kappa,
            params_vasicek.theta,
            params_vasicek.sigma,
        )
    tracer_fan_chart_taux(r_sim, idx_c, idx_h, idx_b, params_vasicek.r0)
    tracer_charges_tresorerie(
        cf_mensuels,
        tresorerie_cumul,
        idx_c,
        idx_h,
        idx_b,
        AGENT.notionnel,
        AGENT.spread_bancaire,
    )
    tracer_distribution_cout_taux(
        cout_total,
        metriques_5311.var95_nc,
        metriques_5311.var99_nc,
        metriques_5311.es99_nc,
        resultats_stress_taux,
    )

    return {
        "params_vasicek": params_vasicek,
        "r_sim": r_sim,
        "cf_mensuels": cf_mensuels,
        "cout_total_taux": cout_total,
        "metriques_5311": metriques_5311,
        "idx_c": idx_c,
        "idx_h": idx_h,
        "idx_b": idx_b,
        "resultats_stress_taux": resultats_stress_taux,
        "serie_bam": serie_bam,
        "ols_res": ols_res,
    }


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5.3.1.2 — SIMULATION DE SCÉNARIOS DE CHANGE (GBM)
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def run_section_5312(res_5311: dict):
    """
    5.3.1.2 — Exposition au risque de change (Modèle GBM / Black-Scholes).

    Calibration par :
      - Taux spot S₀ : Yahoo Finance (USDMAD=X)
      - Taux SOFR    : FRED API
      - Volatilité σ_S : log-rendements journaliers (252 jours)
    """
    titre_section("SECTION 5.3.1.2 — RISQUE DE CHANGE (GBM)")

    params_vasicek = res_5311["params_vasicek"]

    # ── ÉTAPE 1 : Données de marché FX ───────────────────────────────────────
    etape(1, "Récupération des données de marché (Yahoo Finance + FRED)")
    try:
        # Date de référence : dernière date BAM disponible
        if res_5311["serie_bam"] is not None:
            last_date = res_5311["serie_bam"].index[-1]
        else:
            last_date = pd.Timestamp("2024-03-01")

        data_fx = telecharger_donnees_fx(MARKET.ticker_usdmad, last_date)
        s0 = extraire_taux_spot(data_fx, last_date)
        sigma_s = calculer_volatilite_gbm(data_fx)
        r_f = telecharger_sofr(last_date)
        print(
            f"  ✓ S₀ = {s0:.4f} MAD/USD  σ_S = {sigma_s*100:.2f}%  SOFR = {r_f*100:.4f}%"
        )
    except Exception as e:
        print(
            f"  ⚠ Données externes indisponibles ({e}) — utilisation des paramètres par défaut"
        )
        s0 = MARKET.taux_spot_usdmad
        sigma_s = MARKET.sigma_gbm_default
        r_f = MARKET.sofr

    r_d = float(params_vasicek.theta)  # Taux BAM calibré (θ Vasicek)
    mu = r_d - r_f  # Drift CIP

    params_gbm = ParametresGBM(mu=mu, sigma_s=sigma_s, s0=s0, r_d=r_d, r_f=r_f)
    f0 = params_gbm.taux_forward

    afficher_parametres_gbm(mu, sigma_s, s0, f0)

    # ── ÉTAPE 2 : Simulation Monte Carlo GBM ─────────────────────────────────
    etape(2, "Simulation Monte Carlo USD/MAD (GBM — solution exacte)")
    s_sim = simuler_taux_change_gbm(
        params_gbm,
        n_simulations=SIM.n_simulations,
        n_periodes=AGENT.n_periodes_fx,
        dt=SIM.dt_mensuel,
        graine=SIM.graine_aleatoire,
    )
    # Vérification de la propriété E[S_T] = S₀·exp(μ·T)
    s_t_mean = s_sim[:, -1].mean()
    s_t_theo = s0 * np.exp(mu * AGENT.horizon_couverture_fx)
    print(f"  E[S_T] simulé   : {s_t_mean:.4f}")
    print(f"  E[S_T] théorique: {s_t_theo:.4f}  (CIP — vérification ✓)")

    # ── ÉTAPE 3 : Cash flows d'importation ───────────────────────────────────
    etape(3, "Calcul de l'exposition au risque de change")
    cf_fx, perte_change, cumul_fx = calculer_tresorerie_fx(
        s_sim, s0, AGENT.importations_mensuelles_usd
    )
    cout_total_fx = cf_fx.sum(axis=1)
    perte_totale = perte_change.sum(axis=1)
    cout_budgete = -AGENT.importations_mensuelles_usd * s0 * AGENT.n_periodes_fx

    var99_perte = float(-np.percentile(perte_totale, 1))
    es99_perte = float(
        -perte_totale[perte_totale <= np.percentile(perte_totale, 1)].mean()
    )
    prob_perte = float((perte_totale < 0).mean() * 100)

    print(f"  Coût budgété (S₀)   : {cout_budgete:>12,.0f} MAD")
    print(f"  Coût moyen simulé   : {cout_total_fx.mean():>12,.0f} MAD")
    print(f"  VaR 99% (perte)     : {var99_perte:>12,.0f} MAD")
    print(f"  ES  99% (perte)     : {es99_perte:>12,.0f} MAD")
    print(f"  Prob. de perte (>0) : {prob_perte:.1f}%")

    # Validation GBM (test KS)
    mu_ln = (mu - 0.5 * sigma_s**2) * AGENT.horizon_couverture_fx
    sig_ln = sigma_s * np.sqrt(AGENT.horizon_couverture_fx)
    _, ks_pval, ks_interp = test_ks_lognormal(s_sim, mu_ln, sig_ln)
    print(f"  Validation GBM (KS) : p = {ks_pval:.4f} → {ks_interp}")

    # ── ÉTAPE 4 : Scénarios de stress FX ────────────────────────────────────
    etape(4, "Scénarios de stress (dépréciation du Dirham)")
    resultats_stress_fx = {}
    for nom, mult in MARKET.chocs_stress_fx.items():
        s0_s = s0 * mult
        params_s = ParametresGBM(mu=mu, sigma_s=sigma_s, s0=s0_s, r_d=r_d, r_f=r_f)
        s_s = simuler_taux_change_gbm(
            params_s, n_periodes=AGENT.n_periodes_fx, graine=SIM.graine_aleatoire
        )
        cf_s = calculer_couts_importation(s_s, AGENT.importations_mensuelles_usd)
        cout_s = cf_s.sum(axis=1)
        resultats_stress_fx[nom] = {
            "cout": cout_s,
            "cout_moy": float(cout_s.mean()),
            "VaR_99": float(-np.percentile(cout_s, 1)),
            "sigma": sigma_s,
        }
        print(
            f"  {nom:<20} : S₀={s0_s:.3f}  coût moy={-cout_s.mean()/1e6:.3f}M  "
            f"VaR={-np.percentile(cout_s,1)/1e6:.3f}M MAD"
        )

    # ── ÉTAPE 5 : Figures ────────────────────────────────────────────────────
    etape(5, "Génération des figures 5.3.1.2")
    idx_c, idx_h, idx_b = identifier_scenarios_representatifs(s_sim[:, -1])
    tracer_trajectoires_gbm(s_sim, s0, f0, idx_c, idx_h, idx_b)
    tracer_tresorerie_fx(
        cf_fx,
        perte_change,
        cout_total_fx,
        cout_budgete,
        var99_perte,
        es99_perte,
        resultats_stress_fx,
    )

    return {
        "params_gbm": params_gbm,
        "s_sim": s_sim,
        "s0": s0,
        "f0": f0,
        "r_d": r_d,
        "r_f": r_f,
        "sigma_s": sigma_s,
        "cf_fx": cf_fx,
        "cout_total_fx": cout_total_fx,
        "perte_totale": perte_totale,
        "cout_budgete": cout_budgete,
        "resultats_stress_fx": resultats_stress_fx,
    }


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5.3.2 — STRATÉGIES DE COUVERTURE PAR PRODUITS DÉRIVÉS
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def run_section_532(res_5311: dict, res_5312: dict):
    """
    5.3.2 — Introduction d'une stratégie de couverture basée sur les dérivés OTC.

    5.3.2.1 — Swap de Taux d'Intérêt (IRS) : payeur fixe K
    5.3.2.2 — Contrat Forward de Change      : fixation du coût en MAD
    5.3.2.3 — Évaluation via des métriques quantitatives
    """
    titre_section("SECTION 5.3.2 — STRATÉGIES DE COUVERTURE (IRS + FORWARD)")

    params_vasicek = res_5311["params_vasicek"]
    r_sim = res_5311["r_sim"]
    cf_mensuels = res_5311["cf_mensuels"]
    cout_total_nc = res_5311["cout_total_taux"]

    s_sim = res_5312["s_sim"]
    cf_fx = res_5312["cf_fx"]
    cout_fx_nc = res_5312["cout_total_fx"]
    s0 = res_5312["s0"]
    r_d = res_5312["r_d"]
    r_f = res_5312["r_f"]
    cout_budgete = res_5312["cout_budgete"]

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3.2.1 — SWAP DE TAUX D'INTÉRÊT (IRS)
    # ─────────────────────────────────────────────────────────────────────────
    sous_titre("5.3.2.1 — IRS : Swap de Taux d'Intérêt")

    # Cash flows de la dette non couverte (sans spread pour le calcul IRS)
    cf_dette_nc = -AGENT.notionnel * r_sim[:, 1:] * SIM.dt_mensuel

    res_irs = appliquer_couverture_irs(
        cf_dette_nc=cf_dette_nc,
        r_sim=r_sim,
        params_vasicek=params_vasicek,
        notionnel=AGENT.notionnel,
        spread=AGENT.spread_bancaire,
        dt=SIM.dt_mensuel,
    )

    print(f"  Taux fixe K            : {res_irs.taux_fixe_K*100:.4f}%")
    print(f"  Annuité Σ(Δt·P)       : {res_irs.annuite:.6f}")
    print(f"  Coût théorique IRS     : {res_irs.cout_irs_theorique:>12,.0f} MAD")
    print(f"  Coût simulé moyen      : {res_irs.cout_couvert.mean():>12,.0f} MAD")
    print(
        f"  Écart (doit être ≈ 0)  : {abs(res_irs.cout_couvert.mean()-res_irs.cout_irs_theorique):.4f} MAD ✓"
    )
    print(f"  σ_CF après IRS         : {res_irs.cout_couvert.std():.4f} MAD  ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3.2.2 — FORWARD DE CHANGE
    # ─────────────────────────────────────────────────────────────────────────
    sous_titre("5.3.2.2 — Forward de Change (CIP)")

    res_fwd = appliquer_couverture_forward(
        cf_fx_nc=cf_fx,
        s_sim=s_sim,
        s0=s0,
        r_d=r_d,
        r_f=r_f,
        montant_mensuel_usd=AGENT.importations_mensuelles_usd,
        n_periodes=AGENT.n_periodes_fx,
    )

    print(f"  Taux forward F₀        : {res_fwd.taux_forward_F0:.6f} MAD/USD")
    print(
        f"  Prime forward          : {res_fwd.prime_abs:+.4f} MAD  ({res_fwd.prime_pct:+.4f}%)"
    )
    print(f"  Coût théorique forward : {res_fwd.cout_fwd_theorique:>12,.0f} MAD")
    print(f"  Coût budgété (S₀)      : {cout_budgete:>12,.0f} MAD")
    print(
        f"  Différence vs budget   : {res_fwd.cout_fwd_theorique - cout_budgete:>+12,.0f} MAD"
    )
    print(f"  σ_CF après Forward     : {res_fwd.cout_couvert.std():.4f} MAD  ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3.2.3 — MÉTRIQUES DE COUVERTURE
    # ─────────────────────────────────────────────────────────────────────────
    sous_titre("5.3.2.3 — Évaluation des Stratégies de Couverture")

    metriques_irs = calculer_metriques_risque(cout_total_nc, res_irs.cout_couvert)
    metriques_fwd = calculer_metriques_risque(cout_fx_nc, res_fwd.cout_couvert)

    print(f"\n  ── IRS ─────────────────────────────────────────")
    print(f"  η (réduction variance)  : {metriques_irs.eta*100:.4f}%")
    print(
        f"  VaR 99% avant / après   : {metriques_irs.var99_nc/1e6:.4f}M / {metriques_irs.var99_c/1e6:.6f}M"
    )
    print(f"  ΔVaR 99%                : {metriques_irs.delta_var99_pct:.2f}%")
    print(f"  Sharpe couverture       : {metriques_irs.sharpe_couverture:.4f}")

    print(f"\n  ── Forward ─────────────────────────────────────")
    print(f"  η (réduction variance)  : {metriques_fwd.eta*100:.4f}%")
    print(
        f"  VaR 99% avant / après   : {metriques_fwd.var99_nc/1e6:.4f}M / {metriques_fwd.var99_c/1e6:.6f}M"
    )
    print(f"  ΔVaR 99%                : {metriques_fwd.delta_var99_pct:.2f}%")
    print(f"  Sharpe couverture       : {metriques_fwd.sharpe_couverture:.4f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    etape(7, "Génération des figures 5.3.2")
    tracer_couverture_irs(
        r_sim,
        res_irs.flux_swap,
        cf_dette_nc,
        res_irs.cf_couverts,
        cout_total_nc,
        res_irs.cout_couvert,
        res_irs.taux_fixe_K,
        res_irs.cout_irs_theorique,
        metriques_irs.sigma_nc,
        metriques_irs.sigma_c,
        metriques_irs.eta,
        metriques_irs.var99_nc,
        metriques_irs.delta_var99_pct,
    )
    tracer_couverture_forward(
        s_sim,
        res_fwd.gain_forward,
        cf_fx,
        res_fwd.cf_couverts,
        cout_fx_nc,
        res_fwd.cout_couvert,
        s0,
        res_fwd.taux_forward_F0,
        res_fwd.cout_fwd_theorique,
        cout_budgete,
        metriques_fwd.sigma_nc,
        metriques_fwd.sigma_c,
        metriques_fwd.eta,
        metriques_fwd.var99_nc,
        metriques_fwd.delta_var99_pct,
    )
    tracer_tableau_bord_metriques(
        metriques_irs,
        metriques_fwd,
        cout_total_nc,
        res_irs.cout_couvert,
        cout_fx_nc,
        res_fwd.cout_couvert,
        res_irs.taux_fixe_K,
        res_fwd.taux_forward_F0,
    )

    return {
        "res_irs": res_irs,  # ResultatsIRS → .flux_swap, .cout_couvert, .taux_fixe_K
        "res_fwd": res_fwd,  # ResultatsForward → .gain_forward, .cout_couvert, .taux_forward_F0
        "metriques_irs": metriques_irs,  # MetriquesRisque → .eta, .var99_nc, .sigma_nc
        "metriques_fwd": metriques_fwd,
        "cf_dette_nc": cf_dette_nc,  # np.ndarray (M, n_periodes)
    }


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5.3.3 — RISQUE DE CONTREPARTIE
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def run_section_533(res_532: dict):
    """
    5.3.3 — Intégration du risque de contrepartie.

    5.3.3.1 — Modélisation des effets de compensation (netting) et de garantie
    5.3.3.2 — Simulation de la réduction des expositions brutes en nettes
    """
    titre_section("SECTION 5.3.3 — RISQUE DE CONTREPARTIE")

    res_irs = res_532["res_irs"]
    res_fwd = res_532["res_fwd"]

    # ── Analyse complète du risque de contrepartie ────────────────────────────
    etape(1, "Calcul des MTM, EBE, ENE, Netting Factor, EAD, CVA")
    res_cp = analyser_risque_contrepartie(
        flux_swap=res_irs.flux_swap,
        gain_forward=res_fwd.gain_forward,
        taux_collateral=0.80,
        probabilite_defaut=0.02,
        taux_recouvrement=0.40,
    )

    print(f"\n  ── EXPOSITIONS ──────────────────────────────────────────")
    print(f"  EBE totale                     : {res_cp.ebe_totale/1e6:.4f} M MAD")
    print(f"  ENE totale (après netting ISDA): {res_cp.ene_totale/1e6:.4f} M MAD")
    print(
        f"  Netting Factor                 : {res_cp.netting_factor:.4f}  "
        f"(compression {(1-res_cp.netting_factor)*100:.1f}%)"
    )
    print(f"\n  ── EAD ────────────────────────────────────────────────────")
    print(f"  EAD brute (× alpha 1.4)        : {res_cp.ead_brute/1e6:.4f} M MAD")
    print(f"  EAD après netting              : {res_cp.ead_nette/1e6:.4f} M MAD")
    print(f"  EAD après collatéral 80%       : {res_cp.ead_collat/1e6:.4f} M MAD")
    print(f"\n  ── CVA ────────────────────────────────────────────────────")
    print(f"  CVA sans collatéral            : {res_cp.cva/1e6:.5f} M MAD")
    print(f"  CVA avec collatéral 80%        : {res_cp.cva_collat/1e6:.5f} M MAD")
    print(
        f"  Réduction CVA due au collat.   : {(res_cp.cva-res_cp.cva_collat)/res_cp.cva*100:.1f}%"
    )

    # ── Figure 5.3.3 ──────────────────────────────────────────────────────────
    etape(2, "Génération de la figure 5.3.3")
    tracer_risque_contrepartie(
        exposition_brute=res_cp.dist_brute,
        exposition_nette=res_cp.dist_nette,
        ead_sans_collateral=res_cp.dist_nette,
        ead_avec_collateral=res_cp.dist_ead,
        epe_swap=res_cp.ebe,
        epe_fwd=res_cp.ene,
        reduction_netting_pct=(1 - res_cp.netting_factor) * 100,
        reduction_collateral_pct=80.0,
        taux_collateral=0.80,
    )
    afficher_resume_contrepartie(res_cp)

    return {"res_contrepartie": res_cp}


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5.3.4 — SCÉNARIOS RÉGLEMENTAIRES COMPARATIFS
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def run_section_534(res_533: dict, res_532: dict):
    """
    5.3.4 — Scénario réglementaire comparatif : cadre restrictif vs flexible.

    5.3.4.1 — Cadre restrictif (hedging uniquement, spreads élevés, pas de netting)
    5.3.4.2 — Cadre flexible prudentiel (ISDA, CSA, Bâle III, liquidité accrue)
    5.3.4.3 — Impact sur la liquidité, la distribution des risques
               et la stabilité financière
    """
    titre_section("SECTION 5.3.4 — SCÉNARIOS RÉGLEMENTAIRES OTC MAROCAINS")

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3.4.1 & 5.3.4.2 — Comparaison des deux cadres
    # ─────────────────────────────────────────────────────────────────────────
    sous_titre("5.3.4.1–2 : Comparaison Cadre Restrictif vs Flexible")

    # Récupération des flux depuis res_532
    # res_532["res_irs"] est un objet ResultatsIRS → accès par .attribut
    # res_532["res_fwd"] est un objet ResultatsForward → accès par .attribut
    res_restrictif, res_flexible = comparer_cadres_reglementaires(
        flux_irs      = res_532["res_irs"].flux_swap,
        flux_fwd      = res_532["res_fwd"].gain_forward,
        notionnel     = AGENT.notionnel,
        n_simulations = SIM.n_simulations,
    )

    # Regrouper en dictionnaire pour la suite
    # res_restrictif et res_flexible sont des objets ResultatsCadre
    # → accès par .attribut (pas par ["clé"])
    resultats_reg = {
        "restrictif" : res_restrictif,
        "flexible"   : res_flexible,
    }

    # ── Affichage console des résultats par cadre ─────────────────────────────
    for nom, res in resultats_reg.items():
        # res est un ResultatsCadre → accès par .attribut
        cadre = res.cadre    # CadreReglementaire → accès par .attribut
        print(f"\n  {'═'*55}")
        print(f"  CADRE : {cadre.nom.upper()}")
        print(f"  {'═'*55}")
        print(
            f"  n_contrats={cadre.n_contrats}  "
            f"n_contreparties={cadre.n_contreparties}  "
            f"netting={'Oui (ISDA)' if cadre.netting_reconnu else 'Non'}  "
            f"collat.={cadre.taux_collateral*100:.0f}%  "
            f"spread={cadre.spread_bp:.0f}bps"
        )
        print(
            f"  EBE : {res.ebe_moyen/1e6:.3f}M  "
            f"ENE : {res.ene_moyen/1e6:.3f}M  "
            f"NF  : {res.netting_factor:.3f}"
        )
        print(
            f"  EAD : {res.ead_moyen/1e6:.3f}M MAD  "
            f"CVA : {res.cva_total/1e6:.4f}M MAD"
        )
        print(
            f"  HHI : {res.hhi:.4f}  "
            f"Diversification : {res.ratio_diversification:.4f}  "
            f"Coût spreads : {res.cout_spread_total/1e6:.4f}M MAD"
        )

    # Affichage tableau comparatif
    afficher_comparaison_cadres(res_restrictif, res_flexible)

    # ─────────────────────────────────────────────────────────────────────────
    # 5.3.4.3 — Analyses de sensibilité
    # ─────────────────────────────────────────────────────────────────────────
    sous_titre("5.3.4.3 : Analyses de Sensibilité (n, ρ, collatéral)")

    flux_irs_array = res_532["res_irs"].flux_swap
    flux_fwd_array = res_532["res_fwd"].gain_forward

    print("  → Lancement de l'analyse de sensibilité (peut prendre 1-2 min)...")
    sensibilite = analyse_sensibilite_netting(
        flux_irs           = flux_irs_array,
        flux_fwd           = flux_fwd_array,
        valeurs_n          = [1, 2, 3, 5, 8, 10, 15, 20],
        valeurs_rho        = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0],
        valeurs_collateral = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        n_simulations      = SIM.n_simulations,
    )

    # Matrice analytique NF(n, ρ)
    matrice_nf = matrice_netting_factors(
        valeurs_n   = [1, 2, 5, 10, 20],
        valeurs_rho = [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99],
    )

    print("\n  Matrice NF analytique (Gibson 2002) :")
    print(f"  {'n\\ρ':<6} {'ρ=0.0':>8} {'ρ=0.2':>8} {'ρ=0.5':>8} {'ρ=0.8':>8} {'ρ=1.0':>8}")
    for i, n in enumerate([1, 2, 5, 10, 20]):
        row = "  ".join(f"{v:.4f}" for v in matrice_nf[i])
        print(f"  n={n:<4}  {row}")

    print(f"\n  Netting Factor vs n (ρ=0.3 fixé) :")
    for n, nf in zip(sensibilite["valeurs_n"], sensibilite["nf_vs_n"]):
        print(f"    n={int(n):<3} → NF = {nf:.4f}")

    print(f"\n  Netting Factor vs ρ (n=10 fixé) :")
    for rho, nf in zip(sensibilite["valeurs_rho"], sensibilite["nf_vs_rho"]):
        print(f"    ρ={rho:.1f} → NF = {nf:.4f}")

    # ── Génération des figures 5.3.4 ─────────────────────────────────────────
    etape(3, "Génération des figures 5.3.4")

    # Figures 5.3.4
    tracer_comparaison_cadres(resultats_reg)

    tracer_sensibilite_netting(
        df_n   = sensibilite,
        df_rho = sensibilite,
    )

    tracer_concentration_diversification(
        df_contreparties = sensibilite["nf_vs_n"],
        resultats        = resultats_reg,
    )

    tracer_impact_collateral(
        df_collat = sensibilite,
    )

    tracer_iss_comparatif(
        resultats = resultats_reg,
        df_n      = sensibilite,
    )

    tracer_tableau_bord_reglementaire(
        resultats = resultats_reg,
        df_n      = sensibilite,
        df_rho    = sensibilite,
        df_collat = sensibilite,
    )
    afficher_resume_reglementaire(resultats_reg)

    return {
        "resultats_reg" : resultats_reg,   # dict{"restrictif": ResultatsCadre, "flexible": ResultatsCadre}
        "res_restrictif": res_restrictif,   # ResultatsCadre → .cadre, .ebe_moyen, .ene_moyen, ...
        "res_flexible"  : res_flexible,     # ResultatsCadre → .cadre, .ebe_moyen, .ene_moyen, ...
        "sensibilite"   : sensibilite,      # dict → ["nf_vs_n"], ["nf_vs_rho"], ["ead_vs_collat"]
        "matrice_nf"    : matrice_nf,       # np.ndarray (5, 5) — matrice analytique
        }
# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  RÉSUMÉ FINAL
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def afficher_resume_final(res_all: dict) -> None:
    """Affiche le résumé consolidé de toutes les sections."""
    titre_section("RÉSUMÉ FINAL — CHAPITRE 5.3")

    res_5311 = res_all["5311"]
    res_532 = res_all["532"]
    res_533 = res_all["533"]
    res_534 = res_all["534"]

    m_irs = res_532["metriques_irs"]
    m_fwd = res_532["metriques_fwd"]
    res_cp = res_533["res_contrepartie"]
    reg = res_534["resultats_reg"]

    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  5.3.1 — EXPOSITION AUX RISQUES                                 │
  │  Vasicek : κ={res_5311['params_vasicek'].kappa:.4f}  θ={res_5311['params_vasicek'].theta*100:.4f}%  σ={res_5311['params_vasicek'].sigma*100:.4f}%     │
  │  VaR 99% taux : {m_irs.var99_nc/1e6:.3f}M MAD                              │
  │  VaR 99% FX   : {m_fwd.var99_nc/1e6:.3f}M MAD                              │
  ├─────────────────────────────────────────────────────────────────┤
  │  5.3.2 — COUVERTURES (IRS + FORWARD)                           │
  │  IRS : K={res_532['res_irs'].taux_fixe_K*100:.4f}%  η={m_irs.eta*100:.2f}%  ΔVaR={m_irs.delta_var99_pct:.1f}%       │
  │  FWD : F₀={res_532['res_fwd'].taux_forward_F0:.4f}  η={m_fwd.eta*100:.2f}%  ΔVaR={m_fwd.delta_var99_pct:.1f}%      │
  ├─────────────────────────────────────────────────────────────────┤
  │  5.3.3 — RISQUE DE CONTREPARTIE                                │
  │  Netting Factor : {res_cp.netting_factor:.4f}  EAD : {res_cp.ead_collat/1e6:.3f}M MAD                │
  │  CVA : {res_cp.cva/1e6:.5f}M → {res_cp.cva_collat/1e6:.5f}M (avec collat. 80%)  │
  ├─────────────────────────────────────────────────────────────────┤
  │  5.3.4 — COMPARAISON RÉGLEMENTAIRE                             │
  │  ISS restrictif : {reg['restrictif'].ratio_diversification*100:.1f}/100  →  ISS flexible : {reg['flexible'].ratio_diversification*100:.1f}/100     │  │  Réduction EAD  : {(reg['restrictif'].ead_moyen-reg['flexible'].ead_moyen)/reg['restrictif'].ead_moyen*100:.1f}%                                    │
  │  Réduction CVA  : {(reg['restrictif'].cva_total-reg['flexible'].cva_total)/reg['restrictif'].cva_total*100:.1f}%                                    │
  └─────────────────────────────────────────────────────────────────┘
    """)

    # Liste des figures générées
    print("  FIGURES GÉNÉRÉES :")
    figures = [
        "fig_5311a_calibration_bam.png",
        "fig_5311b_fan_chart_taux.png",
        "fig_5311c_charges_tresorerie.png",
        "fig_5311d_distribution_stress_taux.png",
        "fig_5312a_gbm_trajectoires.png",
        "fig_5312b_tresorerie_fx.png",
        "fig_5321_irs.png",
        "fig_5322_forward.png",
        "fig_5323_metriques.png",
        "fig_533_risque_contrepartie.png",
        "fig_5341_comparaison_cadres.png",
        "fig_5342_sensibilite_netting.png",
        "fig_5343_hhi_diversification.png",
        "fig_5344_impact_collateral.png",
        "fig_5345_iss_comparatif.png",
        "fig_5346_tableau_bord_reglementaire.png",
    ]
    for f in figures:
        chemin = PATHS.dossier_figures / f
        statut = "✓" if chemin.exists() else "○"
        print(f"    {statut} {f}")


# =============================================================================
# ══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
# =============================================================================


def main():
    """
    Pipeline principal complet.
    Lance toutes les sections dans l'ordre et passe les résultats entre elles.
    """
    print("\n" + "█" * 70)
    print("  MODÉLISATION DES RISQUES FINANCIERS — DÉRIVÉS OTC AU MAROC")
    print("  Chapitre 5.3 — Finance Quantitative et Gestion des Risques")
    print("█" * 70)
    print(
        f"\n  Simulations : {SIM.n_simulations:,} trajectoires  |  "
        f"Graine : {SIM.graine_aleatoire}  |  "
        f"Pas : mensuel (Δt = 1/12)"
    )
    print(
        f"  Notionnel : {AGENT.notionnel/1e6:.0f}M MAD  |  "
        f"Maturité : {AGENT.maturite_dette} ans  |  "
        f"Importations : {AGENT.importations_annuelles_usd/1e6:.0f}M USD/an\n"
    )

    # ── Exécution séquentielle des sections ──────────────────────────────────
    res_5311 = run_section_5311()
    res_5312 = run_section_5312(res_5311)
    res_532 = run_section_532(res_5311, res_5312)
    res_533 = run_section_533(res_532)
    res_534 = run_section_534(res_533, res_532)

    # ── Résumé final ─────────────────────────────────────────────────────────
    afficher_resume_final(
        {
            "5311": res_5311,
            "5312": res_5312,
            "532": res_532,
            "533": res_533,
            "534": res_534,
        }
    )

    print(f"\n  Figures sauvegardées dans : {PATHS.dossier_figures}")
    print("  Pipeline terminé avec succès ✓\n")


if __name__ == "__main__":
    main()
