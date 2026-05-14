"""
=============================================================================
visualization/figures.py — Génération de toutes les figures du projet
=============================================================================
Chaque fonction produit une figure analytique correspondant à une
sous-section du mémoire. Le style graphique est unifié via PlotParams.

Figures générées :
  fig_5311a  : Données BAM historiques + calibration OLS
  fig_5311b  : Fan chart Monte Carlo des taux (Vasicek)
  fig_5311c  : Charges mensuelles + trésorerie cumulée (risque de taux)
  fig_5311d  : Distribution du coût total (VaR, ES, stress)
  fig_5312a  : Fan chart GBM (trajectoires USD/MAD)
  fig_5312b  : Trésorerie FX + exposition au risque de change
  fig_5312c  : Distribution FX (VaR, perte) + sensibilité σ_S
  fig_5321   : IRS — mécanique et réduction de risque
  fig_5322   : Forward — mécanique et réduction de risque
  fig_5323   : Tableau de bord comparatif des métriques (IRS vs Forward)
  fig_533    : Risque de contrepartie (exposition brute vs nette vs EAD)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional

from src.config.settings import PLOT, PATHS


# =============================================================================
# UTILITAIRES INTERNES
# =============================================================================

def _style_ax(ax) -> None:
    """Applique le style sombre unifié à un axes matplotlib."""
    ax.set_facecolor(PLOT.bg_ax)
    ax.tick_params(colors=PLOT.txt, labelsize=8)
    ax.spines[:].set_color("#333")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(PLOT.txt)


def _nouvelle_figure(nrows=1, ncols=1, figsize=None) -> tuple:
    """Crée une figure avec fond sombre."""
    if figsize is None:
        figsize = (PLOT.fig_width_large, PLOT.fig_height_large)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(PLOT.bg)
    return fig, axes


def _sauvegarder(fig, nom_fichier: str, dossier: Path = None) -> str:
    """Sauvegarde la figure et retourne le chemin complet."""
    if dossier is None:
        dossier = PATHS.dossier_figures
    chemin = dossier / nom_fichier
    fig.savefig(chemin, dpi=PLOT.dpi, bbox_inches="tight", facecolor=PLOT.bg)
    plt.close(fig)
    print(f"  ✓ {nom_fichier}")
    return str(chemin)


# =============================================================================
# FIGURE 5.3.1.1 — A : DONNÉES BAM ET CALIBRATION OLS
# =============================================================================

def tracer_calibration_bam(
    serie_bam: pd.Series,
    ols_result,
    r_t: np.ndarray,
    delta_r: np.ndarray,
    kappa_ols: float,
    theta_ols: float,
    sigma_ols: float,
) -> str:
    """Figure 5.3.1.1a — Données BAM historiques et régression OLS."""
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(PLOT.bg)
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    # ── AX1 : Série historique des taux BAM ──────────────────────────────────
    ax = fig.add_subplot(gs[0])
    _style_ax(ax)
    ax.plot(serie_bam.index, serie_bam.values * 100, color=PLOT.c_obs, lw=2, label="Taux BAM observé")
    ax.axhline(theta_ols * 100, color=PLOT.c_mn, lw=1.5, ls="--",
               label=f"θ (OLS) = {theta_ols*100:.2f}%  (équilibre LT)")
    ax.set_title("Taux Directeur BAM — Données Historiques\n(Forward-fill mensuel)", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Date", color=PLOT.txt)
    ax.set_ylabel("Taux (%)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : Régression OLS (Δr_t vs r_t) ──────────────────────────────────
    ax = fig.add_subplot(gs[1])
    _style_ax(ax)
    ax.scatter(r_t * 100, delta_r * 100, alpha=0.6, s=25, color=PLOT.c_sim, label="Observations")
    x_line = np.linspace(r_t.min(), r_t.max(), 100)
    y_line = ols_result.params[0] + ols_result.params[1] * x_line
    ax.plot(x_line * 100, y_line * 100, color=PLOT.c_mn, lw=2, label=f"OLS  R²={ols_result.rsquared:.4f}")
    ax.axhline(0, color="white", lw=0.8, ls=":", alpha=0.5)
    ax.set_title(f"Régression OLS : Δr_t = A + B·r_t\n"
                 f"κ={kappa_ols:.4f}  θ={theta_ols*100:.4f}%  σ={sigma_ols*100:.4f}%",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("r_t (%)", color=PLOT.txt)
    ax.set_ylabel("Δr_t (%)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.1.1a — Données BAM et Calibration OLS du Modèle de Vasicek",
                 color="white", fontsize=12, fontweight="bold")
    return _sauvegarder(fig, "fig_5311a_calibration_bam.png")


# =============================================================================
# FIGURE 5.3.1.1 — B : FAN CHART DES TAUX SIMULÉS
# =============================================================================

def tracer_fan_chart_taux(
    r_sim: np.ndarray,
    idx_central: int,
    idx_haussier: int,
    idx_baissier: int,
    r0: float,
) -> str:
    """Figure 5.3.1.1b — Fan chart Monte Carlo des taux directeurs simulés."""
    fig, ax = _nouvelle_figure()
    _style_ax(ax)

    n_periods = r_sim.shape[1] - 1
    t = np.arange(n_periods + 1)

    # Intervalles de confiance (fan chart)
    p = lambda q: np.percentile(r_sim, q, axis=0) * 100
    ax.fill_between(t, p(1), p(99), alpha=0.12, color=PLOT.c_ci, label="IC 99%")
    ax.fill_between(t, p(5), p(95), alpha=0.20, color=PLOT.c_ci, label="IC 90%")
    ax.fill_between(t, p(25), p(75), alpha=0.30, color=PLOT.c_ci, label="IC 50%")

    # Trajectoires représentatives
    ax.plot(t, r_sim.mean(axis=0) * 100, color=PLOT.c_mn, lw=2.5, label="Moyenne")
    ax.plot(t, r_sim[idx_central] * 100,  color=PLOT.c_obs, lw=1.8, label="Scén. central")
    ax.plot(t, r_sim[idx_haussier] * 100, color=PLOT.c_var, lw=1.8, label="Scén. haussier")
    ax.plot(t, r_sim[idx_baissier] * 100, color=PLOT.c_eq,  lw=1.8, label="Scén. baissier")
    ax.axhline(r0 * 100, color="white", lw=1, ls=":", alpha=0.6, label=f"r₀ = {r0*100:.2f}%")

    ax.set_title(f"Figure 5.3.1.1b — Fan Chart Monte Carlo des Taux Directeurs\n"
                 f"Modèle de Vasicek — {r_sim.shape[0]:,} trajectoires",
                 color=PLOT.txt, fontsize=11)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Taux directeur (%)", color=PLOT.txt)
    ax.legend(fontsize=9, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)
    return _sauvegarder(fig, "fig_5311b_fan_chart_taux.png")


# =============================================================================
# FIGURE 5.3.1.1 — C : CHARGES MENSUELLES ET TRÉSORERIE CUMULÉE
# =============================================================================

def tracer_charges_tresorerie(
    cf_mensuels: np.ndarray,
    tresorerie_cumul: np.ndarray,
    idx_central: int,
    idx_haussier: int,
    idx_baissier: int,
    notionnel: float,
    spread: float,
) -> str:
    """Figure 5.3.1.1c — Charges mensuelles et trésorerie cumulée."""
    fig, axes = _nouvelle_figure(1, 2, figsize=(16, 7))
    mois = np.arange(1, cf_mensuels.shape[1] + 1)

    # ── AX1 : Charges mensuelles ──────────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    p5  = np.percentile(cf_mensuels, 5, axis=0) / 1e6
    p95 = np.percentile(cf_mensuels, 95, axis=0) / 1e6
    ax.fill_between(mois, p5, p95, alpha=0.25, color=PLOT.c_ci, label="IC 90%")
    ax.plot(mois, cf_mensuels.mean(axis=0) / 1e6, color=PLOT.c_mn, lw=2.5, label="Charge moyenne")
    ax.plot(mois, cf_mensuels[idx_haussier] / 1e6, color=PLOT.c_var, lw=2, label="Scén. haussier")
    ax.plot(mois, cf_mensuels[idx_baissier] / 1e6, color=PLOT.c_eq,  lw=2, label="Scén. baissier")
    for q in range(0, cf_mensuels.shape[1], 3):
        ax.axvline(q + 1, color="white", alpha=0.07, lw=0.8, ls="--")
    ax.set_title(f"Charges Mensuelles — Dette {notionnel/1e6:.0f}M MAD\n"
                 f"Taux variable = BAM + {spread*100:.0f}bp", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Charge (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=9, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : Trésorerie cumulée ─────────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    p5c  = np.percentile(tresorerie_cumul, 5, axis=0) / 1e6
    p95c = np.percentile(tresorerie_cumul, 95, axis=0) / 1e6
    ax.fill_between(mois, p5c, p95c, alpha=0.20, color=PLOT.c_ci, label="IC 90%")
    ax.plot(mois, tresorerie_cumul.mean(axis=0) / 1e6, color=PLOT.c_mn, lw=2.5, label="Moyenne")
    ax.plot(mois, tresorerie_cumul[idx_central] / 1e6,  color=PLOT.c_obs, lw=2, label="Central")
    ax.plot(mois, tresorerie_cumul[idx_haussier] / 1e6, color=PLOT.c_var, lw=2, label="Haussier")
    ax.plot(mois, tresorerie_cumul[idx_baissier] / 1e6, color=PLOT.c_eq,  lw=2, label="Baissier")
    ax.set_title("Impact Cumulé sur la Trésorerie\n(Charges financières accumulées)", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Charges cumulées (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=9, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.1.1c — Impact du Risque de Taux sur la Trésorerie",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5311c_charges_tresorerie.png")


# =============================================================================
# FIGURE 5.3.1.1 — D : DISTRIBUTION DU COÛT TOTAL + STRESS
# =============================================================================

def tracer_distribution_cout_taux(
    cout_total: np.ndarray,
    var95: float,
    var99: float,
    es99: float,
    resultats_stress: Dict,
) -> str:
    """Figure 5.3.1.1d — Distribution du coût total et scénarios de stress."""
    fig, axes = _nouvelle_figure(1, 2, figsize=(16, 6))

    # ── AX1 : Distribution du coût total ──────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    ax.hist(cout_total / 1e6, bins=70, color=PLOT.c_sim, alpha=0.80, density=True,
            edgecolor="none", label="Distribution simulée")
    ax.axvline(cout_total.mean() / 1e6, color=PLOT.c_mn, lw=2, label=f"Moyenne = {cout_total.mean()/1e6:.3f}M")
    ax.axvline(-var99 / 1e6, color=PLOT.c_var, lw=2, ls="--", label=f"VaR 99% = {var99/1e6:.3f}M")
    ax.axvline(-var95 / 1e6, color=PLOT.c_str, lw=1.5, ls="--", label=f"VaR 95% = {var95/1e6:.3f}M")
    ax.axvline(-es99 / 1e6, color="#FF6B6B", lw=1.5, ls=":", label=f"ES  99% = {es99/1e6:.3f}M")
    ax.set_title("Distribution du Coût Total de la Dette (3 ans)\nMétriques de risque", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : Scénarios de stress ─────────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    couleurs = {"Base": PLOT.c_obs, "+100bp": PLOT.c_mn, "+200bp": PLOT.c_var}
    for nom, res in resultats_stress.items():
        c = couleurs.get(nom, PLOT.c_str)
        ax.hist(res["cout_total"] / 1e6, bins=60, alpha=0.55, density=True,
                color=c, edgecolor="none",
                label=f"{nom} — moy={res['cout_moy']/1e6:.3f}M  VaR={res['VaR_99']/1e6:.3f}M")
        ax.axvline(-res["VaR_99"] / 1e6, color=c, lw=1.5, ls="--", alpha=0.9)
    ax.set_title("Scénarios de Stress (Taux)\nImpact d'un choc de politique monétaire", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.1.1d — Distribution des Coûts et Analyse de Stress",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5311d_distribution_stress_taux.png")


# =============================================================================
# FIGURE 5.3.1.2 — A : TRAJECTOIRES GBM USD/MAD
# =============================================================================

def tracer_trajectoires_gbm(
    s_sim: np.ndarray,
    s0: float,
    f0: float,
    idx_central: int,
    idx_haussier: int,
    idx_baissier: int,
) -> str:
    """Figure 5.3.1.2a — Fan chart des trajectoires USD/MAD (GBM)."""
    fig, ax = _nouvelle_figure()
    _style_ax(ax)

    n = s_sim.shape[1] - 1
    t = np.arange(n + 1)

    for i in range(300):
        ax.plot(t, s_sim[i], alpha=0.03, color=PLOT.c_nc, lw=0.5)
    ax.fill_between(t, np.percentile(s_sim, 5, axis=0),
                    np.percentile(s_sim, 95, axis=0), alpha=0.20, color=PLOT.c_ci, label="IC 90%")
    ax.plot(t, s_sim.mean(axis=0), color=PLOT.c_mn, lw=2.5, label="Moyenne simulée")
    ax.plot(t, s_sim[idx_central],  color=PLOT.c_obs, lw=2, label=f"Central  : {s_sim[idx_central,-1]:.4f}")
    ax.plot(t, s_sim[idx_haussier], color=PLOT.c_var, lw=2, label=f"Haussier : {s_sim[idx_haussier,-1]:.4f}")
    ax.plot(t, s_sim[idx_baissier], color=PLOT.c_eq,  lw=2, label=f"Baissier : {s_sim[idx_baissier,-1]:.4f}")
    ax.axhline(s0, color=PLOT.c_obs, lw=1.5, ls=":", alpha=0.8, label=f"S₀ = {s0:.4f} (budget)")
    ax.axhline(f0, color=PLOT.c_fwd, lw=2, ls="--", label=f"F₀ = {f0:.4f} (forward)")

    ax.set_title(f"Figure 5.3.1.2a — Trajectoires USD/MAD (Modèle GBM)\n"
                 f"S₀={s0:.4f}  F₀={f0:.4f}  ({s_sim.shape[0]:,} simulations)",
                 color=PLOT.txt, fontsize=11)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Taux USD/MAD", color=PLOT.txt)
    ax.legend(fontsize=9, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)
    return _sauvegarder(fig, "fig_5312a_gbm_trajectoires.png")


# =============================================================================
# FIGURE 5.3.1.2 — B : TRÉSORERIE FX
# =============================================================================

def tracer_tresorerie_fx(
    cf_fx: np.ndarray,
    perte_change: np.ndarray,
    cout_total_fx: np.ndarray,
    cout_budgete: float,
    var99_perte: float,
    es99_perte: float,
    resultats_stress_fx: Dict,
) -> str:
    """Figure 5.3.1.2b — Impact du risque de change sur la trésorerie."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(PLOT.bg)
    mois = np.arange(1, cf_fx.shape[1] + 1)

    # AX1 — Coûts mensuels FX
    ax = axes[0, 0]
    _style_ax(ax)
    ax.fill_between(mois, np.percentile(cf_fx, 5, axis=0)/1e6,
                    np.percentile(cf_fx, 95, axis=0)/1e6, alpha=0.25, color=PLOT.c_ci, label="IC 90%")
    ax.plot(mois, cf_fx.mean(axis=0)/1e6, color=PLOT.c_mn, lw=2.5, label="Coût moyen")
    ax.axhline(cout_budgete/cf_fx.shape[1]/1e6, color="white", lw=1.5, ls=":", alpha=0.7, label="Budget mensuel")
    ax.set_title("Coûts Mensuels d'Importation\n(USD/MAD × X_mensuel)", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Coût mensuel (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX2 — Pertes vs budget
    ax = axes[0, 1]
    _style_ax(ax)
    ax.hist(perte_change.sum(axis=1)/1e6, bins=70, color=PLOT.c_sim, alpha=0.75, density=True, edgecolor="none")
    ax.axvline(-var99_perte/1e6, color=PLOT.c_var, lw=2, ls="--", label=f"VaR 99% = {var99_perte/1e6:.3f}M")
    ax.axvline(-es99_perte/1e6,  color=PLOT.c_es,  lw=1.5, ls=":", label=f"ES  99% = {es99_perte/1e6:.3f}M")
    ax.axvline(0, color="white", lw=1, ls=":", alpha=0.6, label="Neutre (≡ budget)")
    ax.set_title("Distribution de la Perte de Change\nvs Budget (S₀)", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Perte (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX3 — Coût total FX
    ax = axes[1, 0]
    _style_ax(ax)
    ax.hist(cout_total_fx/1e6, bins=70, color=PLOT.c_sim, alpha=0.80, density=True, edgecolor="none")
    ax.axvline(cout_total_fx.mean()/1e6, color=PLOT.c_mn, lw=2, label=f"Moy = {cout_total_fx.mean()/1e6:.3f}M")
    ax.axvline(cout_budgete/1e6, color="white", lw=1.5, ls=":", label=f"Budget = {cout_budgete/1e6:.3f}M")
    ax.set_title("Distribution du Coût Total FX\n(1 an)", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX4 — Stress FX
    ax = axes[1, 1]
    _style_ax(ax)
    couleurs_stress = {
        "Base": PLOT.c_obs, "Dépréc. -5%": PLOT.c_mn,
        "Dépréc. -10%": PLOT.c_str, "Dépréc. -15%": PLOT.c_var,
    }
    for nom, res in resultats_stress_fx.items():
        c = couleurs_stress.get(nom, PLOT.c_str)
        ax.hist(res["cout"]/1e6, bins=60, alpha=0.50, density=True, color=c, edgecolor="none",
                label=f"{nom}  VaR={res['VaR_99']/1e6:.2f}M")
    ax.axvline(cout_budgete/1e6, color="white", lw=1.5, ls=":", label=f"Budget = {cout_budgete/1e6:.2f}M")
    ax.set_title("Scénarios de Stress FX\nDépréciation du Dirham", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.1.2b — Impact du Risque de Change sur la Trésorerie",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _sauvegarder(fig, "fig_5312b_tresorerie_fx.png")


# =============================================================================
# FIGURE 5.3.2.1 — IRS : MÉCANIQUE ET RÉDUCTION DE RISQUE
# =============================================================================

def tracer_couverture_irs(
    r_sim: np.ndarray,
    flux_swap: np.ndarray,
    cf_dette_nc: np.ndarray,
    cf_irs: np.ndarray,
    cout_dette_nc: np.ndarray,
    cout_irs: np.ndarray,
    K: float,
    cout_irs_theorique: float,
    sigma_dette_nc: float,
    sigma_irs: float,
    eta_irs: float,
    var99_nc: float,
    delta_var_irs: float,
) -> str:
    """Figure 5.3.2.1 — IRS : mécanique du swap et effet sur la trésorerie."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PLOT.bg)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    n = r_sim.shape[1] - 1
    mois = np.arange(1, n + 1)

    idx_up = int(np.argmin(np.abs(r_sim[:, -1] - np.percentile(r_sim[:, -1], 95))))
    idx_dn = int(np.argmin(np.abs(r_sim[:, -1] - np.percentile(r_sim[:, -1], 5))))
    idx_mn = int(np.argmin(np.abs(r_sim[:, -1] - r_sim[:, -1].mean())))

    # AX1 — Taux variable vs K
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax)
    for i in range(300):
        ax.plot(mois, r_sim[i, 1:] * 100, alpha=0.03, color=PLOT.c_nc, lw=0.5)
    ax.fill_between(mois, np.percentile(r_sim[:, 1:], 5, axis=0)*100,
                    np.percentile(r_sim[:, 1:], 95, axis=0)*100, alpha=0.18, color=PLOT.c_ci, label="IC 90%")
    ax.plot(mois, r_sim[idx_up, 1:]*100, color=PLOT.c_var, lw=1.8,
            label=f"Scén. haussier (+{(r_sim[idx_up,-1]-r_sim[idx_mn,-1])*100:+.2f}%)")
    ax.plot(mois, r_sim[idx_dn, 1:]*100, color=PLOT.c_irs, lw=1.8,
            label=f"Scén. baissier ({(r_sim[idx_dn,-1]-r_sim[idx_mn,-1])*100:+.2f}%)")
    ax.axhline(K*100, color=PLOT.c_mn, lw=2.5, ls="--", label=f"K = {K*100:.4f}% (taux fixe swap)")
    ax.set_title("Taux Variable vs Taux Fixe K\n(IRS : payeur fixe / receveur variable)", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Taux (%)", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX2 — Flux nets du swap
    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax)
    for i in range(300):
        ax.plot(mois, flux_swap[i]/1e3, alpha=0.03, color=PLOT.c_nc, lw=0.5)
    ax.fill_between(mois, np.percentile(flux_swap, 5, axis=0)/1e3,
                    np.percentile(flux_swap, 95, axis=0)/1e3, alpha=0.20, color=PLOT.c_ci, label="IC 90%")
    ax.plot(mois, flux_swap[idx_up]/1e3, color=PLOT.c_var, lw=1.8, label="Flux swap (haussier) — gain")
    ax.plot(mois, flux_swap[idx_dn]/1e3, color=PLOT.c_irs, lw=1.8, label="Flux swap (baissier) — perte")
    ax.plot(mois, flux_swap.mean(axis=0)/1e3, color=PLOT.c_mn, lw=2, label="Flux moyen")
    ax.axhline(0, color="white", lw=1, ls=":", alpha=0.7)
    ax.set_title(r"Flux Nets du Swap par Période : $N \cdot (r_t - K) \cdot \Delta t$",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Flux (K MAD)", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX3 — Trésorerie cumulée NC vs IRS
    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax)
    tresor_nc  = cf_dette_nc.cumsum(axis=1)
    tresor_irs = cf_irs.cumsum(axis=1)
    ax.fill_between(mois, np.percentile(tresor_nc, 5, axis=0)/1e6,
                    np.percentile(tresor_nc, 95, axis=0)/1e6, alpha=0.20, color=PLOT.c_nc, label="IC 90% (NC)")
    ax.plot(mois, tresor_nc.mean(axis=0)/1e6, color=PLOT.c_nc, lw=2,
            label=f"Moyen NC : {cout_dette_nc.mean()/1e6:.3f}M MAD")
    ax.plot(mois, tresor_irs[0]/1e6, color=PLOT.c_mn, lw=2.5, ls="--",
            label=f"IRS (certain) : {cout_irs_theorique/1e6:.3f}M MAD")
    ax.fill_between(mois, np.percentile(tresor_nc, 5, axis=0)/1e6, tresor_irs[0]/1e6,
                    alpha=0.12, color=PLOT.c_irs, label="Incertitude supprimée")
    ax.set_title("Trésorerie Cumulée — Avant vs Après IRS\nL'incertitude est totalement éliminée",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Charge cumulée (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX4 — Distribution NC vs IRS
    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax)
    ax.hist(cout_dette_nc/1e6, bins=70, alpha=0.65, color=PLOT.c_nc, density=True,
            edgecolor="none", label=f"Non couvert  σ={sigma_dette_nc/1e6:.3f}M")
    if cout_irs.std() < 1e-6:
        ax.axvline(cout_irs.mean()/1e6, color=PLOT.c_irs, lw=3,
                   label=f"IRS (certain) = {cout_irs.mean()/1e6:.4f}M MAD")
    else:
        ax.hist(cout_irs/1e6, bins=30, alpha=0.90, color=PLOT.c_irs, density=True,
                edgecolor="none", label=f"Avec IRS     σ≈{sigma_irs:.2f} MAD")
    ax.axvline(-var99_nc/1e6, color=PLOT.c_var, lw=2, ls="--",
               label=f"VaR 99% NC = {var99_nc/1e6:.3f}M")
    ax.axvline(cout_irs_theorique/1e6, color=PLOT.c_irs, lw=2.5,
               label=f"IRS = {cout_irs_theorique/1e6:.4f}M (certain)")
    ax.set_title(f"Distribution du Coût Total (3 ans)\nη = {eta_irs*100:.2f}%  |  ΔVaR = {delta_var_irs:.2f}%",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle(f"Figure 5.3.2.1 — Swap de Taux d'Intérêt (IRS)\n"
                 f"Couverture dette 100M MAD — Taux fixe K = {K*100:.4f}%",
                 color="white", fontsize=12, fontweight="bold")
    return _sauvegarder(fig, "fig_5321_irs.png")


# =============================================================================
# FIGURE 5.3.2.2 — FORWARD : MÉCANIQUE ET RÉDUCTION DE RISQUE
# =============================================================================

def tracer_couverture_forward(
    s_sim: np.ndarray,
    gain_forward: np.ndarray,
    cf_fx_nc: np.ndarray,
    cf_fwd: np.ndarray,
    cout_fx_nc: np.ndarray,
    cout_fwd: np.ndarray,
    s0: float,
    f0: float,
    cout_fwd_theorique: float,
    cout_budget: float,
    sigma_fx_nc: float,
    sigma_fwd: float,
    eta_fwd: float,
    var99_nc: float,
    delta_var_fwd: float,
) -> str:
    """Figure 5.3.2.2 — Forward de change : mécanique et effet sur la trésorerie."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(PLOT.bg)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)
    n = s_sim.shape[1] - 1
    mois = np.arange(1, n + 1)

    idx_up = int(np.argmin(np.abs(s_sim[:, -1] - np.percentile(s_sim[:, -1], 95))))
    idx_dn = int(np.argmin(np.abs(s_sim[:, -1] - np.percentile(s_sim[:, -1], 5))))

    # AX1 — Trajectoires USD/MAD vs F0
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax)
    for i in range(300):
        ax.plot(range(n + 1), s_sim[i], alpha=0.03, color=PLOT.c_nc, lw=0.5)
    ax.fill_between(range(n + 1), np.percentile(s_sim, 5, axis=0),
                    np.percentile(s_sim, 95, axis=0), alpha=0.20, color=PLOT.c_ci, label="IC 90%")
    ax.plot(range(n + 1), s_sim.mean(axis=0), color=PLOT.c_mn, lw=2, label="Moyenne simulée")
    ax.plot(range(n + 1), s_sim[idx_up], color=PLOT.c_var, lw=1.8,
            label=f"Haussier : {s_sim[idx_up,-1]:.3f}")
    ax.plot(range(n + 1), s_sim[idx_dn], color=PLOT.c_irs, lw=1.8,
            label=f"Baissier : {s_sim[idx_dn,-1]:.3f}")
    ax.axhline(s0, color=PLOT.c_obs, lw=1.5, ls=":", label=f"S₀ = {s0:.2f} (budget)")
    ax.axhline(f0, color=PLOT.c_fwd, lw=2.5, ls="--", label=f"F₀ = {f0:.4f} (forward)")
    ax.set_title("Taux USD/MAD vs Taux Forward F₀\nL'importateur est protégé au-delà de F₀",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("MAD / USD", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX2 — Gain/perte sur le forward
    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax)
    for i in range(300):
        ax.plot(mois, gain_forward[i]/1e3, alpha=0.03, color=PLOT.c_nc, lw=0.5)
    ax.fill_between(mois, np.percentile(gain_forward, 5, axis=0)/1e3,
                    np.percentile(gain_forward, 95, axis=0)/1e3, alpha=0.20, color=PLOT.c_ci, label="IC 90%")
    ax.plot(mois, gain_forward[idx_up]/1e3, color=PLOT.c_var, lw=1.8, label="Gain (scén. haussier)")
    ax.plot(mois, gain_forward[idx_dn]/1e3, color=PLOT.c_irs, lw=1.8, label="Perte (scén. baissier)")
    ax.plot(mois, gain_forward.mean(axis=0)/1e3, color=PLOT.c_mn, lw=2, label="Gain moyen")
    ax.axhline(0, color="white", lw=1, ls=":", alpha=0.7)
    ax.set_title(r"Gain/Perte sur le Contrat Forward : $X_{mensuel} \cdot (S_t - F_0)$",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Gain (K MAD)", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX3 — Trésorerie cumulée NC vs Forward
    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax)
    cumul_nc  = cf_fx_nc.cumsum(axis=1)
    cumul_fwd = cf_fwd.cumsum(axis=1)
    ax.fill_between(mois, np.percentile(cumul_nc, 5, axis=0)/1e6,
                    np.percentile(cumul_nc, 95, axis=0)/1e6, alpha=0.20, color=PLOT.c_nc, label="IC 90% (NC)")
    ax.plot(mois, cumul_nc.mean(axis=0)/1e6, color=PLOT.c_nc, lw=2,
            label=f"Moyen NC : {cout_fx_nc.mean()/1e6:.3f}M MAD")
    ax.plot(mois, cumul_fwd[0]/1e6, color=PLOT.c_mn, lw=2.5, ls="--",
            label=f"Forward (certain) : {cout_fwd_theorique/1e6:.3f}M MAD")
    ax.axhline(cout_budget/1e6, color=PLOT.c_fwd, lw=1.5, ls=":",
               label=f"Budget S₀ : {cout_budget/1e6:.3f}M MAD")
    ax.set_title("Coût d'Importation Cumulé — Avant vs Après Forward\n"
                 "Le coût est fixé à F₀, indépendant de S_T", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("Coût cumulé (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX4 — Distribution NC vs Forward
    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax)
    ax.hist(cout_fx_nc/1e6, bins=70, alpha=0.65, color=PLOT.c_nc, density=True,
            edgecolor="none", label=f"Non couvert  σ={sigma_fx_nc/1e6:.3f}M")
    if cout_fwd.std() < 1e-6:
        ax.axvline(cout_fwd.mean()/1e6, color=PLOT.c_fwd, lw=3,
                   label=f"Forward (certain) = {cout_fwd.mean()/1e6:.4f}M MAD")
    else:
        ax.hist(cout_fwd/1e6, bins=30, alpha=0.90, color=PLOT.c_fwd, density=True,
                edgecolor="none", label=f"Forward      σ≈{sigma_fwd:.2f} MAD")
    ax.axvline(-var99_nc/1e6, color=PLOT.c_var, lw=2, ls="--",
               label=f"VaR 99% NC = {var99_nc/1e6:.3f}M")
    ax.axvline(cout_fwd_theorique/1e6, color=PLOT.c_fwd, lw=2.5,
               label=f"Forward = {cout_fwd_theorique/1e6:.4f}M (certain)")
    ax.axvline(cout_budget/1e6, color=PLOT.c_mn, lw=1.5, ls=":",
               label=f"Budget S₀ = {cout_budget/1e6:.4f}M")
    ax.set_title(f"Distribution du Coût FX Total (1 an)\nη = {eta_fwd*100:.2f}%  |  ΔVaR = {delta_var_fwd:.2f}%",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle(f"Figure 5.3.2.2 — Contrat Forward de Change\n"
                 f"Couverture importations 5M USD — F₀ = {f0:.4f} MAD/USD",
                 color="white", fontsize=12, fontweight="bold")
    return _sauvegarder(fig, "fig_5322_forward.png")


# =============================================================================
# FIGURE 5.3.2.3 — TABLEAU DE BORD COMPARATIF DES MÉTRIQUES
# =============================================================================

def tracer_tableau_bord_metriques(
    metriques_taux,
    metriques_fx,
    cout_dette_nc: np.ndarray,
    cout_irs: np.ndarray,
    cout_fx_nc: np.ndarray,
    cout_fwd: np.ndarray,
    K: float,
    f0: float,
) -> str:
    """Figure 5.3.2.3 — Tableau de bord comparatif IRS vs Forward."""
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(PLOT.bg)
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    cats = ["VaR 95%", "VaR 99%", "ES 99%"]
    x = np.arange(3)
    w = 0.35

    def barplot_compare(ax, vals_nc, vals_c, color_nc, color_c, label_nc, label_c, titre):
        _style_ax(ax)
        ax.bar(x - w/2, vals_nc, w, color=color_nc, alpha=0.85, label=label_nc)
        ax.bar(x + w/2, vals_c,  w, color=color_c,  alpha=0.85, label=label_c)
        for i, (a, b) in enumerate(zip(vals_nc, vals_c)):
            ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
            ax.text(i + w/2, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, color=PLOT.txt, fontsize=9)
        ax.set_title(titre, color=PLOT.txt, fontsize=10)
        ax.set_ylabel("M MAD", color=PLOT.txt)
        ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX1 — VaR/ES taux
    ax = fig.add_subplot(gs[0, 0])
    barplot_compare(ax,
        [metriques_taux.var95_nc/1e6, metriques_taux.var99_nc/1e6, metriques_taux.es99_nc/1e6],
        [metriques_taux.var95_c/1e6,  metriques_taux.var99_c/1e6,  metriques_taux.es99_c/1e6],
        PLOT.c_nc, PLOT.c_irs, "Non couvert", "Avec IRS",
        f"VaR & ES — Risque de Taux\nη = {metriques_taux.eta*100:.2f}%")

    # AX2 — VaR/ES change
    ax = fig.add_subplot(gs[0, 1])
    barplot_compare(ax,
        [metriques_fx.var95_nc/1e6, metriques_fx.var99_nc/1e6, metriques_fx.es99_nc/1e6],
        [metriques_fx.var95_c/1e6,  metriques_fx.var99_c/1e6,  metriques_fx.es99_c/1e6],
        PLOT.c_nc, PLOT.c_fwd, "Non couvert", "Avec Forward",
        f"VaR & ES — Risque de Change\nη = {metriques_fx.eta*100:.2f}%")

    # AX3 — Écart-types σ_CF
    ax = fig.add_subplot(gs[0, 2])
    _style_ax(ax)
    labels_sig = ["Taux\n(NC)", "Taux\n(IRS)", "Change\n(NC)", "Change\n(FWD)"]
    vals_sig = [metriques_taux.sigma_nc/1e6, metriques_taux.sigma_c/1e6,
                metriques_fx.sigma_nc/1e6, metriques_fx.sigma_c/1e6]
    cols_sig = [PLOT.c_nc, PLOT.c_irs, PLOT.c_nc, PLOT.c_fwd]
    bars = ax.bar(labels_sig, vals_sig, color=cols_sig, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals_sig):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, color=PLOT.txt)
    ax.set_title("Écart-type des Cash Flows\n(σ_CF ≈ 0 = couverture parfaite)", color=PLOT.txt, fontsize=10)
    ax.set_ylabel("σ_CF (M MAD)", color=PLOT.txt)

    # AX4 — Distributions taux
    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax)
    ax.hist(cout_dette_nc/1e6, bins=80, alpha=0.60, color=PLOT.c_nc, density=True,
            edgecolor="none", label="Non couvert")
    ax.axvline(-metriques_taux.var99_nc/1e6, color=PLOT.c_var, lw=2, ls="--",
               label=f"VaR 99% = {metriques_taux.var99_nc/1e6:.3f}M")
    ax.axvline(-metriques_taux.es99_nc/1e6, color=PLOT.c_es, lw=1.5, ls="--",
               label=f"ES  99% = {metriques_taux.es99_nc/1e6:.3f}M")
    if cout_irs.std() < 1e-6:
        ax.axvline(cout_irs.mean()/1e6, color=PLOT.c_irs, lw=3,
                   label=f"IRS = {cout_irs.mean()/1e6:.3f}M (certain)")
    else:
        ax.hist(cout_irs/1e6, bins=30, alpha=0.90, color=PLOT.c_irs, density=True,
                edgecolor="none", label=f"IRS = {cout_irs.mean()/1e6:.3f}M (certain)")
    ax.set_title("Distribution Coût Taux\nAvant vs Après IRS", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX5 — Distributions change
    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax)
    ax.hist(cout_fx_nc/1e6, bins=80, alpha=0.60, color=PLOT.c_nc, density=True,
            edgecolor="none", label="Non couvert")
    ax.axvline(-metriques_fx.var99_nc/1e6, color=PLOT.c_var, lw=2, ls="--",
               label=f"VaR 99% = {metriques_fx.var99_nc/1e6:.3f}M")
    ax.axvline(-metriques_fx.es99_nc/1e6, color=PLOT.c_es, lw=1.5, ls="--",
               label=f"ES  99% = {metriques_fx.es99_nc/1e6:.3f}M")
    if cout_fwd.std() < 1e-6:
        ax.axvline(cout_fwd.mean()/1e6, color=PLOT.c_fwd, lw=3,
                   label=f"Forward = {cout_fwd.mean()/1e6:.3f}M (certain)")
    else:
        ax.hist(cout_fwd/1e6, bins=30, alpha=0.90, color=PLOT.c_fwd, density=True,
                edgecolor="none", label=f"Forward = {cout_fwd.mean()/1e6:.3f}M (certain)")
    ax.set_title("Distribution Coût FX\nAvant vs Après Forward", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Coût total (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX6 — Tableau synthétique
    ax = fig.add_subplot(gs[1, 2])
    _style_ax(ax)
    ax.axis("off")
    data_table = [
        ["Métrique", "IRS", "Forward"],
        ["η (réduct. var.)", f"{metriques_taux.eta*100:.2f}%", f"{metriques_fx.eta*100:.2f}%"],
        ["VaR 99% avant", f"{metriques_taux.var99_nc/1e6:.3f}M", f"{metriques_fx.var99_nc/1e6:.3f}M"],
        ["VaR 99% après", f"{metriques_taux.var99_c/1e6:.4f}M", f"{metriques_fx.var99_c/1e6:.4f}M"],
        ["ΔVaR%", f"{metriques_taux.delta_var99_pct:.2f}%", f"{metriques_fx.delta_var99_pct:.2f}%"],
        ["ES 99% avant", f"{metriques_taux.es99_nc/1e6:.3f}M", f"{metriques_fx.es99_nc/1e6:.3f}M"],
        ["ES 99% après", f"{metriques_taux.es99_c/1e6:.4f}M", f"{metriques_fx.es99_c/1e6:.4f}M"],
        ["σ_CF avant", f"{metriques_taux.sigma_nc/1e6:.3f}M", f"{metriques_fx.sigma_nc/1e6:.3f}M"],
        ["σ_CF après", f"≈{metriques_taux.sigma_c:.0f} MAD", f"≈{metriques_fx.sigma_c:.0f} MAD"],
        ["Taux / Prix", f"K={K*100:.4f}%", f"F₀={f0:.4f}"],
    ]
    tbl = ax.table(cellText=data_table[1:], colLabels=data_table[0],
                   cellLoc="center", loc="center", colWidths=[0.42, 0.29, 0.29])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if r == 0:
            cell.set_facecolor("#1F3864")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1a2535")
            cell.set_text_props(color=PLOT.txt)
        else:
            cell.set_facecolor("#0f1827")
            cell.set_text_props(color=PLOT.txt)
    ax.set_title("Tableau Comparatif — IRS vs Forward", color=PLOT.txt, fontsize=10, pad=12)

    fig.suptitle("Figure 5.3.2.3 — Évaluation des Stratégies de Couverture\n"
                 "Métriques : η, VaR, ES, σ_CF  (IRS et Forward de Change)",
                 color="white", fontsize=12, fontweight="bold")
    return _sauvegarder(fig, "fig_5323_metriques.png")


# =============================================================================
# FIGURE 5.3.3 — RISQUE DE CONTREPARTIE
# =============================================================================

def tracer_risque_contrepartie(
    exposition_brute: np.ndarray,
    exposition_nette: np.ndarray,
    ead_sans_collateral: np.ndarray,
    ead_avec_collateral: np.ndarray,
    epe_swap: np.ndarray,
    epe_fwd: np.ndarray,
    reduction_netting_pct: float,
    reduction_collateral_pct: float,
    taux_collateral: float,
) -> str:
    """Figure 5.3.3 — Risque de contrepartie : netting, collatéral, EPE."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor(PLOT.bg)

    # AX1 — Distribution des expositions (brute → nette → EAD)
    ax = axes[0, 0]
    _style_ax(ax)
    ax.hist(exposition_brute/1e6, bins=70, alpha=0.65, color=PLOT.c_var, density=True,
            edgecolor="none", label=f"Exposition brute  moy={exposition_brute.mean()/1e6:.2f}M")
    ax.hist(exposition_nette/1e6, bins=70, alpha=0.65, color=PLOT.c_mn, density=True,
            edgecolor="none", label=f"Après netting     moy={exposition_nette.mean()/1e6:.2f}M")
    ax.hist(ead_avec_collateral/1e6, bins=70, alpha=0.65, color=PLOT.c_irs, density=True,
            edgecolor="none", label=f"EAD (collatéral)  moy={ead_avec_collateral.mean()/1e6:.2f}M")
    ax.set_title(f"Réduction de l'Exposition par Étapes\nNetting (−{reduction_netting_pct:.1f}%) "
                 f"→ Collatéral {taux_collateral*100:.0f}% (−{reduction_collateral_pct:.1f}%)",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Exposition (M MAD)", color=PLOT.txt)
    ax.set_ylabel("Densité", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX2 — EPE par période (IRS et Forward)
    ax = axes[0, 1]
    _style_ax(ax)
    mois_epe = np.arange(1, len(epe_swap) + 1)
    ax.plot(mois_epe, epe_swap/1e3, color=PLOT.c_irs, lw=2.5, label="EPE — IRS (swap de taux)")
    n_fwd = len(epe_fwd)
    ax.plot(mois_epe[:n_fwd], epe_fwd[:n_fwd]/1e3, color=PLOT.c_fwd, lw=2.5, ls="--",
            label="EPE — Forward de change")
    ax.set_title("Exposition Positive Attendue (EPE) par Période\n"
                 r"EPE_t = E[max(V_t, 0)]", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Mois", color=PLOT.txt)
    ax.set_ylabel("EPE (K MAD)", color=PLOT.txt)
    ax.legend(fontsize=9, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # AX3 — Boxplot comparatif des expositions
    ax = axes[1, 0]
    _style_ax(ax)
    data_bp = [
        exposition_brute/1e6,
        exposition_nette/1e6,
        ead_sans_collateral/1e6,
        ead_avec_collateral/1e6,
    ]
    bp = ax.boxplot(data_bp, patch_artist=True, notch=False,
                    medianprops=dict(color=PLOT.txt, lw=2))
    colors_bp = [PLOT.c_var, PLOT.c_mn, PLOT.c_str, PLOT.c_irs]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(["Brute", "Nette\n(Netting)", "Sans\nCollat.", "Avec\nCollat."],
                       color=PLOT.txt, fontsize=8)
    ax.set_title("Comparaison des Expositions (Boxplot)\nEffet cumulé des mécanismes de réduction",
                 color=PLOT.txt, fontsize=10)
    ax.set_ylabel("Exposition (M MAD)", color=PLOT.txt)

    # AX4 — Profil de réduction (barchart des moyennes)
    ax = axes[1, 1]
    _style_ax(ax)
    noms = ["Brute", "Nette\n(Netting)", "EAD\n(sans collat.)", "EAD\n(avec collat.)"]
    vals = [exposition_brute.mean()/1e6, exposition_nette.mean()/1e6,
            ead_sans_collateral.mean()/1e6, ead_avec_collateral.mean()/1e6]
    cols_bar = [PLOT.c_var, PLOT.c_mn, PLOT.c_str, PLOT.c_irs]
    bars = ax.bar(noms, vals, color=cols_bar, alpha=0.85, width=0.6)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}M", ha="center", va="bottom", fontsize=9, color=PLOT.txt)
    ax.set_title("Exposition Moyenne — Avant et Après Atténuation\n"
                 "Netting ISDA + Collatéralisation CSA", color=PLOT.txt, fontsize=10)
    ax.set_ylabel("Exposition moyenne (M MAD)", color=PLOT.txt)

    fig.suptitle("Figure 5.3.3 — Risque de Contrepartie\n"
                 "Modélisation des effets de netting et de mise en garantie (collatéral)",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _sauvegarder(fig, "fig_533_risque_contrepartie.png")
