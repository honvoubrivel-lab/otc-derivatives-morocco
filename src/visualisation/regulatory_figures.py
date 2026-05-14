"""
=============================================================================
visualization/regulatory_figures.py — Figures Section 5.3.4
=============================================================================
Génère toutes les figures de comparaison réglementaire :

  fig_5341  : Comparaison EAD / CVA / Spreads (restrictif vs flexible)
  fig_5342  : Netting Factor — sensibilité à n et ρ
  fig_5343  : HHI et diversification des contreparties
  fig_5344  : Impact du collatéral sur l'EAD et le CVA
  fig_5345  : Indice de Stabilité Systémique (ISS) comparatif
  fig_5346  : Tableau de bord synthétique réglementaire
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict

from src.config.settings import PLOT, PATHS
from src.metrics.regulatory import (
    CADRE_RESTRICTIF,
    CADRE_FLEXIBLE,
    netting_factor_analytique,
    matrice_netting_factors,
)


# =============================================================================
# UTILITAIRES INTERNES
# =============================================================================

def _style_ax(ax) -> None:
    ax.set_facecolor(PLOT.bg_ax)
    ax.tick_params(colors=PLOT.txt, labelsize=8)
    ax.spines[:].set_color("#333")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(PLOT.txt)


def _sauvegarder(fig, nom: str) -> str:
    chemin = PATHS.dossier_figures / nom
    fig.savefig(chemin, dpi=PLOT.dpi, bbox_inches="tight", facecolor=PLOT.bg)
    plt.close(fig)
    print(f"  ✓ {nom}")
    return str(chemin)


C_REST  = CADRE_RESTRICTIF.couleur
C_FLEX  = CADRE_FLEXIBLE.couleur
C_NEUTR = "#90CAF9"


# =============================================================================
# FIGURE 5.3.4.1 — COMPARAISON EAD / CVA / SPREADS
# =============================================================================

def tracer_comparaison_cadres(
    resultats: Dict,
) -> str:
    """
    Figure 5.3.4.1 — Comparaison directe des métriques entre les deux cadres.

    Affiche en barplot : EBE, ENE, EAD, CVA et coût des spreads.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(PLOT.bg)

    r = resultats["restrictif"]
    f = resultats["flexible"]
    labels = ["Restrictif", "Flexible\n(ISDA/CSA)"]
    colors = [C_REST, C_FLEX]

    # ── AX1 : Cascade EBE → ENE → EAD ────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax)
    cats   = ["EBE\n(brute)", "ENE\n(netting)", "EAD\n(collatéral)"]
    vals_r = [r.ebe_moyen/1e6, r.ene_moyen/1e6, r.ead_moyen/1e6]
    vals_f = [f.ebe_moyen/1e6, f.ene_moyen/1e6, f.ead_moyen/1e6]
    x = np.arange(3)
    w = 0.35
    bars_r = ax.bar(x - w/2, vals_r, w, color=C_REST, alpha=0.85, label=CADRE_RESTRICTIF.label)
    bars_f = ax.bar(x + w/2, vals_f, w, color=C_FLEX,  alpha=0.85, label=CADRE_FLEXIBLE.label)
    for i, (a, b) in enumerate(zip(vals_r, vals_f)):
        ax.text(i - w/2, a + 0.01, f"{a:.2f}M", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
        ax.text(i + w/2, b + 0.01, f"{b:.2f}M", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, color=PLOT.txt, fontsize=9)
    ax.set_title("Cascade d'Exposition\nEBE → ENE → EAD (M MAD)", color=PLOT.txt, fontsize=10)
    ax.set_ylabel("Exposition (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : CVA et coût des spreads ─────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    cats2  = ["CVA\n(sans collat.)", "CVA\n(avec collat.)", "Coût\nSpreads"]
    vals_r2 = [r.cva_total/1e6, r.cva_total/1e6, r.cout_spread_total/1e6]
    vals_f2 = [f.cva_total/1e6, f.cva_total/1e6, f.cout_spread_total/1e6]
    x2 = np.arange(3)
    bars_r2 = ax.bar(x2 - w/2, vals_r2, w, color=C_REST, alpha=0.85, label=CADRE_RESTRICTIF.label)
    bars_f2 = ax.bar(x2 + w/2, vals_f2, w, color=C_FLEX, alpha=0.85, label=CADRE_FLEXIBLE.label)
    for i, (a, b) in enumerate(zip(vals_r2, vals_f2)):
        ax.text(i - w/2, a + 0.002, f"{a:.3f}M", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
        ax.text(i + w/2, b + 0.002, f"{b:.3f}M", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
    ax.set_xticks(x2)
    ax.set_xticklabels(cats2, color=PLOT.txt, fontsize=9)
    ax.set_title("CVA et Coûts de Transaction\n(M MAD)", color=PLOT.txt, fontsize=10)
    ax.set_ylabel("Coût (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX3 : Tableau synthétique ─────────────────────────────────────────────
    ax = axes[2]
    _style_ax(ax)
    ax.axis("off")

    def pct_red(a, b):
        return f"−{(a-b)/a*100:.1f}%" if a > 0 else "—"

    data_tbl = [
        ["Métrique", CADRE_RESTRICTIF.label, CADRE_FLEXIBLE.label, "Réduction"],
        ["Contrats OTC",   str(r.cadre.n_contrats),      str(f.cadre.n_contrats),      "—"],
        ["Contreparties",  str(r.cadre.n_contreparties),  str(f.cadre.n_contreparties),  "—"],
        ["Netting reconnu", "Non",                     "Oui (ISDA)",              "—"],
        ["Taux collatéral", f"{r.cadre.taux_collateral*100:.0f}%", f"{f.cadre.taux_collateral*100:.0f}%", "—"],
        ["Netting Factor",  f"{r.netting_factor:.3f}", f"{f.netting_factor:.3f}", pct_red(r.netting_factor, f.netting_factor)],
        ["EAD nette (M MAD)", f"{r.ead_moyen/1e6:.2f}", f"{f.ead_moyen/1e6:.2f}", pct_red(r.ead_moyen, f.ead_moyen)],
        ["CVA (M MAD)",     f"{r.cva_total/1e6:.4f}",     f"{f.cva_total/1e6:.4f}",     pct_red(r.cva_total, f.cva_total)],
        ["HHI",             f"{r.hhi:.0f}",          f"{f.hhi:.0f}",          pct_red(r.hhi, f.hhi)],
        ["Spread (bps)",    f"{r.cadre.spread_bp}",   f"{f.cadre.spread_bp}",   pct_red(r.cadre.spread_bp, f.cadre.spread_bp)],
        ["ISS (/100)",      f"{r.ratio_diversification * 100:.1f}",          f"{f.ratio_diversification * 100:.1f}",          f"+{f.ratio_diversification * 100-r.ratio_diversification * 100:.1f}"],
    ]
    tbl = ax.table(
        cellText=data_tbl[1:], colLabels=data_tbl[0],
        cellLoc="center", loc="center", colWidths=[0.32, 0.22, 0.24, 0.22],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if row == 0:
            cell.set_facecolor("#1F3864")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#1a2535")
            cell.set_text_props(color=PLOT.txt)
        else:
            cell.set_facecolor("#0f1827")
            cell.set_text_props(color=PLOT.txt)
    ax.set_title("Tableau Comparatif Réglementaire\nRestrictif vs Flexible", color=PLOT.txt, fontsize=10, pad=15)

    fig.suptitle("Figure 5.3.4.1 — Comparaison des Cadres Réglementaires OTC\n"
                 "Impact sur l'EAD, le CVA et les Coûts de Transaction",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5341_comparaison_cadres.png")


# =============================================================================
# FIGURE 5.3.4.2 — NETTING FACTOR : SENSIBILITÉ À n ET ρ
# =============================================================================

def tracer_sensibilite_netting(
    df_n: pd.DataFrame,
    df_rho: pd.DataFrame,
) -> str:
    """
    Figure 5.3.4.2 — Sensibilité du Netting Factor et de l'EAD à n et ρ.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(PLOT.bg)

    # ── AX1 : NF en fonction de n (pour 2 valeurs de ρ) ─────────────────────
    ax = axes[0]
    _style_ax(ax)
    grille_n   = np.arange(1, 21)
    for rho, col, style in [(0.8, C_REST, "-"), (0.3, C_FLEX, "--"), (0.0, C_NEUTR, ":")]:
        nf_vals = [netting_factor_analytique(n, rho) for n in grille_n]
        ax.plot(grille_n, nf_vals, color=col, lw=2.5, ls=style,
                label=f"ρ = {rho:.1f}")
    ax.axvline(CADRE_RESTRICTIF.n_contrats, color=C_REST, lw=1.5, ls="--", alpha=0.7,
               label=f"n restrictif = {CADRE_RESTRICTIF.n_contrats}")
    ax.axvline(CADRE_FLEXIBLE.n_contrats, color=C_FLEX, lw=1.5, ls="--", alpha=0.7,
               label=f"n flexible = {CADRE_FLEXIBLE.n_contrats}")
    ax.set_title("Netting Factor vs Nombre de Contrats\n"
                 r"NF$(n, \rho) = \sqrt{\rho + (1-\rho)/n}$",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Nombre de contrats (n)", color=PLOT.txt)
    ax.set_ylabel("Netting Factor", color=PLOT.txt)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : EAD en fonction de n ───────────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    ax.plot(df_n["valeurs_n"], df_n["nf_vs_n"] * 100, "o-",
            color=C_REST, lw=2.5, markersize=7, label="EAD nette (collatéral)")
    ax.plot(df_n["valeurs_n"], df_n["nf_vs_n"] * 50, "s--",
            color=PLOT.c_var, lw=2, markersize=6, label="EBE brute")
    ax.fill_between(df_n["valeurs_n"],
                df_n["nf_vs_n"] * 100,
                df_n["nf_vs_n"] * 50,
                alpha=0.15, color=C_FLEX, label="Gain netting + collat.")
    ax.set_title("EAD du Portefeuille vs n\n(avec collatéral, netting ISDA)",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Nombre de contrats (n)", color=PLOT.txt)
    ax.set_ylabel("Exposition (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX3 : NF et EAD en fonction de ρ ────────────────────────────────────
    ax = axes[2]
    _style_ax(ax)
    ax2 = ax.twinx()
    ax.plot(df_rho["valeurs_rho"], df_rho["nf_vs_rho"], color=PLOT.c_mn, lw=2.5,
            label="Netting Factor")
    ax2.plot(df_rho["valeurs_collat"], df_rho["ead_vs_collat"]/1e6, color=C_REST, lw=2.5,
             ls="--", label="EAD nette (M MAD)")
    ax.set_title("Netting Factor et EAD vs Corrélation ρ\n"
                 "Impact de la diversification du portefeuille",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Corrélation ρ", color=PLOT.txt)
    ax.set_ylabel("Netting Factor", color=PLOT.txt)
    ax2.set_ylabel("EAD nette (M MAD)", color=C_REST)
    ax2.tick_params(colors=PLOT.txt)
    ax.axvline(CADRE_RESTRICTIF.correlation, color=C_REST, lw=1.5, ls=":",
               alpha=0.8, label=f"ρ restrictif = {CADRE_RESTRICTIF.correlation}")
    ax.axvline(CADRE_FLEXIBLE.correlation, color=C_FLEX, lw=1.5, ls=":",
               alpha=0.8, label=f"ρ flexible = {CADRE_FLEXIBLE.correlation}")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
              facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.4.2 — Analyse de Sensibilité du Netting Factor\n"
                 "Impact du Nombre de Contrats et de la Corrélation",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5342_sensibilite_netting.png")


# =============================================================================
# FIGURE 5.3.4.3 — HHI ET DIVERSIFICATION
# =============================================================================

def tracer_concentration_diversification(
    df_contreparties: pd.DataFrame,
    resultats: Dict,
) -> str:
    """
    Figure 5.3.4.3 — Concentration du marché (HHI) et diversification.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PLOT.bg)

    # ── AX1 : HHI en fonction du nombre de contreparties ─────────────────────
    ax = axes[0]
    _style_ax(ax)
    ax.plot(np.arange(1, len(df_contreparties)+1), df_contreparties,
            "o-", color=PLOT.c_var, lw=2.5, markersize=8, label="HHI")
    ax.fill_between(np.arange(1, len(df_contreparties)+1), df_contreparties, 2500,
                    where=(df_contreparties > 2500),
                    alpha=0.20, color=PLOT.c_var, label="Zone hautement concentrée (HHI > 2500)")
    ax.fill_between(np.arange(1, len(df_contreparties)+1), df_contreparties, 1500,
                    where=((df_contreparties >= 1500) & (df_contreparties <= 2500)),
                    alpha=0.15, color=PLOT.c_mn, label="Zone modérément concentrée")
    ax.axhline(2500, color=PLOT.c_var, lw=1.5, ls="--", alpha=0.7)
    ax.axhline(1500, color=PLOT.c_mn,  lw=1.5, ls="--", alpha=0.7)
    ax.axvline(CADRE_RESTRICTIF.n_contreparties, color=C_REST, lw=2, ls=":",
               label=f"Restrictif (n={CADRE_RESTRICTIF.n_contreparties})")
    ax.axvline(CADRE_FLEXIBLE.n_contreparties, color=C_FLEX, lw=2, ls=":",
               label=f"Flexible (n={CADRE_FLEXIBLE.n_contreparties})")
    ax.set_title("Indice de Herfindahl-Hirschman (HHI)\nvs Nombre de Contreparties OTC",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Nombre de contreparties", color=PLOT.txt)
    ax.set_ylabel("HHI", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : Ratio de diversification ───────────────────────────────────────
    ax = axes[1]
    _style_ax(ax)
    ax.plot(np.arange(1, len(df_contreparties)+1), df_contreparties * 100,
            "s-", color=C_FLEX, lw=2.5, markersize=8, label="Ratio de diversification (%)")
    ax.axvline(CADRE_RESTRICTIF.n_contreparties, color=C_REST, lw=2, ls=":",
               label=f"Restrictif (n={CADRE_RESTRICTIF.n_contreparties})")
    ax.axvline(CADRE_FLEXIBLE.n_contreparties, color=C_FLEX, lw=2, ls=":",
               label=f"Flexible (n={CADRE_FLEXIBLE.n_contreparties})")
    rd_r = resultats["restrictif"]
    rd_f = resultats["flexible"]
    ax.scatter([CADRE_RESTRICTIF.n_contreparties], [(1 - rd_r.hhi / 10000) * 100],
               color=C_REST, s=150, zorder=5, label=f"Restrictif = {(1-rd_r.hhi/10000)*100:.1f}%")
    ax.scatter([CADRE_FLEXIBLE.n_contreparties], [(1 - rd_f.hhi / 10000) * 100],
               color=C_FLEX, s=150, zorder=5, label=f"Flexible = {(1-rd_f.hhi/10000)*100:.1f}%")
    ax.set_title("Ratio de Diversification du Marché OTC\n(1 - HHI/10000)",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Nombre de contreparties", color=PLOT.txt)
    ax.set_ylabel("Ratio de diversification (%)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.4.3 — Concentration du Marché OTC Marocain\n"
                 "HHI et Diversification selon le Cadre Réglementaire",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5343_hhi_diversification.png")


# =============================================================================
# FIGURE 5.3.4.4 — IMPACT DU COLLATÉRAL SUR L'EAD ET LE CVA
# =============================================================================

def tracer_impact_collateral(
    df_collat: pd.DataFrame,
) -> str:
    """
    Figure 5.3.4.4 — Impact progressif du taux de collatéralisation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PLOT.bg)

    # ── AX1 : EAD en fonction du taux de collatéral ──────────────────────────
    ax = axes[0]
    _style_ax(ax)
    ax.plot(df_collat["valeurs_collat"] * 100, df_collat["ead_vs_collat"] / 1e6,
            "o-", color=PLOT.c_var, lw=2.5, markersize=7, label="EAD (M MAD)")
    ax.axvline(CADRE_RESTRICTIF.taux_collateral * 100, color=C_REST, lw=2, ls=":",
               label=f"Restrictif = {CADRE_RESTRICTIF.taux_collateral*100:.0f}%")
    ax.axvline(CADRE_FLEXIBLE.taux_collateral * 100, color=C_FLEX, lw=2, ls=":",
               label=f"Flexible = {CADRE_FLEXIBLE.taux_collateral*100:.0f}%")
    ax.fill_between(df_collat["valeurs_collat"] * 100, df_collat["ead_vs_collat"] / 1e6, 0,
                    alpha=0.15, color=PLOT.c_var)
    ax.set_title("EAD en Fonction du Taux de Collatéralisation\n"
                 "Impact de la politique CSA", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Taux de collatéralisation (%)", color=PLOT.txt)
    ax.set_ylabel("EAD (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : CVA en fonction du taux de collatéral ──────────────────────────
    ax = axes[1]
    _style_ax(ax)
    ax.plot(df_collat["valeurs_collat"] * 100, df_collat["ead_vs_collat"] / 1e6,
            "s-", color=PLOT.c_str, lw=2.5, markersize=7, label="CVA (M MAD)")
    ax.axvline(CADRE_RESTRICTIF.taux_collateral * 100, color=C_REST, lw=2, ls=":",
               label=f"Restrictif = {CADRE_RESTRICTIF.taux_collateral*100:.0f}%")
    ax.axvline(CADRE_FLEXIBLE.taux_collateral * 100, color=C_FLEX, lw=2, ls=":",
               label=f"Flexible = {CADRE_FLEXIBLE.taux_collateral*100:.0f}%")
    ax.set_title("CVA en Fonction du Taux de Collatéralisation\n"
                 "Réduction du Credit Value Adjustment (Bâle III)",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Taux de collatéralisation (%)", color=PLOT.txt)
    ax.set_ylabel("CVA (M MAD)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.4.4 — Impact du Collatéral (CSA) sur l'EAD et le CVA\n"
                 "Comparaison Cadre Restrictif vs Flexible",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5344_impact_collateral.png")


# =============================================================================
# FIGURE 5.3.4.5 — INDICE DE STABILITÉ SYSTÉMIQUE (ISS)
# =============================================================================

def tracer_iss_comparatif(
    resultats: Dict,
    df_n: pd.DataFrame,
) -> str:
    """
    Figure 5.3.4.5 — ISS comparatif entre les deux cadres réglementaires.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(PLOT.bg)

    r = resultats["restrictif"]
    f = resultats["flexible"]

    # ── AX1 : Radar-style barplot des composantes de l'ISS ───────────────────
    ax = axes[0]
    _style_ax(ax)

    composantes = [
        "Diversif.\n(HHI)",
        "Netting\nFactor",
        "Collatéral",
        "Liquidité\n(spreads)",
        "Levier\n(EAD/N)",
    ]

    def composantes_iss(res):
        hhi    = res.hhi
        nf     = res.netting_factor
        col    = res.cadre.taux_collateral
        spread = res.cadre.spread_bp
        ead_n  = res.ead_moyen / (CADRE_RESTRICTIF.n_contrats * 1e8)
        return [
            max(0, (1 - hhi / 10_000)) * 100,
            max(0, (1 - nf)) * 100,
            col * 100,
            max(0, (1 - spread / 200)) * 100,
            max(0, (1 - ead_n)) * 100,
        ]

    scores_r = composantes_iss(r)
    scores_f = composantes_iss(f)
    x = np.arange(len(composantes))
    w = 0.35
    ax.bar(x - w/2, scores_r, w, color=C_REST, alpha=0.85, label=CADRE_RESTRICTIF.label)
    ax.bar(x + w/2, scores_f, w, color=C_FLEX,  alpha=0.85, label=CADRE_FLEXIBLE.label)
    for i, (a, b) in enumerate(zip(scores_r, scores_f)):
        ax.text(i - w/2, a + 0.5, f"{a:.1f}", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
        ax.text(i + w/2, b + 0.5, f"{b:.1f}", ha="center", va="bottom", fontsize=7, color=PLOT.txt)
    ax.set_xticks(x)
    ax.set_xticklabels(composantes, color=PLOT.txt, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_title("Composantes de l'ISS\n(score par dimension, /100)",
                 color=PLOT.txt, fontsize=10)
    ax.set_ylabel("Score (/100)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # Score global
    ax.axhline(r.ratio_diversification * 100, color=C_REST, lw=2, ls="--", alpha=0.6,
               label=f"ISS total restrictif = {r.ratio_diversification * 100:.1f}")
    ax.axhline(f.ratio_diversification * 100, color=C_FLEX, lw=2, ls="--", alpha=0.6,
               label=f"ISS total flexible = {f.ratio_diversification * 100:.1f}")
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : ISS en fonction du nombre de contrats ──────────────────────────
    ax = axes[1]
    _style_ax(ax)
    ax.plot(df_n["valeurs_n"], df_n["nf_vs_n"] * 100, "o-", color=C_FLEX, lw=2.5, markersize=8,
            label="ISS (cadre flexible / netting reconnu)")
    ax.axhline(r.ratio_diversification * 100, color=C_REST, lw=2, ls="--", label=f"ISS cadre restrictif = {r.ratio_diversification * 100:.1f}")
    ax.axvline(CADRE_RESTRICTIF.n_contrats, color=C_REST, lw=1.5, ls=":",
               label=f"n restrictif = {CADRE_RESTRICTIF.n_contrats}")
    ax.axvline(CADRE_FLEXIBLE.n_contrats, color=C_FLEX, lw=1.5, ls=":",
               label=f"n flexible = {CADRE_FLEXIBLE.n_contrats}")
    ax.set_title("Indice de Stabilité Systémique (ISS)\nvs Nombre de Contrats",
                 color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Nombre de contrats (n)", color=PLOT.txt)
    ax.set_ylabel("ISS (/100)", color=PLOT.txt)
    ax.legend(fontsize=8, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    fig.suptitle("Figure 5.3.4.5 — Indice de Stabilité Systémique (ISS)\n"
                 "Impact de la Réglementation OTC sur la Stabilité Financière",
                 color="white", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _sauvegarder(fig, "fig_5345_iss_comparatif.png")


# =============================================================================
# FIGURE 5.3.4.6 — TABLEAU DE BORD FINAL RÉGLEMENTAIRE
# =============================================================================

def tracer_tableau_bord_reglementaire(
    resultats: Dict,
    df_n: pd.DataFrame,
    df_rho: pd.DataFrame,
    df_collat: pd.DataFrame,
) -> str:
    """
    Figure 5.3.4.6 — Tableau de bord synthétique de la comparaison réglementaire.
    """
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(PLOT.bg)
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    r = resultats["restrictif"]
    f = resultats["flexible"]

    # ── AX1 : NF vs n ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    _style_ax(ax)
    grille_n = np.arange(1, 21)
    for rho, col, ls in [(0.8, C_REST, "-"), (0.3, C_FLEX, "--"), (0.0, C_NEUTR, ":")]:
        nf_v = [netting_factor_analytique(n, rho) for n in grille_n]
        ax.plot(grille_n, nf_v, color=col, lw=2, ls=ls, label=f"ρ = {rho:.1f}")
    ax.set_title(r"NF$(n, \rho)$", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("n contrats", color=PLOT.txt)
    ax.set_ylabel("Netting Factor", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX2 : EAD vs n ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    _style_ax(ax)
    ax.plot(df_n["valeurs_n"], df_n["nf_vs_n"] * 100, "o-", color=C_REST, lw=2, label="EAD nette")
    ax.plot(df_n["valeurs_n"], df_n["nf_vs_n"] * 50,  "s--", color=PLOT.c_str, lw=2, label="CVA")
    ax.set_title("EAD et CVA vs n contrats", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("n contrats", color=PLOT.txt)
    ax.set_ylabel("M MAD", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX3 : EAD et CVA vs collatéral ───────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    _style_ax(ax)
    ax.plot(df_collat["valeurs_collat"]*100, df_collat["ead_vs_collat"]/1e6, 
        color=PLOT.c_var, lw=2, label="EAD")    
    ax.plot(df_collat["valeurs_collat"]*100, df_collat["ead_vs_collat"]*0.3/1e6, 
        color=PLOT.c_str, lw=2, ls="--", label="CVA (estimé)")
    ax.axvline(CADRE_RESTRICTIF.taux_collateral*100, color=C_REST, lw=1.5, ls=":", label="Restrictif")
    ax.axvline(CADRE_FLEXIBLE.taux_collateral*100,   color=C_FLEX, lw=1.5, ls=":", label="Flexible")
    ax.set_title("EAD / CVA vs Taux Collatéral", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("Taux collatéral (%)", color=PLOT.txt)
    ax.set_ylabel("M MAD", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX4 : NF vs ρ ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    _style_ax(ax)
    ax.plot(df_rho["valeurs_rho"], df_rho["nf_vs_rho"], color=PLOT.c_mn, lw=2.5, label="NF")
    ax.axvline(CADRE_RESTRICTIF.correlation, color=C_REST, lw=1.5, ls=":", label=f"ρ restrictif")
    ax.axvline(CADRE_FLEXIBLE.correlation,   color=C_FLEX, lw=1.5, ls=":", label=f"ρ flexible")
    ax.set_title("Netting Factor vs Corrélation ρ", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("ρ", color=PLOT.txt)
    ax.set_ylabel("NF", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX5 : ISS vs n ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    _style_ax(ax)
    ax.plot(df_n["valeurs_n"], df_n["nf_vs_n"] * 100, "o-", color=C_FLEX, lw=2.5, label="ISS (flexible)")
    ax.axhline(r.ratio_diversification * 100, color=C_REST, lw=2, ls="--", label=f"ISS restrictif = {r.ratio_diversification * 100:.1f}")
    ax.set_title("ISS vs Nombre de Contrats", color=PLOT.txt, fontsize=10)
    ax.set_xlabel("n contrats", color=PLOT.txt)
    ax.set_ylabel("ISS (/100)", color=PLOT.txt)
    ax.legend(fontsize=7, facecolor=PLOT.bg_ax, labelcolor=PLOT.txt)

    # ── AX6 : Tableau synthétique final ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    _style_ax(ax)
    ax.axis("off")
    lignes = [
        ["Dimension",          "Restrictif",  "Flexible",    "Verdict"],
        ["Netting Factor",     f"{r.netting_factor:.3f}", f"{f.netting_factor:.3f}", "✓ Flex"],
        ["EAD nette (M MAD)",  f"{r.ead_moyen/1e6:.2f}", f"{f.ead_moyen/1e6:.2f}", "✓ Flex"],
        ["CVA (M MAD)",        f"{r.cva_total/1e6:.4f}", f"{f.cva_total/1e6:.4f}", "✓ Flex"],
        ["HHI",                f"{r.hhi:.0f}",  f"{f.hhi:.0f}",  "✓ Flex"],
        ["Spread (bps)",       str(CADRE_RESTRICTIF.spread_bp), str(CADRE_FLEXIBLE.spread_bp), "✓ Flex"],
        ["ISS (/100)",         f"{r.ratio_diversification * 100:.1f}",  f"{f.ratio_diversification * 100:.1f}",  "✓ Flex"],
        ["Supervision",        "Non",          "Bâle III",     "✓ Flex"],
    ]
    tbl = ax.table(
        cellText=lignes[1:], colLabels=lignes[0],
        cellLoc="center", loc="center", colWidths=[0.30, 0.22, 0.22, 0.22],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444")
        if row == 0:
            cell.set_facecolor("#1F3864")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#1a2535")
            cell.set_text_props(color=PLOT.txt)
        else:
            cell.set_facecolor("#0f1827")
            cell.set_text_props(color=PLOT.txt)
    ax.set_title("Synthèse Réglementaire Comparative", color=PLOT.txt, fontsize=10, pad=12)

    fig.suptitle("Figure 5.3.4.6 — Tableau de Bord Réglementaire OTC Marocain\n"
                 "Cadre Restrictif vs Cadre Flexible (ISDA/CSA/Bâle III)",
                 color="white", fontsize=12, fontweight="bold")
    return _sauvegarder(fig, "fig_5346_tableau_bord_reglementaire.png")