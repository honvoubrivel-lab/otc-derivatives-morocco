"""
=============================================================================
utils/console.py — Utilitaires d'affichage console
=============================================================================
Fonctions d'affichage structuré pour les sorties console.
Chaque section du projet produit des résumés lisibles et organisés.
=============================================================================
"""


# =============================================================================
# SÉPARATEURS ET TITRES
# =============================================================================

def titre_section(titre: str, largeur: int = 70) -> None:
    """Affiche un titre de section principal avec bordure."""
    print("\n" + "=" * largeur)
    print(f"  {titre}")
    print("=" * largeur)


def sous_titre(titre: str, largeur: int = 65) -> None:
    """Affiche un sous-titre avec tirets."""
    print(f"\n{'─' * largeur}")
    print(f"  {titre:^{largeur - 2}}")
    print(f"{'─' * largeur}")


def separateur(largeur: int = 65) -> None:
    """Affiche une ligne séparatrice."""
    print("─" * largeur)


def etape(num: int, description: str) -> None:
    """Affiche une étape numérotée."""
    print(f"\n  ── ÉTAPE {num} : {description} ──")


# =============================================================================
# AFFICHAGE DES PARAMÈTRES
# =============================================================================

def afficher_parametres_vasicek(kappa: float, theta: float, sigma: float, r0: float) -> None:
    """Affiche les paramètres calibrés du modèle de Vasicek."""
    sous_titre("PARAMÈTRES VASICEK (MLE)")
    print(f"    κ (vitesse de retour à la moyenne) : {kappa:.4f}  "
          f"→ demi-vie = {0.693/kappa:.2f} ans")
    print(f"    θ (taux d'équilibre long terme)    : {theta*100:.4f}%")
    print(f"    σ (volatilité instantanée)         : {sigma*100:.4f}%")
    print(f"    r₀ (dernier taux BAM observé)      : {r0*100:.4f}%")


def afficher_parametres_gbm(mu: float, sigma_s: float, s0: float, f0: float) -> None:
    """Affiche les paramètres calibrés du modèle GBM."""
    sous_titre("PARAMÈTRES GBM (USD/MAD)")
    print(f"    μ  = r_d - r_f  : {mu*100:.4f}%  (drift CIP)")
    print(f"    σ_S             : {sigma_s*100:.2f}%  (volatilité annualisée)")
    print(f"    S₀              : {s0:.4f} MAD/USD  (taux spot)")
    print(f"    F₀              : {f0:.4f} MAD/USD  (taux forward 1 an)")


# =============================================================================
# AFFICHAGE DES MÉTRIQUES DE RISQUE
# =============================================================================

def afficher_tableau_metriques(
    label_nc: str,
    label_c: str,
    moy_nc: float,
    moy_c: float,
    sigma_nc: float,
    sigma_c: float,
    var95_nc: float,
    var95_c: float,
    var99_nc: float,
    var99_c: float,
    es99_nc: float,
    es99_c: float,
    eta: float,
    delta_var99: float,
) -> None:
    """
    Affiche un tableau comparatif des métriques de risque avant/après couverture.

    Toutes les valeurs sont en MAD (non divisées par 1e6).
    L'affichage convertit en M MAD automatiquement.
    """
    print(f"\n  {'─'*63}")
    print(f"  {'COMPARAISON RISQUE':^63}")
    print(f"  {'─'*63}")
    print(f"  {'Métrique':<30} {label_nc:>12} {label_c:>12} {'Réduction':>10}")
    print(f"  {'─'*63}")

    def ligne(nom, avant, apres, suffixe=""):
        diff = apres - avant
        print(f"  {nom:<30} {avant/1e6:>12.4f} {apres/1e6:>12.4f} {diff/1e6:>+10.4f}{suffixe}")

    ligne("Moyenne coût (M MAD)", moy_nc, moy_c)
    print(f"  {'Écart-type (M MAD)':<30} {sigma_nc/1e6:>12.4f} {sigma_c/1e6:>12.6f} "
          f"{'η='+str(round(eta*100,2))+'%':>10}")
    ligne("VaR 95% (M MAD)", var95_nc, var95_c)
    ligne("VaR 99% (M MAD)", var99_nc, var99_c)
    ligne("ES  99% (M MAD)", es99_nc, es99_c)
    print(f"\n  Réduction de variance η    : {eta*100:.4f}%  "
          f"({'couverture parfaite ✓' if eta > 0.999 else 'couverture partielle'})")
    print(f"  Réduction de VaR 99% ΔVaR : {delta_var99:.2f}%")


# =============================================================================
# AFFICHAGE RÉSUMÉS
# =============================================================================

def afficher_resume_irs(
    K: float,
    cout_irs_theorique: float,
    eta_irs: float,
    var99_nc: float,
    var99_irs: float,
    delta_var99_irs: float,
    sigma_dette_nc: float,
    sigma_irs: float,
) -> None:
    """Résumé final du swap de taux d'intérêt."""
    sous_titre("RÉSUMÉ IRS (Swap de Taux)")
    print(f"  Taux fixe K              : {K*100:.4f}%")
    print(f"  Charge certaine 3 ans    : {cout_irs_theorique/1e6:.4f} M MAD")
    print(f"  η (réduction variance)   : {eta_irs*100:.4f}%")
    print(f"  VaR 99% avant / après    : {var99_nc/1e6:.4f}M / {var99_irs/1e6:.6f}M MAD")
    print(f"  ΔVaR 99%                 : {delta_var99_irs:.2f}%")
    print(f"  σ_CF avant / après       : {sigma_dette_nc/1e6:.4f}M / {sigma_irs:.4f} MAD")


def afficher_resume_forward(
    F0: float,
    cout_fwd_theorique: float,
    cout_budget: float,
    eta_fwd: float,
    var99_nc: float,
    var99_fwd: float,
    delta_var99_fwd: float,
    sigma_fx_nc: float,
    sigma_fwd: float,
) -> None:
    """Résumé final du contrat forward de change."""
    sous_titre("RÉSUMÉ Forward de Change")
    print(f"  Taux forward F₀          : {F0:.6f} MAD/USD")
    print(f"  Coût certain 1 an        : {cout_fwd_theorique/1e6:.4f} M MAD")
    print(f"  vs Budget (S₀)           : {(cout_fwd_theorique - cout_budget)/1e3:+.2f}K MAD")
    print(f"  η (réduction variance)   : {eta_fwd*100:.4f}%")
    print(f"  VaR 99% avant / après    : {var99_nc/1e6:.4f}M / {var99_fwd/1e6:.6f}M MAD")
    print(f"  ΔVaR 99%                 : {delta_var99_fwd:.2f}%")
    print(f"  σ_CF avant / après       : {sigma_fx_nc/1e6:.4f}M / {sigma_fwd:.4f} MAD")
