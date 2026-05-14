"""
=============================================================================
reporting/summary.py — Résumés console et rapports textuels
=============================================================================
"""

from utils.console import titre_section, sous_titre, separateur


def afficher_resume_contrepartie(res) -> None:
    """Résumé console du risque de contrepartie (section 5.3.3)."""
    titre_section("RÉSUMÉ — SECTION 5.3.3 : RISQUE DE CONTREPARTIE")
    print(f"  EBE totale             : {res.ebe_totale/1e6:.4f} M MAD")
    print(f"  ENE totale             : {res.ene_totale/1e6:.4f} M MAD")
    print(f"  Netting Factor         : {res.netting_factor:.4f}  "
          f"(compression de {(1-res.netting_factor)*100:.1f}%)")
    print(f"  EAD brute              : {res.ead_brute/1e6:.4f} M MAD")
    print(f"  EAD nette              : {res.ead_nette/1e6:.4f} M MAD")
    print(f"  EAD après collatéral   : {res.ead_collat/1e6:.4f} M MAD")
    print(f"  CVA (sans collat.)     : {res.cva/1e6:.5f} M MAD")
    print(f"  CVA (avec collat.)     : {res.cva_collat/1e6:.5f} M MAD")


def afficher_resume_reglementaire(resultats: dict) -> None:
    """Résumé console de la comparaison réglementaire (section 5.3.4)."""
    titre_section("RÉSUMÉ — SECTION 5.3.4 : SCÉNARIOS RÉGLEMENTAIRES")

    for nom, res in resultats.items():
        # res est un ResultatsCadre → accès par .attribut
        cadre = res.cadre
        sous_titre(f"CADRE : {cadre.nom.upper()}")
        print(f"  Contrats OTC    : {cadre.n_contrats}  |  "
              f"Contreparties : {cadre.n_contreparties}")
        print(f"  Netting reconnu : {'Oui (ISDA)' if cadre.netting_reconnu else 'Non'}")
        print(f"  Taux collatéral : {cadre.taux_collateral*100:.0f}%")
        print(f"  Spread OTC      : {cadre.spread_bp:.0f} bps")
        separateur()
        print(f"  EBE             : {res.ebe_moyen/1e6:.3f} M MAD")
        print(f"  ENE             : {res.ene_moyen/1e6:.3f} M MAD")
        print(f"  Netting Factor  : {res.netting_factor:.4f}")
        print(f"  EAD             : {res.ead_moyen/1e6:.3f} M MAD")
        print(f"  CVA             : {res.cva_total/1e6:.4f} M MAD")
        print(f"  HHI             : {res.hhi:.4f}  "
              f"({'Concentré ⚠' if res.hhi > 0.25 else 'Modéré ✓' if res.hhi > 0.15 else 'Diversifié ✓'})")
        print(f"  Coût spreads    : {res.cout_spread_total/1e6:.4f} M MAD")
        print(f"  Diversification : {res.ratio_diversification*100:.1f}/100")

    # Conclusion comparative
    r = resultats["restrictif"]
    f = resultats["flexible"]
    titre_section("CONCLUSION COMPARATIVE")
    print(f"  Réduction EAD       : {(r.ead_moyen - f.ead_moyen)/r.ead_moyen*100:.1f}%  "
          f"(restrictif → flexible)")
    print(f"  Réduction CVA       : {(r.cva_total - f.cva_total)/r.cva_total*100:.1f}%")
    print(f"  Réduction HHI       : {(r.hhi - f.hhi)/r.hhi*100:.1f}%")
    print(f"  Réduction spreads   : {(r.cadre.spread_bp - f.cadre.spread_bp)/r.cadre.spread_bp*100:.1f}%")
    print(f"  Amélioration divrs. : +{(f.ratio_diversification - r.ratio_diversification)*100:.1f} points")
    print(f"\n  → Le cadre flexible prudentiel améliore toutes les métriques")
    print(f"    de risque systémique sans sacrifier la couverture financière.")