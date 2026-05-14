"""
=============================================================================
settings.py — Configuration globale du projet
=============================================================================
Centralise tous les paramètres du projet pour garantir la cohérence
entre les modules et faciliter la reproductibilité scientifique.

Structure des paramètres :
  - AgentParams    : caractéristiques de l'agent économique
  - SimulationParams : paramètres Monte Carlo
  - MarketParams   : paramètres de marché (spread, SOFR, etc.)
  - PlotParams     : palette graphique et style
  - PathParams     : chemins de fichiers
=============================================================================
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Racine du projet ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# PARAMÈTRES DE L'AGENT ÉCONOMIQUE REPRÉSENTATIF
# =============================================================================

@dataclass(frozen=True)
class AgentParams:
    """
    Paramètres définissant l'agent économique :
      - Emprunteur de N MAD à taux variable sur T années
      - Importateur de X_annuel USD sur T_fx années
    """
    # ── Dette à taux variable ────────────────────────────────────────────────
    notionnel: float = 100_000_000   # Notionnel de la dette en MAD (100M MAD)
    maturite_dette: int = 3          # Horizon de la dette en années
    spread_bancaire: float = 0.015   # Spread commercial BAM (+150bp)

    # ── Importations en devises ──────────────────────────────────────────────
    importations_annuelles_usd: float = 5_000_000   # Importations annuelles (USD)
    horizon_couverture_fx: int = 1                   # Horizon de couverture FX (années)

    @property
    def importations_mensuelles_usd(self) -> float:
        """Montant mensuel des importations en USD."""
        return self.importations_annuelles_usd / 12

    @property
    def n_periodes_dette(self) -> int:
        """Nombre de périodes mensuelles sur la durée de la dette."""
        return self.maturite_dette * 12     # 36 mois

    @property
    def n_periodes_fx(self) -> int:
        """Nombre de périodes mensuelles sur l'horizon FX."""
        return self.horizon_couverture_fx * 12   # 12 mois


# =============================================================================
# PARAMÈTRES DE SIMULATION MONTE CARLO
# =============================================================================

@dataclass(frozen=True)
class SimulationParams:
    """
    Paramètres de la simulation Monte Carlo.
    Le pas de temps est mensuel (dt = 1/12) pour la dette et le change.
    Le modèle Vasicek est calibré au pas mensuel sur les données BAM.
    """
    n_simulations: int = 10_000   # Nombre de trajectoires Monte Carlo
    dt_mensuel: float = 1 / 12    # Pas de temps mensuel
    graine_aleatoire: int = 42    # Graine pour la reproductibilité


# =============================================================================
# PARAMÈTRES DE MARCHÉ (valeurs par défaut, écrasées par les données réelles)
# =============================================================================

@dataclass
class MarketParams:
    """
    Paramètres de marché. Ces valeurs constituent des repères par défaut ;
    elles sont systématiquement remplacées par les données téléchargées
    (FRED pour le SOFR, Yahoo Finance pour USD/MAD, fichier CSV pour BAM).
    """
    # ── Taux de change USD/MAD (spot) ────────────────────────────────────────
    taux_spot_usdmad: float = 10.05   # S₀ : dernier cours USD/MAD observé

    # ── Taux d'intérêt ───────────────────────────────────────────────────────
    sofr: float = 0.0530             # r_f : SOFR (taux USD de référence)

    # ── Volatilité implicite du GBM ──────────────────────────────────────────
    sigma_gbm_default: float = 0.035  # σ_S par défaut si données insuffisantes

    # ── Clé API FRED ────────────────────────────────────────────────────────
    fred_api_key: str = "1e13c6e8a9f214cafd4c16b8cae21c1f"

    # ── Ticker Yahoo Finance ─────────────────────────────────────────────────
    ticker_usdmad: str = "USDMAD=X"

    # ── Scénarios de stress (taux directeur) ────────────────────────────────
    chocs_stress_taux: dict = field(default_factory=lambda: {
        "Base"   : 0.00,    # Scénario de base
        "+100bp" : 0.01,    # Choc haussier modéré
        "+200bp" : 0.02,    # Choc haussier sévère
    })

    # ── Scénarios de stress (change) ────────────────────────────────────────
    chocs_stress_fx: dict = field(default_factory=lambda: {
        "Base"         : 1.00,   # Taux de base
        "Dépréc. -5%"  : 1.05,   # Dépréciation 5% du MAD
        "Dépréc. -10%" : 1.10,   # Dépréciation 10%
        "Dépréc. -15%" : 1.15,   # Dépréciation 15%
    })


# =============================================================================
# PALETTE GRAPHIQUE
# =============================================================================

@dataclass(frozen=True)
class PlotParams:
    """
    Palette de couleurs et style des graphiques.
    Thème sombre (fond noir, texte clair) pour une meilleure lisibilité
    des figures dans un mémoire académique imprimé ou projeté.
    """
    bg: str = "#0f1117"       # Fond de la figure
    bg_ax: str = "#1a1d27"    # Fond des axes
    txt: str = "#E0E0E0"      # Texte et labels

    c_obs: str = "#4FC3F7"    # Observations historiques (bleu clair)
    c_sim: str = "#FF8A65"    # Trajectoires simulées (orange)
    c_ci: str = "#90CAF9"     # Intervalles de confiance (bleu pâle)
    c_eq: str = "#A5D6A7"     # Équilibre / scénario baissier (vert pâle)
    c_mn: str = "#FFD54F"     # Moyenne / niveau (jaune ambré)
    c_var: str = "#EF5350"    # VaR / risque extrême (rouge)
    c_str: str = "#CE93D8"    # Scénarios de stress (violet)
    c_fwd: str = "#66BB6A"    # Forward de change / FX couvert (vert)

    c_nc: str = "#FF8A65"     # Non couvert (orange)
    c_irs: str = "#66BB6A"    # Couvert IRS (vert)
    c_es: str = "#CE93D8"     # Expected Shortfall (violet)

    dpi: int = 150
    fig_width_large: float = 16.0
    fig_height_large: float = 10.0


# =============================================================================
# CHEMINS DE FICHIERS
# =============================================================================

@dataclass
class PathParams:
    """
    Chemins de fichiers du projet.
    Tous les chemins sont relatifs à PROJECT_ROOT pour la portabilité.
    """
    donnees_bam: str = "taux_directeur_BAM.csv"   # Données BAM brutes
    dossier_figures: Path = PROJECT_ROOT / "figures"
    dossier_resultats: Path = PROJECT_ROOT / "results"
    dossier_logs: Path = PROJECT_ROOT / "logs"

    def __post_init__(self):
        """Crée les dossiers de sortie s'ils n'existent pas."""
        for d in [self.dossier_figures, self.dossier_resultats, self.dossier_logs]:
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# INSTANCES GLOBALES (importées par les autres modules)
# =============================================================================

AGENT   = AgentParams()
SIM     = SimulationParams()
MARKET  = MarketParams()
PLOT    = PlotParams()
PATHS   = PathParams()
