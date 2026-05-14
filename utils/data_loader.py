"""
=============================================================================
utils/data_loader.py — Chargement et préparation des données externes
=============================================================================
Fonctions de récupération et nettoyage des données de marché :
  - Taux directeur BAM depuis un fichier CSV local
  - Taux de change USD/MAD depuis Yahoo Finance
  - Taux SOFR depuis la Federal Reserve (FRED API)
=============================================================================
"""

import numpy as np
import pandas as pd
from fredapi import Fred
import yfinance as yf
from datetime import timedelta
from typing import Tuple

from src.config.settings import MARKET


# =============================================================================
# DONNÉES BAM (TAUX DIRECTEUR MAROCAIN)
# =============================================================================

def charger_taux_bam(chemin_csv: str) -> pd.DataFrame:
    """
    Charge et nettoie le fichier CSV des taux directeurs BAM.

    Format attendu : CSV séparé par ';', colonnes [date, taux],
    taux exprimé en pourcentage (ex. "3,00%" ou "3.00%").

    Parameters
    ----------
    chemin_csv : str
        Chemin vers le fichier CSV des taux BAM.

    Returns
    -------
    pd.DataFrame
        DataFrame trié par date avec colonnes ['date', 'taux'] (taux décimal).
    """
    df = pd.read_csv(chemin_csv, sep=";", usecols=[0, 1])

    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    df["taux"] = (
        df["taux"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(";", ".", regex=False)
    )
    df["taux"] = pd.to_numeric(df["taux"], errors="coerce") / 100

    df = df.dropna(subset=["date", "taux"])
    df = df.sort_values("date").reset_index(drop=True)

    return df[["date", "taux"]]


def construire_serie_mensuelle_bam(df_bam: pd.DataFrame) -> pd.Series:
    """
    Construit une série mensuelle continue par propagation avant (forward-fill).

    Le taux directeur BAM n'évolue qu'aux dates de décision du comité de
    politique monétaire (≈ 4 fois/an). Entre deux décisions, on maintient
    le dernier taux connu. On produit une série mensuelle (fréquence MS).

    Parameters
    ----------
    df_bam : pd.DataFrame
        Données brutes chargées par `charger_taux_bam`.

    Returns
    -------
    pd.Series
        Série mensuelle indexée par date, taux en décimal.
    """
    df = df_bam.set_index("date").sort_index()

    # Resampling mensuel avec propagation du dernier taux connu
    serie = df["taux"].resample("MS").ffill().dropna()

    return serie


# =============================================================================
# DONNÉES DE CHANGE USD/MAD (YAHOO FINANCE)
# =============================================================================

def telecharger_donnees_fx(
    ticker: str,
    date_fin: pd.Timestamp,
    horizon_jours: int = 365,
) -> pd.DataFrame:
    """
    Télécharge l'historique des cours USD/MAD depuis Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker Yahoo Finance (ex. "USDMAD=X").
    date_fin : pd.Timestamp
        Date de fin des données (typiquement la dernière date BAM disponible).
    horizon_jours : int
        Fenêtre historique pour calibrer la volatilité (défaut : 1 an).

    Returns
    -------
    pd.DataFrame
        DataFrame avec colonne 'Close' (cours de clôture journalier).
    """
    date_debut = date_fin - timedelta(days=horizon_jours)
    date_fin_dl = date_fin + timedelta(days=2)   # marge pour les jours fériés

    data = yf.download(ticker, start=date_debut, end=date_fin_dl, progress=False)
    return data


def extraire_taux_spot(
    donnees_fx: pd.DataFrame,
    date_reference: pd.Timestamp,
) -> float:
    """
    Extrait le taux spot USD/MAD à la date de référence.

    Si la date exacte n'est pas disponible (marché fermé, week-end),
    on prend le dernier cours disponible avant cette date.

    Parameters
    ----------
    donnees_fx : pd.DataFrame
        Données de change téléchargées.
    date_reference : pd.Timestamp
        Date souhaitée pour le taux spot.

    Returns
    -------
    float
        Taux spot S₀ (MAD par USD).
    """
    try:
        s0 = float(donnees_fx.loc[date_reference.strftime("%Y-%m-%d"), "Close"])
    except KeyError:
        series_avant = donnees_fx.loc[: date_reference.strftime("%Y-%m-%d")]
        s0 = float(series_avant["Close"].iloc[-1])
    return s0


def calculer_volatilite_gbm(donnees_fx: pd.DataFrame) -> float:
    """
    Calcule la volatilité annualisée du taux de change à partir des
    log-rendements journaliers (méthode des log-rendements).

    σ_S = std(ln(S_t / S_{t-1})) × √252

    Parameters
    ----------
    donnees_fx : pd.DataFrame
        Données de change avec colonne 'Close'.

    Returns
    -------
    float
        Volatilité annualisée σ_S (en décimal, ex. 0.035 = 3.5%).
    """
    log_rendements = np.log(
        donnees_fx["Close"] / donnees_fx["Close"].shift(1)
    ).dropna()
    return float(log_rendements.std() * np.sqrt(252))


# =============================================================================
# DONNÉES SOFR (FEDERAL RESERVE — FRED API)
# =============================================================================

def telecharger_sofr(
    date_reference: pd.Timestamp,
    api_key: str = None,
) -> float:
    """
    Récupère le taux SOFR (Secured Overnight Financing Rate) depuis FRED.

    Le SOFR est le taux de référence USD utilisé pour calculer le forward
    de change via la Parité des Taux Couverte (CIP).

    Parameters
    ----------
    date_reference : pd.Timestamp
        Date de référence pour le taux SOFR.
    api_key : str, optional
        Clé API FRED (défaut : depuis MARKET.fred_api_key).

    Returns
    -------
    float
        Taux SOFR en décimal (ex. 0.0530 = 5.30%).
    """
    if api_key is None:
        api_key = MARKET.fred_api_key

    fred = Fred(api_key=api_key)
    date_debut = date_reference - timedelta(days=5)
    date_fin = date_reference + timedelta(days=2)

    serie_sofr = fred.get_series(
        "SOFR",
        observation_start=date_debut,
        observation_end=date_fin,
    )
    return float(serie_sofr.iloc[-1] / 100)
