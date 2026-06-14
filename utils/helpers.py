"""Fonctions utilitaires pour l'application Coupe du Monde 2026."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


# Liste des 48 equipes utilisees par l'application.
# Les groupes correspondent a une simulation configurable, car tous les groupes
# officiels ne sont pas figes tant que le tirage final n'a pas eu lieu.
QUALIFIED_TEAMS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


FLAGS = {
    "Algeria": "🇩🇿",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Belgium": "🇧🇪",
    "Brazil": "🇧🇷",
    "Cameroon": "🇨🇲",
    "Canada": "🇨🇦",
    "Chile": "🇨🇱",
    "Colombia": "🇨🇴",
    "Costa Rica": "🇨🇷",
    "Croatia": "🇭🇷",
    "Czech Republic": "🇨🇿",
    "Denmark": "🇩🇰",
    "Ecuador": "🇪🇨",
    "Egypt": "🇪🇬",
    "England": "🏴",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Iran": "🇮🇷",
    "Iraq": "🇮🇶",
    "Italy": "🇮🇹",
    "Japan": "🇯🇵",
    "Mexico": "🇲🇽",
    "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱",
    "Nigeria": "🇳🇬",
    "Norway": "🇳🇴",
    "Paraguay": "🇵🇾",
    "Peru": "🇵🇪",
    "Poland": "🇵🇱",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴",
    "Senegal": "🇸🇳",
    "Serbia": "🇷🇸",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭",
    "Tunisia": "🇹🇳",
    "Turkey": "🇹🇷",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Uzbekistan": "🇺🇿",
    "Wales": "🏴",
}


OFFICIAL_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Nations League",
    "Copa America",
    "AFC Asian Cup",
    "African Cup of Nations",
    "CONCACAF Gold Cup",
    "Oceania Nations Cup",
}


def team_label(team: str) -> str:
    """Retourne le libelle d'une equipe avec son drapeau."""

    return f"{FLAGS.get(team, '🏳️')} {team}"


def clean_team_label(label: str) -> str:
    """Retire le drapeau affiche dans les selectbox."""

    return label.split(" ", 1)[1] if " " in label else label


def all_teams() -> List[str]:
    """Retourne toutes les equipes triees alphabetiquement."""

    teams = [team for group in QUALIFIED_TEAMS.values() for team in group]
    return sorted(teams)


def normalize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les resultats et ajoute des colonnes de travail."""

    df = results.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").fillna(0).astype(int)
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").fillna(0).astype(int)
    df["neutral"] = df["neutral"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["is_official"] = df["tournament"].isin(OFFICIAL_TOURNAMENTS).astype(int)
    df["target"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        [2, 1],
        default=0,
    )
    return df.sort_values("date").reset_index(drop=True)


def latest_rankings(ranking: pd.DataFrame) -> Dict[str, float]:
    """Retourne le classement FIFA le plus recent par equipe."""

    df = ranking.copy()
    df["rank_date"] = pd.to_datetime(df["rank_date"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["rank_date", "country_full", "rank"])
    latest = df.sort_values("rank_date").groupby("country_full").tail(1)
    return latest.set_index("country_full")["rank"].to_dict()


def ranking_before(ranking: pd.DataFrame, team: str, date: pd.Timestamp, fallback: float) -> float:
    """Recupere le dernier classement connu avant une date."""

    rows = ranking[(ranking["country_full"] == team) & (ranking["rank_date"] <= date)]
    if rows.empty:
        return fallback
    return float(rows.sort_values("rank_date").iloc[-1]["rank"])


def team_matches_before(results: pd.DataFrame, team: str, date: pd.Timestamp) -> pd.DataFrame:
    """Filtre les matchs d'une equipe avant une date donnee."""

    mask = ((results["home_team"] == team) | (results["away_team"] == team)) & (results["date"] < date)
    return results.loc[mask].sort_values("date")


def recent_form(results: pd.DataFrame, team: str, date: pd.Timestamp, n: int = 5) -> float:
    """Calcule les points moyens sur les derniers matchs."""

    matches = team_matches_before(results, team, date).tail(n)
    if matches.empty:
        return 1.0
    points = []
    for row in matches.itertuples():
        if row.home_team == team:
            points.append(3 if row.home_score > row.away_score else 1 if row.home_score == row.away_score else 0)
        else:
            points.append(3 if row.away_score > row.home_score else 1 if row.home_score == row.away_score else 0)
    return float(np.mean(points))


def goal_averages(results: pd.DataFrame, team: str, date: pd.Timestamp, n: int = 10) -> Tuple[float, float]:
    """Calcule les buts marques et encaisses moyens recents."""

    matches = team_matches_before(results, team, date).tail(n)
    if matches.empty:
        return 1.2, 1.2
    scored, conceded = [], []
    for row in matches.itertuples():
        if row.home_team == team:
            scored.append(row.home_score)
            conceded.append(row.away_score)
        else:
            scored.append(row.away_score)
            conceded.append(row.home_score)
    return float(np.mean(scored)), float(np.mean(conceded))


def head_to_head_score(results: pd.DataFrame, team_a: str, team_b: str, date: pd.Timestamp) -> float:
    """Mesure l'avantage historique de l'equipe A contre l'equipe B."""

    mask = (
        (((results["home_team"] == team_a) & (results["away_team"] == team_b)) |
         ((results["home_team"] == team_b) & (results["away_team"] == team_a))) &
        (results["date"] < date)
    )
    matches = results.loc[mask].sort_values("date").tail(8)
    if matches.empty:
        return 0.0
    values = []
    for row in matches.itertuples():
        if row.home_score == row.away_score:
            values.append(0)
        elif (row.home_team == team_a and row.home_score > row.away_score) or (
            row.away_team == team_a and row.away_score > row.home_score
        ):
            values.append(1)
        else:
            values.append(-1)
    return float(np.mean(values))


def build_features_for_match(
    results: pd.DataFrame,
    ranking: pd.DataFrame,
    team_a: str,
    team_b: str,
    match_date: datetime | pd.Timestamp | None = None,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
) -> pd.DataFrame:
    """Construit les variables ML pour un match."""

    date = pd.Timestamp(match_date or datetime(2026, 6, 13))
    fallback_rank = float(pd.to_numeric(ranking["rank"], errors="coerce").median())
    rank_a = ranking_before(ranking, team_a, date, fallback_rank)
    rank_b = ranking_before(ranking, team_b, date, fallback_rank)
    goals_a_for, goals_a_against = goal_averages(results, team_a, date)
    goals_b_for, goals_b_against = goal_averages(results, team_b, date)
    data = {
        "rank_home": rank_a,
        "rank_away": rank_b,
        "rank_diff": rank_a - rank_b,
        "form_home": recent_form(results, team_a, date),
        "form_away": recent_form(results, team_b, date),
        "form_diff": recent_form(results, team_a, date) - recent_form(results, team_b, date),
        "goals_for_home": goals_a_for,
        "goals_against_home": goals_a_against,
        "goals_for_away": goals_b_for,
        "goals_against_away": goals_b_against,
        "h2h_home": head_to_head_score(results, team_a, team_b, date),
        "is_official": int(tournament in OFFICIAL_TOURNAMENTS),
        "neutral": int(neutral),
    }
    return pd.DataFrame([data])


def last_matches(results: pd.DataFrame, team: str, n: int = 5) -> pd.DataFrame:
    """Retourne les derniers matchs affichables d'une equipe."""

    today = pd.Timestamp.today().normalize()
    safe_results = results[results["date"] <= today]
    matches = team_matches_before(safe_results, team, today + pd.Timedelta(days=1)).tail(n).copy()
    if matches.empty:
        return pd.DataFrame(columns=["date", "match", "score", "tournament"])
    matches["match"] = matches["home_team"] + " vs " + matches["away_team"]
    matches["score"] = matches["home_score"].astype(str) + " - " + matches["away_score"].astype(str)
    return matches[["date", "match", "score", "tournament"]].sort_values("date", ascending=False)


def poisson_score(prob_home: float, prob_draw: float, prob_away: float) -> str:
    """Produit un score plausible a partir des probabilites."""

    if prob_draw >= max(prob_home, prob_away):
        return "1 - 1"
    if prob_home > prob_away:
        return "2 - 1" if prob_home < 0.55 else "3 - 1"
    return "1 - 2" if prob_away < 0.55 else "1 - 3"


def win_rate_table(results: pd.DataFrame) -> pd.DataFrame:
    """Calcule le taux de victoire par equipe."""

    rows = []
    for team in sorted(set(results["home_team"]).union(results["away_team"])):
        matches = results[(results["home_team"] == team) | (results["away_team"] == team)]
        wins = 0
        goals = 0
        for row in matches.itertuples():
            if row.home_team == team:
                wins += int(row.home_score > row.away_score)
                goals += row.home_score
            else:
                wins += int(row.away_score > row.home_score)
                goals += row.away_score
        rows.append({"Equipe": team, "Matchs": len(matches), "Win rate": wins / max(len(matches), 1), "Buts": goals})
    return pd.DataFrame(rows).sort_values("Win rate", ascending=False)
