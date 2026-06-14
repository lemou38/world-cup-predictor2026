"""Fonctions de prediction chargeant le modele sauvegarde."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from utils.helpers import build_features_for_match, poisson_score


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


def load_model_bundle() -> Dict:
    """Charge le modele sauvegarde par joblib."""

    return joblib.load(MODEL_PATH)


def predict_match(
    bundle: Dict,
    results: pd.DataFrame,
    ranking: pd.DataFrame,
    team_a: str,
    team_b: str,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
) -> Dict:
    """Retourne les probabilites, le choix recommande et le score predit."""

    features = build_features_for_match(results, ranking, team_a, team_b, tournament=tournament, neutral=neutral)
    model = bundle["model"]
    classes = list(model.classes_)
    probabilities = model.predict_proba(features)[0]
    mapped = {int(label): float(probabilities[index]) for index, label in enumerate(classes)}
    prob_away = mapped.get(0, 0.0)
    prob_draw = mapped.get(1, 0.0)
    prob_home = mapped.get(2, 0.0)
    choices = {
        "Victoire equipe 1": prob_home,
        "Match nul": prob_draw,
        "Victoire equipe 2": prob_away,
    }
    return {
        "home_win": prob_home,
        "draw": prob_draw,
        "away_win": prob_away,
        "recommended": max(choices, key=choices.get),
        "score": poisson_score(prob_home, prob_draw, prob_away),
        "features": features,
    }
