"""Entrainement local du modele Random Forest + XGBoost."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - garde utile si xgboost n'est pas installe.
    XGBClassifier = None

from utils.helpers import DATA_DIR, build_features_for_match, normalize_results


MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
FEATURE_COLUMNS = [
    "rank_home",
    "rank_away",
    "rank_diff",
    "form_home",
    "form_away",
    "form_diff",
    "goals_for_home",
    "goals_against_home",
    "goals_for_away",
    "goals_against_away",
    "h2h_home",
    "is_official",
    "neutral",
]


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les CSV necessaires a l'entrainement."""

    results = normalize_results(pd.read_csv(DATA_DIR / "results.csv"))
    ranking = pd.read_csv(DATA_DIR / "fifa_ranking.csv")
    ranking["rank_date"] = pd.to_datetime(ranking["rank_date"], errors="coerce")
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
    return results, ranking


def build_training_frame(results: pd.DataFrame, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Transforme les matchs historiques en matrice de variables."""

    rows = []
    labels = []
    # On saute les premiers matchs pour laisser les features de forme se stabiliser.
    for row in results.sort_values("date").iloc[20:].itertuples():
        features = build_features_for_match(
            results=results,
            ranking=ranking,
            team_a=row.home_team,
            team_b=row.away_team,
            match_date=row.date,
            tournament=row.tournament,
            neutral=bool(row.neutral),
        )
        rows.append(features.iloc[0])
        labels.append(int(row.target))
    return pd.DataFrame(rows)[FEATURE_COLUMNS], pd.Series(labels)


def make_model() -> VotingClassifier:
    """Cree l'ensemble Random Forest + XGBoost."""

    rf = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=2,
        max_depth=9,
        random_state=42,
        class_weight="balanced_subsample",
    )
    if XGBClassifier is None:
        # Fallback local si xgboost n'est pas disponible pendant un test rapide.
        return VotingClassifier(estimators=[("rf", rf)], voting="soft")
    xgb = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")


def train_and_save() -> Dict:
    """Entraine le modele, calcule l'accuracy et sauvegarde le bundle."""

    results, ranking = load_training_data()
    x, y = build_training_frame(results, ranking)
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    model = make_model()
    model.fit(x_train, y_train)
    accuracy = float(accuracy_score(y_test, model.predict(x_test)))
    bundle = {
        "model": model,
        "accuracy": accuracy,
        "features": FEATURE_COLUMNS,
        "trained_rows": int(len(x_train)),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    bundle = train_and_save()
    print(f"Modele sauvegarde dans {MODEL_PATH} - accuracy: {bundle['accuracy']:.3f}")
