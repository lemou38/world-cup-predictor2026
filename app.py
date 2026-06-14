"""Application Streamlit de prediction Coupe du Monde 2026."""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.predict import MODEL_PATH, predict_match
from model.train import train_and_save
from utils.helpers import (
    DATA_DIR,
    FLAGS,
    QUALIFIED_TEAMS,
    all_teams,
    clean_team_label,
    last_matches,
    normalize_results,
    team_label,
    win_rate_table,
)


st.set_page_config(
    page_title="World Cup Predictor 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    """Ajoute le theme visuel vert fonce et les animations."""

    st.markdown(
        """
        <style>
        :root { --wc-green: #1a4731; --wc-gold: #d6b25e; --wc-red: #b83a3a; }
        .stApp { background: #f5f7f3; color: #17231d; }
        [data-testid="stSidebar"] { background: #102c21; }
        [data-testid="stSidebar"] * { color: #f4f7ef !important; }
        h1, h2, h3 { color: #1a4731; letter-spacing: 0; }
        .hero {
            background: linear-gradient(135deg, #1a4731, #0e2b20);
            color: white;
            padding: 22px 26px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,.16);
            overflow: hidden;
        }
        .ticker-wrap {
            margin-top: 16px;
            overflow: hidden;
            white-space: nowrap;
            border-top: 1px solid rgba(255,255,255,.18);
            border-bottom: 1px solid rgba(255,255,255,.18);
            padding: 10px 0;
        }
        .ticker {
            display: inline-block;
            animation: scrollFlags 55s linear infinite;
            font-size: 1.05rem;
        }
        @keyframes scrollFlags {
            from { transform: translateX(0); }
            to { transform: translateX(-50%); }
        }
        .metric-card {
            background: white;
            border: 1px solid #dfe8de;
            border-radius: 8px;
            padding: 16px;
        }
        .prediction-box {
            background: white;
            border: 1px solid #dfe8de;
            border-radius: 8px;
            padding: 18px;
        }
        .stProgress > div > div { background-color: #1a4731; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les datasets locaux avec cache Streamlit."""

    results = normalize_results(pd.read_csv(DATA_DIR / "results.csv"))
    ranking = pd.read_csv(DATA_DIR / "fifa_ranking.csv")
    ranking["rank_date"] = pd.to_datetime(ranking["rank_date"], errors="coerce")
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
    goalscorers = pd.read_csv(DATA_DIR / "goalscorers.csv")
    shootouts = pd.read_csv(DATA_DIR / "shootouts.csv")
    return results, ranking, goalscorers, shootouts


@st.cache_data(show_spinner=False)
def load_notebook_insights() -> dict[str, pd.DataFrame]:
    """Charge les tableaux extraits du notebook Kaggle fourni."""

    return {
        "stages": pd.read_csv(DATA_DIR / "notebook_stage_counts.csv"),
        "venues": pd.read_csv(DATA_DIR / "notebook_venues.csv"),
        "group_travel": pd.read_csv(DATA_DIR / "notebook_group_travel.csv"),
        "tough_travel": pd.read_csv(DATA_DIR / "notebook_tough_travel.csv"),
        "rivalries": pd.read_csv(DATA_DIR / "notebook_returning_rivalries.csv"),
    }


@st.cache_resource(show_spinner=False)
def load_or_train_model() -> dict:
    """Charge le modele sauvegarde ou l'entraine si besoin."""

    if MODEL_PATH.exists():
        import joblib

        return joblib.load(MODEL_PATH)
    return train_and_save()


def header() -> None:
    """Affiche l'en-tete avec ticker de drapeaux et compte a rebours."""

    teams = all_teams()
    flag_line = "  ·  ".join(team_label(team) for team in teams)
    doubled = f"{flag_line}  ·  {flag_line}"
    kickoff = datetime(2026, 6, 13, 0, 0, 0)
    now = datetime.now()
    delta = kickoff - now
    if delta.total_seconds() > 0:
        countdown = f"{delta.days} jours · {delta.seconds // 3600} h · {(delta.seconds // 60) % 60} min"
    else:
        countdown = "La competition a commence"
    st.markdown(
        f"""
        <div class="hero">
            <h1 style="color:white;margin:0;">World Cup Predictor 2026</h1>
            <p style="margin:8px 0 0;color:#dfeee6;">Predictions ML hors ligne, donnees locales, simulation du tournoi.</p>
            <p style="margin:10px 0 0;color:#d6b25e;font-weight:700;">Compte a rebours: {countdown}</p>
            <div class="ticker-wrap"><div class="ticker">{doubled}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_results(results: pd.DataFrame, start_year: int, match_type: str) -> pd.DataFrame:
    """Applique les filtres globaux de la sidebar."""

    filtered = results[results["date"].dt.year >= start_year].copy()
    if match_type == "Competitions officielles":
        filtered = filtered[filtered["is_official"] == 1]
    elif match_type == "Amicaux":
        filtered = filtered[filtered["is_official"] == 0]
    return filtered


def sidebar() -> tuple[str, int, str, bool]:
    """Cree la navigation et les filtres lateraux."""

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", ["Prediction", "Groupes", "Head-to-Head", "Statistiques", "Route 2026", "Bracket"])
    st.sidebar.divider()
    start_year = st.sidebar.slider("Periode depuis", 1990, 2026, 2014)
    match_type = st.sidebar.selectbox("Type de match", ["Tous", "Competitions officielles", "Amicaux"])
    neutral = st.sidebar.checkbox("Terrain neutre", value=True)
    return page, start_year, match_type, neutral


def probability_bar(label: str, value: float, color: str) -> None:
    """Affiche une probabilite avec barre coloree."""

    st.markdown(f"**{label}** · {value:.1%}")
    st.markdown(
        f"""
        <div style="height:14px;background:#e8eee8;border-radius:7px;overflow:hidden;margin-bottom:12px;">
            <div style="width:{value * 100:.2f}%;height:14px;background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_prediction(results: pd.DataFrame, ranking: pd.DataFrame, bundle: dict, neutral: bool) -> None:
    """Page de prediction d'un match."""

    st.header("Prediction")
    labels = [team_label(team) for team in all_teams()]
    col1, col2, col3 = st.columns([1, 1, 0.8])
    with col1:
        home_label = st.selectbox("Equipe 1", labels, index=labels.index(team_label("France")))
    with col2:
        away_label = st.selectbox("Equipe 2", labels, index=labels.index(team_label("Brazil")))
    with col3:
        tournament = st.selectbox("Contexte", ["FIFA World Cup", "Friendly", "Copa America", "UEFA Euro"])

    team_a = clean_team_label(home_label)
    team_b = clean_team_label(away_label)
    if st.button("Predire", type="primary", use_container_width=True):
        if team_a == team_b:
            st.warning("Choisissez deux equipes differentes.")
            return
        prediction = predict_match(bundle, results, ranking, team_a, team_b, tournament=tournament, neutral=neutral)
        left, right = st.columns([0.95, 1.05])
        with left:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.subheader(f"{team_label(team_a)} vs {team_label(team_b)}")
            st.metric("Choix recommande", prediction["recommended"])
            st.metric("Score predit", prediction["score"])
            st.caption(f"Accuracy du modele: {bundle.get('accuracy', 0):.1%}")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            probability_bar(f"Victoire {team_a}", prediction["home_win"], "#1a4731")
            probability_bar("Match nul", prediction["draw"], "#87928a")
            probability_bar(f"Victoire {team_b}", prediction["away_win"], "#b83a3a")

 

def simulate_group(group: list[str], results: pd.DataFrame, ranking: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Simule tous les matchs d'un groupe."""

    table = {team: {"Equipe": team, "Pts": 0, "J": 0, "G": 0, "N": 0, "P": 0, "BP": 0, "BC": 0} for team in group}
    for team_a, team_b in combinations(group, 2):
        pred = predict_match(bundle, results, ranking, team_a, team_b)
        score_a, score_b = [int(x.strip()) for x in pred["score"].split("-")]
        table[team_a]["J"] += 1
        table[team_b]["J"] += 1
        table[team_a]["BP"] += score_a
        table[team_a]["BC"] += score_b
        table[team_b]["BP"] += score_b
        table[team_b]["BC"] += score_a
        if score_a > score_b:
            table[team_a]["Pts"] += 3
            table[team_a]["G"] += 1
            table[team_b]["P"] += 1
        elif score_a == score_b:
            table[team_a]["Pts"] += 1
            table[team_b]["Pts"] += 1
            table[team_a]["N"] += 1
            table[team_b]["N"] += 1
        else:
            table[team_b]["Pts"] += 3
            table[team_b]["G"] += 1
            table[team_a]["P"] += 1
    df = pd.DataFrame(table.values())
    df["Diff"] = df["BP"] - df["BC"]
    return df.sort_values(["Pts", "Diff", "BP"], ascending=False)


def page_groupes(results: pd.DataFrame, ranking: pd.DataFrame, bundle: dict) -> None:
    """Page des groupes et simulations."""

    st.header("Groupes")
    group_rows = [{"Groupe": group, "Equipe": team_label(team)} for group, teams in QUALIFIED_TEAMS.items() for team in teams]
    st.dataframe(pd.DataFrame(group_rows), use_container_width=True, hide_index=True)
    st.subheader("Simulation phase de groupes")
    cols = st.columns(3)
    for index, (group, teams) in enumerate(QUALIFIED_TEAMS.items()):
        with cols[index % 3]:
            st.markdown(f"**Groupe {group}**")
            st.dataframe(simulate_group(teams, results, ranking, bundle), use_container_width=True, hide_index=True)


def page_h2h(results: pd.DataFrame) -> None:
    """Page historique des confrontations."""

    st.header("Head-to-Head")
    labels = [team_label(team) for team in all_teams()]
    c1, c2 = st.columns(2)
    team_a = clean_team_label(c1.selectbox("Equipe A", labels, index=labels.index(team_label("Argentina"))))
    team_b = clean_team_label(c2.selectbox("Equipe B", labels, index=labels.index(team_label("France"))))
    mask = ((results["home_team"] == team_a) & (results["away_team"] == team_b)) | (
        (results["home_team"] == team_b) & (results["away_team"] == team_a)
    )
    h2h = results.loc[mask].sort_values("date").copy()
    if h2h.empty:
        st.info("Aucune confrontation trouvee dans les donnees locales.")
        return
    h2h["Affiche"] = h2h["home_team"] + " " + h2h["home_score"].astype(str) + " - " + h2h["away_score"].astype(str) + " " + h2h["away_team"]
    st.dataframe(h2h[["date", "Affiche", "tournament", "city", "country"]], use_container_width=True, hide_index=True)
    h2h["Buts equipe domicile"] = h2h["home_score"]
    h2h["Buts equipe exterieur"] = h2h["away_score"]
    fig = px.line(h2h, x="date", y=["Buts equipe domicile", "Buts equipe exterieur"], markers=True)
    st.plotly_chart(fig, use_container_width=True)


def page_stats(results: pd.DataFrame) -> None:
    """Page de statistiques globales."""

    st.header("Statistiques")
    win_rates = win_rate_table(results)
    c1, c2 = st.columns(2)
    c1.subheader("Top 10 win rate")
    c1.dataframe(win_rates.head(10), use_container_width=True, hide_index=True)
    c2.subheader("Equipes les plus prolifiques")
    fig_goals = px.bar(win_rates.sort_values("Buts", ascending=False).head(12), x="Equipe", y="Buts", color="Buts", color_continuous_scale="Greens")
    c2.plotly_chart(fig_goals, use_container_width=True)

    teams = all_teams()[:24]
    matrix = pd.DataFrame(0, index=teams, columns=teams, dtype=int)
    for row in results.itertuples():
        if row.home_team in matrix.index and row.away_team in matrix.columns:
            matrix.loc[row.home_team, row.away_team] += 1
            matrix.loc[row.away_team, row.home_team] += 1
    st.subheader("Heatmap des confrontations")
    fig_heat = px.imshow(matrix, color_continuous_scale="Greens", aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)


def page_route_2026() -> None:
    """Page issue du notebook World Cup Rivalries and 2026 Road."""

    st.header("Route 2026")
    insights = load_notebook_insights()
    stages = insights["stages"]
    venues = insights["venues"]
    group_travel = insights["group_travel"].sort_values("mean_km", ascending=False)
    tough_travel = insights["tough_travel"]
    rivalries = insights["rivalries"]

    st.caption("Synthese locale extraite du notebook fourni: structure 104 matchs, hôtes, stades, voyages et rivalites historiques.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matchs 2026", int(stages["matches"].sum()))
    m2.metric("Phase de groupes", int(stages.loc[stages["stage"] == "Group Stage", "matches"].iloc[0]))
    m3.metric("Stades", venues["venue"].nunique())
    m4.metric("Pays hôtes", venues["country"].nunique())

    c1, c2 = st.columns([0.9, 1.1])
    with c1:
        st.subheader("Structure du tournoi")
        fig_stage = px.bar(stages, x="stage", y="matches", color="matches", color_continuous_scale="Greens")
        fig_stage.update_layout(xaxis_title="", yaxis_title="Matchs", height=390)
        st.plotly_chart(fig_stage, use_container_width=True)
    with c2:
        st.subheader("Carte des stades")
        fig_map = px.scatter_geo(
            venues,
            lat="lat",
            lon="lon",
            size="matches",
            color="country",
            hover_name="venue",
            hover_data={"city": True, "matches": True, "lat": False, "lon": False},
            scope="north america",
            size_max=30,
        )
        fig_map.update_layout(height=390, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Charge de voyage par groupe")
    fig_travel = px.bar(
        group_travel,
        x="group",
        y="mean_km",
        color="max_km",
        color_continuous_scale="Greens",
        labels={"group": "Groupe", "mean_km": "Distance moyenne (km)", "max_km": "Max équipe (km)"},
    )
    st.plotly_chart(fig_travel, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Trajets individuels les plus lourds")
        st.dataframe(tough_travel, use_container_width=True, hide_index=True)
    with c4:
        st.subheader("Rivalités retrouvées")
        st.dataframe(rivalries, use_container_width=True, hide_index=True)


def page_bracket(results: pd.DataFrame, ranking: pd.DataFrame, bundle: dict) -> None:
    """Page de simulation de phase finale."""

    st.header("Bracket")
    qualifiers = []
    for group, teams in QUALIFIED_TEAMS.items():
        table = simulate_group(teams, results, ranking, bundle)
        qualifiers.extend(table["Equipe"].head(2).tolist())
    round_teams = qualifiers[:32]
    rounds = {"32es": round_teams}
    current = round_teams
    for round_name in ["16es", "Quarts", "Demies", "Finale", "Champion"]:
        winners = []
        for index in range(0, len(current), 2):
            if index + 1 >= len(current):
                winners.append(current[index])
                continue
            team_a, team_b = current[index], current[index + 1]
            pred = predict_match(bundle, results, ranking, team_a, team_b)
            winners.append(team_a if pred["home_win"] >= pred["away_win"] else team_b)
        rounds[round_name] = winners
        current = winners
    max_len = max(len(values) for values in rounds.values())
    bracket = pd.DataFrame({name: values + [""] * (max_len - len(values)) for name, values in rounds.items()})
    st.dataframe(bracket, use_container_width=True, hide_index=True)
    st.success(f"Champion simule: {team_label(rounds['Champion'][0])}")

    fig = go.Figure()
    for col_index, (name, values) in enumerate(rounds.items()):
        for row_index, team in enumerate(values):
            fig.add_trace(go.Scatter(x=[col_index], y=[-row_index], mode="text", text=[team_label(team)], textfont=dict(size=13), showlegend=False))
    fig.update_layout(height=620, xaxis=dict(tickmode="array", tickvals=list(range(len(rounds))), ticktext=list(rounds.keys())), yaxis=dict(visible=False), plot_bgcolor="#f5f7f3")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Point d'entree de l'application."""

    inject_css()
    header()
    page, start_year, match_type, neutral = sidebar()
    results, ranking, _, _ = load_data()
    filtered_results = filter_results(results, start_year, match_type)
    bundle = load_or_train_model()
    if page == "Prediction":
        page_prediction(filtered_results, ranking, bundle, neutral)
    elif page == "Groupes":
        page_groupes(filtered_results, ranking, bundle)
    elif page == "Head-to-Head":
        page_h2h(filtered_results)
    elif page == "Statistiques":
        page_stats(filtered_results)
    elif page == "Route 2026":
        page_route_2026()
    else:
        page_bracket(filtered_results, ranking, bundle)


if __name__ == "__main__":
    main()
