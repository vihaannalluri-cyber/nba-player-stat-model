from datetime import timedelta
from pathlib import Path

import streamlit as st

from nba_predictor.data import read_games
from nba_predictor.features import add_future_matchup
from nba_predictor.model import load_and_predict


PROJECT_FOLDER = Path(__file__).parent
DATA_FILE = PROJECT_FOLDER / "data" / "player_games.csv"
MODEL_FOLDER = PROJECT_FOLDER / "models"

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]


@st.cache_data
def load_data():
    return read_games(DATA_FILE)


st.set_page_config(page_title="NBA Player Stat Predictor", page_icon="🏀")
st.title("NBA Player Stat Predictor")
st.write("Choose a player and matchup to predict points, rebounds, and assists.")

games = load_data()
players = sorted(games["PLAYER_NAME"].unique())
first_future_date = games["GAME_DATE"].max().date() + timedelta(days=1)

with st.form("prediction_form"):
    player = st.selectbox("Player", players)

    left, right = st.columns(2)
    with left:
        opponent = st.selectbox("Opponent", NBA_TEAMS)
    with right:
        game_date = st.date_input(
            "Game date",
            value=first_future_date,
            min_value=first_future_date,
        )

    location = st.radio("Location", ["Home", "Away"], horizontal=True)
    predict_button = st.form_submit_button("Predict stats")

if predict_button:
    with st.spinner("Running the model..."):
        featured_games, row_index = add_future_matchup(
            games,
            player,
            opponent,
            str(game_date),
            location == "Home",
        )
        prediction = load_and_predict(featured_games, row_index, MODEL_FOLDER)

    st.subheader(f"{player} vs. {opponent}")
    points, rebounds, assists = st.columns(3)
    points.metric("Points", prediction["PTS"])
    rebounds.metric("Rebounds", prediction["REB"])
    assists.metric("Assists", prediction["AST"])

st.caption("Predictions are for educational use and do not include injury or lineup news.")
