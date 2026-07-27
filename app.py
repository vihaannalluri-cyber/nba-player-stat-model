from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from nba_predictor.model import load_and_predict


PROJECT_FOLDER = Path(__file__).parent
DATA_FOLDER = PROJECT_FOLDER / "data"
MODEL_FOLDER = PROJECT_FOLDER / "models"

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

app = Flask(__name__)
players = pd.read_csv(DATA_FOLDER / "web_players.csv", parse_dates=["LAST_GAME_DATE"])
matchups = pd.read_csv(DATA_FOLDER / "web_matchups.csv")
opponents = pd.read_csv(DATA_FOLDER / "web_opponents.csv")


def make_prediction_row(player, opponent, game_date, is_home):
    player_info = players[players["PLAYER_NAME"] == player]
    if player_info.empty:
        raise ValueError(f"Player not found in data: {player}")

    row = player_info.iloc[0].to_dict()
    last_game_date = row.pop("LAST_GAME_DATE")
    row.pop("PLAYER_NAME")

    days_rest = (pd.Timestamp(game_date) - last_game_date).days
    days_rest = max(0, min(days_rest, 10))
    row["IS_HOME"] = int(is_home)
    row["DAYS_REST"] = days_rest
    row["IS_BACK_TO_BACK"] = int(days_rest <= 1)
    row["OPPONENT"] = opponent

    matchup = matchups[
        (matchups["PLAYER_NAME"] == player) & (matchups["OPPONENT"] == opponent)
    ]
    if matchup.empty:
        row["H2H_GAMES_BEFORE"] = 0
        row["PTS_H2H_5"] = np.nan
        row["REB_H2H_5"] = np.nan
        row["AST_H2H_5"] = np.nan
    else:
        matchup_info = matchup.iloc[0]
        for column in ["H2H_GAMES_BEFORE", "PTS_H2H_5", "REB_H2H_5", "AST_H2H_5"]:
            row[column] = matchup_info[column]

    opponent_info = opponents[opponents["OPPONENT"] == opponent].iloc[0]
    for column in ["OPP_ALLOWED_PTS_10", "OPP_ALLOWED_REB_10", "OPP_ALLOWED_AST_10"]:
        row[column] = opponent_info[column]

    return pd.DataFrame([row])


@app.route("/")
def home():
    player_names = sorted(players["PLAYER_NAME"].unique())
    first_future_date = players["LAST_GAME_DATE"].max().date() + timedelta(days=1)
    return render_template(
        "index.html",
        players=player_names,
        teams=NBA_TEAMS,
        first_future_date=first_future_date.isoformat(),
    )


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    form = request.get_json()
    player = form.get("player", "").strip()
    opponent = form.get("opponent", "").upper()
    game_date = form.get("date", "")
    location = form.get("location", "home")

    if not player or opponent not in NBA_TEAMS or not game_date:
        return jsonify({"error": "Please fill out every field."}), 400

    try:
        prediction_row = make_prediction_row(player, opponent, game_date, location == "home")
        prediction = load_and_predict(prediction_row, 0, MODEL_FOLDER)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "player": player,
            "opponent": opponent,
            "location": location,
            "prediction": prediction,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
