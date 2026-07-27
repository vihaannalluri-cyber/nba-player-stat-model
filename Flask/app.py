"""Flask interface for generating NBA player stat projections."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from nba_predictor.data import read_games
from nba_predictor.features import add_future_matchup
from nba_predictor.model import load_and_predict

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "player_games.csv"
MODEL_PATH = BASE_DIR / "models"
NBA_TEAMS = (
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
)

app = Flask(__name__)


@lru_cache(maxsize=1)
def load_games():
    return read_games(DATA_PATH)


@app.get("/")
def index():
    games = load_games()
    players = sorted(games["PLAYER_NAME"].dropna().unique())
    latest_date = games["GAME_DATE"].max().date().isoformat()
    return render_template(
        "index.html",
        players=players,
        teams=NBA_TEAMS,
        latest_date=latest_date,
    )


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    player = str(payload.get("player", "")).strip()
    opponent = str(payload.get("opponent", "")).strip().upper()
    game_date = str(payload.get("date", "")).strip()
    location = str(payload.get("location", "home")).lower()

    if not all((player, opponent, game_date)):
        return jsonify({"error": "Choose a player, opponent, and game date."}), 400
    if opponent not in NBA_TEAMS:
        return jsonify({"error": "Choose a valid NBA opponent."}), 400
    if location not in {"home", "away"}:
        return jsonify({"error": "Location must be home or away."}), 400

    try:
        featured, row_index = add_future_matchup(
            load_games(), player, opponent, game_date, location == "home"
        )
        projection = load_and_predict(featured, row_index, MODEL_PATH)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "player": player,
            "opponent": opponent,
            "date": game_date,
            "location": location,
            "prediction": projection,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
