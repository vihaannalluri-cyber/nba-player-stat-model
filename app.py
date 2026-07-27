from datetime import timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template, request

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

app = Flask(__name__)
games = read_games(DATA_FILE)


@app.route("/")
def home():
    players = sorted(games["PLAYER_NAME"].unique())
    first_future_date = games["GAME_DATE"].max().date() + timedelta(days=1)
    return render_template(
        "index.html",
        players=players,
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
        featured_games, row_index = add_future_matchup(
            games,
            player,
            opponent,
            game_date,
            location == "home",
        )
        prediction = load_and_predict(featured_games, row_index, MODEL_FOLDER)
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
