import csv
from datetime import date, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from nba_predictor.web_model import load_web_models, predict as predict_stats


PROJECT_FOLDER = Path(__file__).parent
DATA_FOLDER = PROJECT_FOLDER / "data"
MODEL_FOLDER = PROJECT_FOLDER / "models"

NBA_TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

def read_summary(filename, key_columns):
    text_columns = set(key_columns) | {"LAST_GAME_DATE", "TEAM_ABBREVIATION"}
    rows = {}
    with (DATA_FOLDER / filename).open(newline="") as file:
        for source_row in csv.DictReader(file):
            key = tuple(source_row[column] for column in key_columns)
            row = {}
            for column, value in source_row.items():
                if column in text_columns:
                    row[column] = value
                else:
                    row[column] = float(value) if value else None
            rows[key] = row
    return rows


app = Flask(__name__)
players = read_summary("web_players.csv", ["PLAYER_NAME"])
matchups = read_summary("web_matchups.csv", ["PLAYER_NAME", "OPPONENT"])
opponents = read_summary("web_opponents.csv", ["OPPONENT"])
web_models = load_web_models(MODEL_FOLDER)


def make_prediction_row(player, opponent, game_date, is_home):
    player_info = players.get((player,))
    if player_info is None:
        raise ValueError(f"Player not found in data: {player}")

    row = player_info.copy()
    last_game_date = date.fromisoformat(row.pop("LAST_GAME_DATE"))
    row.pop("PLAYER_NAME")

    days_rest = (date.fromisoformat(game_date) - last_game_date).days
    days_rest = max(0, min(days_rest, 10))
    row["IS_HOME"] = int(is_home)
    row["DAYS_REST"] = days_rest
    row["IS_BACK_TO_BACK"] = int(days_rest <= 1)
    row["OPPONENT"] = opponent

    matchup = matchups.get((player, opponent))
    if matchup is None:
        row["H2H_GAMES_BEFORE"] = 0
        row["PTS_H2H_5"] = None
        row["REB_H2H_5"] = None
        row["AST_H2H_5"] = None
    else:
        for column in ["H2H_GAMES_BEFORE", "PTS_H2H_5", "REB_H2H_5", "AST_H2H_5"]:
            row[column] = matchup[column]

    opponent_info = opponents[(opponent,)]
    for column in ["OPP_ALLOWED_PTS_10", "OPP_ALLOWED_REB_10", "OPP_ALLOWED_AST_10"]:
        row[column] = opponent_info[column]

    return row


@app.route("/")
def home():
    player_names = sorted(key[0] for key in players)
    last_game_date = max(date.fromisoformat(row["LAST_GAME_DATE"]) for row in players.values())
    first_future_date = last_game_date + timedelta(days=1)
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
        prediction = predict_stats(prediction_row, web_models)
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
