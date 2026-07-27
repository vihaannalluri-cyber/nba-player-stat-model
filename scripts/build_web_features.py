from pathlib import Path

import numpy as np
import pandas as pd

from nba_predictor.data import read_games
from nba_predictor.features import ROLL_WINDOWS, TARGETS


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / "data"


def build_player_features(games):
    rows = []

    for player_name, player_games in games.groupby("PLAYER_NAME"):
        player_games = player_games.sort_values(["GAME_DATE", "GAME_ID"])
        latest_game = player_games.iloc[-1]
        row = {
            "PLAYER_NAME": player_name,
            "TEAM_ABBREVIATION": latest_game["TEAM_ABBREVIATION"],
            "LAST_GAME_DATE": latest_game["GAME_DATE"].date().isoformat(),
            "CAREER_GAMES_BEFORE": len(player_games),
        }

        for stat in ["MIN"] + TARGETS:
            for window in ROLL_WINDOWS:
                row[f"{stat}_ROLL_{window}"] = player_games[stat].tail(window).mean()
            row[f"{stat}_EXPANDING"] = player_games[stat].mean()

        safe_minutes = player_games["MIN"].replace(0, np.nan)
        for stat in TARGETS:
            per_minute = player_games[stat] / safe_minutes
            row[f"{stat}_PER_MIN_ROLL_10"] = per_minute.tail(10).mean()

        rows.append(row)

    return pd.DataFrame(rows)


def build_matchup_features(games):
    rows = []
    grouped = games.groupby(["PLAYER_NAME", "OPPONENT"])

    for (player_name, opponent), matchups in grouped:
        matchups = matchups.sort_values(["GAME_DATE", "GAME_ID"])
        row = {
            "PLAYER_NAME": player_name,
            "OPPONENT": opponent,
            "H2H_GAMES_BEFORE": len(matchups),
        }
        for stat in TARGETS:
            row[f"{stat}_H2H_5"] = matchups[stat].tail(5).mean()
        rows.append(row)

    return pd.DataFrame(rows)


def build_opponent_features(games):
    team_games = (
        games.groupby(
            ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "OPPONENT"],
            as_index=False,
        )[TARGETS]
        .sum()
        .sort_values(["GAME_DATE", "GAME_ID"])
    )

    rows = []
    for opponent, opponent_games in team_games.groupby("OPPONENT"):
        row = {"OPPONENT": opponent}
        for stat in TARGETS:
            row[f"OPP_ALLOWED_{stat}_10"] = opponent_games[stat].tail(10).mean()
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    games = read_games(DATA_FOLDER / "player_games.csv")
    build_player_features(games).to_csv(DATA_FOLDER / "web_players.csv", index=False)
    build_matchup_features(games).to_csv(DATA_FOLDER / "web_matchups.csv", index=False)
    build_opponent_features(games).to_csv(DATA_FOLDER / "web_opponents.csv", index=False)
    print("Saved web feature files in data/")


if __name__ == "__main__":
    main()
