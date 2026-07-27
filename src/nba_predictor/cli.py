from __future__ import annotations

import argparse
import json

from .data import download_seasons, read_games
from .features import add_future_matchup
from .model import load_and_predict, train_models


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="nba_predictor",
        description="Train NBA player models and project future box scores.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="download NBA player game logs")
    download.add_argument("--seasons", nargs="+", required=True)
    download.add_argument("--output", default="data/player_games.csv")
    download.add_argument("--pause", type=float, default=1.0)

    validate = commands.add_parser("validate", help="validate a historical CSV")
    validate.add_argument("--data", required=True)

    train = commands.add_parser("train", help="train and evaluate models")
    train.add_argument("--data", required=True)
    train.add_argument("--models", default="models")
    train.add_argument("--report", default="outputs/metrics.json")

    predict = commands.add_parser("predict", help="predict a future player matchup")
    predict.add_argument("--data", required=True)
    predict.add_argument("--models", default="models")
    predict.add_argument("--player", required=True)
    predict.add_argument("--opponent", required=True)
    predict.add_argument("--date", required=True)
    location = predict.add_mutually_exclusive_group(required=True)
    location.add_argument("--home", action="store_true")
    location.add_argument("--away", action="store_true")
    return root


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "download":
        games = download_seasons(args.seasons, args.output, args.pause)
        print(f"Saved {len(games):,} player-games to {args.output}")
        return

    if args.command == "validate":
        games = read_games(args.data)
        print(f"Valid: {len(games):,} rows, {games['PLAYER_NAME'].nunique():,} players")
        return

    if args.command == "train":
        report = train_models(read_games(args.data), args.models, args.report)
        print(json.dumps(report, indent=2))
        return

    games = read_games(args.data)
    featured, row_index = add_future_matchup(
        games, args.player, args.opponent, args.date, args.home
    )
    prediction = load_and_predict(featured, row_index, args.models)
    print(
        json.dumps(
            {
                "player": args.player,
                "opponent": args.opponent.upper(),
                "date": args.date,
                "location": "home" if args.home else "away",
                "prediction": prediction,
            },
            indent=2,
        )
    )
