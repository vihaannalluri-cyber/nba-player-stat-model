"""Build the model-ready feature table from the historical game log."""

from pathlib import Path

from nba_predictor.data import read_games
from nba_predictor.features import build_features


DATA_FILE = Path("data/player_games.csv")
FEATURE_FILE = Path("data/features.csv")


def main():
    games = read_games(DATA_FILE)
    features = build_features(games)
    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_FILE, index=False, date_format="%Y-%m-%d")
    print(f"Built {len(features):,} feature rows in {FEATURE_FILE}")


if __name__ == "__main__":
    main()
