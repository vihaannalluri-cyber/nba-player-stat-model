"""Train and evaluate the three player-stat models."""

import json
from pathlib import Path

import pandas as pd

from nba_predictor.model import train_feature_models


FEATURE_FILE = Path("data/features.csv")
MODEL_FOLDER = Path("models")
REPORT_FILE = Path("outputs/metrics.json")


def main():
    if not FEATURE_FILE.exists():
        raise FileNotFoundError("Run `python3 build_features.py` first.")

    features = pd.read_csv(FEATURE_FILE, parse_dates=["GAME_DATE"])
    report = train_feature_models(features, MODEL_FOLDER, REPORT_FILE)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
