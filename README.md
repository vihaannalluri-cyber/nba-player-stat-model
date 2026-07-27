# NBA Player Stat Predictor

This is a Python machine learning project that predicts a player's points, rebounds, and assists for a future NBA game.

I designed the model logic, prepared the game data, and decided which stats the model should use. I used AI to help generate some of the boilerplate code based on my logic and data requirements.

## What the model uses

- The player's last 5, 10, and 20 games
- Average minutes and stats per minute
- Season averages
- Previous games against the same opponent
- Stats the opponent has recently allowed
- Home or away games
- Days of rest and back-to-backs

The rolling stats are shifted by one game. This is important because it stops the model from accidentally using information from the game it is trying to predict.

## Model and evaluation

The project trains one histogram gradient-boosting model for each target: points, rebounds, and assists. Evaluation uses the newest 20% of game dates as a held-out test set, so future games never leak into training.

| Target | Model MAE | Rolling-average baseline MAE |
|---|---:|---:|
| Points | 4.5996 | 4.6764 |
| Rebounds | 1.8942 | 1.9389 |
| Assists | 1.3494 | 1.3714 |

After evaluation, each model is retrained on every available historical row and saved in `models/` for predictions.

## Project files

```text
data/player_games.csv        historical player game data
build_features.py            builds the model-ready feature table
train.py                     trains and evaluates all three models
models/                      saved points, rebounds, and assists models
src/nba_predictor/data.py    loads and cleans the data
src/nba_predictor/features.py creates the model inputs
src/nba_predictor/model.py   trains and runs the models
src/nba_predictor/cli.py     command-line options
main.py                      starts the program
```

## Setup

This project works with Python 3.10 through 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Make a prediction

```bash
python3 main.py predict \
  --data data/player_games.csv \
  --models models \
  --player "Jayson Tatum" \
  --opponent NYK \
  --date 2026-10-23 \
  --home
```

Change the player, opponent, date, and `--home` or `--away` values to predict a different matchup.

Example output:

```json
{
  "player": "Jayson Tatum",
  "opponent": "NYK",
  "date": "2026-10-23",
  "location": "home",
  "prediction": {
    "PTS": 24.0,
    "REB": 8.5,
    "AST": 5.1
  }
}
```

## Other commands

Check that the dataset is formatted correctly:

```bash
python3 main.py validate --data data/player_games.csv
```

Download game logs:

```bash
python3 main.py download \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --output data/player_games.csv
```

Build the features, then train new models:

```bash
python3 build_features.py
python3 train.py
```

`build_features.py` creates `data/features.csv` from the historical games. `train.py` tests the last 20% of game dates, prints the evaluation results, retrains on all available rows, and saves the finished models in `models/`.

The original one-command training option still works:

```bash
python3 main.py train --data data/player_games.csv --models models --report outputs/metrics.json
```

The predictions do not include injuries, minutes limits, or last-minute lineup changes, so they should not be treated as betting advice.
