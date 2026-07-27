# NBA Player Stat Predictor

Predict an NBA player's points, rebounds, and assists for a future matchup using only information that would have been available before tipoff.

## Overview

This project turns historical NBA player game logs into matchup-specific box-score projections. It builds leakage-safe pregame features, trains a separate gradient-boosting model for each target, evaluates each model chronologically, and exposes predictions through both a command-line runner and a Flask web interface.

## Features

- Points, rebounds, and assists projections.
- Rolling 5-, 10-, and 20-game player form.
- Season-to-date and per-minute production signals.
- Prior head-to-head matchup performance.
- Opponent points, rebounds, and assists allowed over its last 10 games.
- Home/away, rest-days, and back-to-back context.
- Chronological evaluation that prevents future-game leakage.
- Command-line and browser interfaces.

## How it works

1. **Collect and validate data:** `nba_predictor.data` downloads or loads player game logs and normalizes them into one canonical schema.
2. **Build pregame features:** `nba_predictor.features` shifts every rolling statistic so the current game's result cannot leak into its own prediction.
3. **Train models:** `nba_predictor.model` fits one histogram gradient-boosting pipeline for each target and compares it with a rolling-10 baseline.
4. **Predict a matchup:** `main.py` or the Flask app appends a future matchup row, derives its pregame features, and loads the trained models.

## Project structure

```text
.
├── Flask/
│   ├── app.py
│   ├── static/
│   └── templates/
├── data/
│   └── player_games.csv
├── models/
│   ├── ast.joblib
│   ├── pts.joblib
│   ├── reb.joblib
│   └── metadata.json
├── src/nba_predictor/
│   ├── cli.py
│   ├── data.py
│   ├── features.py
│   └── model.py
├── Tests/
│   └── test_web_app.py
├── main.py
├── pyproject.toml
└── requirements.txt
```

## Requirements

- Python 3.10–3.13
- Pandas, NumPy, scikit-learn, joblib, nba_api, and Flask

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick start

Validate the included dataset:

```bash
python3 main.py validate --data data/player_games.csv
```

Generate a matchup projection:

```bash
python3 main.py predict \
  --data data/player_games.csv \
  --models models \
  --player "Jayson Tatum" \
  --opponent NYK \
  --date 2026-10-23 \
  --home
```

## Flask web app

```bash
flask --app Flask/app.py --debug run
```

Open `http://127.0.0.1:5000` and choose a player, opponent, date, and game location.

## Data pipeline and training

Download multiple regular seasons:

```bash
python3 main.py download \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --output data/player_games.csv
```

Train and evaluate all three target models:

```bash
python3 main.py train \
  --data data/player_games.csv \
  --models models \
  --report outputs/metrics.json
```

The final 20% of game dates are held out for evaluation. After evaluation, each model is refit on all available history for future predictions.

## Tests

```bash
python3 -m unittest discover -s Tests -v
```

## Responsible use

Historical box scores do not contain injuries, minutes restrictions, late lineup changes, or every factor that affects a player's next game. These projections are for analytical and educational use and should not be treated as betting advice.

## Development disclosure

The project owner designed the prediction logic, prepared the data, defined the feature and leakage constraints, and architected the machine-learning workflow. AI assistance was used to generate and refine boilerplate implementation code around those requirements.
