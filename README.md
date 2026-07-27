# NBA Player Stat Predictor

Predict a player's points, rebounds, and assists for a future NBA matchup using
only information that would have been known before tipoff.

The model uses:

- rolling 5, 10, and 20-game player form;
- season-to-date player form;
- recent minutes and per-minute production;
- prior head-to-head results (with conservative sample handling);
- opponent recent points/rebounds/assists allowed;
- home/away, days of rest, and back-to-back status.

## Setup

Python 3.10–3.13 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 1. Download historical data

```bash
python -m nba_predictor download \
  --seasons 2021-22 2022-23 2023-24 2024-25 2025-26 \
  --output data/player_games.csv
```

The downloader uses NBA.com through `nba_api`. NBA.com may throttle requests;
the command pauses between seasons and caches each completed season.

You can instead provide your own CSV. Required columns are documented by:

```bash
python -m nba_predictor validate --data data/player_games.csv
```

## 2. Train and evaluate

```bash
python -m nba_predictor train \
  --data data/player_games.csv \
  --models models \
  --report outputs/metrics.json
```

The final 20% of games are held out chronologically. The report compares the
ML model with a rolling-10-game baseline. No random train/test split is used.

## 3. Predict a matchup

```bash
python -m nba_predictor predict \
  --data data/player_games.csv \
  --models models \
  --player "Jayson Tatum" \
  --opponent NYK \
  --date 2026-10-23 \
  --home
```

The future prediction is only as informed as the supplied history. During the
2026–27 season, append completed games and retrain periodically. Injuries and
late lineup news are not present in historical box scores, so the output should
not be treated as betting advice.

## Tests

```bash
python -m unittest discover -s tests -v
```
