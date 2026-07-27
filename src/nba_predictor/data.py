from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "GAME_ID",
    "GAME_DATE",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "MATCHUP",
    "MIN",
    "PTS",
    "REB",
    "AST",
}


def normalize_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an NBA player-game export into the project's canonical schema."""
    frame = frame.copy()
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"], errors="raise")
    frame["IS_HOME"] = frame["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    frame["OPPONENT"] = frame["MATCHUP"].str.extract(r"(?:vs\.|@)\s+([A-Z]{3})$")[0]
    if frame["OPPONENT"].isna().any():
        examples = frame.loc[frame["OPPONENT"].isna(), "MATCHUP"].head(3).tolist()
        raise ValueError(f"Could not parse opponent from MATCHUP values: {examples}")

    numeric = ["MIN", "PTS", "REB", "AST"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric)
    frame = frame.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).drop_duplicates(
        ["GAME_ID", "PLAYER_ID"], keep="last"
    )
    return frame.reset_index(drop=True)


def read_games(path: str | Path) -> pd.DataFrame:
    return normalize_games(pd.read_csv(path))


def download_seasons(seasons: list[str], output: str | Path, pause: float = 1.0) -> pd.DataFrame:
    """Download regular-season player game logs, caching one CSV per season."""
    try:
        from nba_api.stats.endpoints import playergamelogs
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    output = Path(output)
    cache = output.parent / "season_cache"
    cache.mkdir(parents=True, exist_ok=True)
    pieces: list[pd.DataFrame] = []
    for season in seasons:
        cached = cache / f"player_games_{season}.csv"
        if cached.exists():
            piece = pd.read_csv(cached)
        else:
            response = playergamelogs.PlayerGameLogs(
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=90,
            )
            piece = response.get_data_frames()[0]
            piece["SEASON"] = season
            piece.to_csv(cached, index=False)
            time.sleep(pause)
        if "SEASON" not in piece:
            piece["SEASON"] = season
        pieces.append(piece)

    games = normalize_games(pd.concat(pieces, ignore_index=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output, index=False, date_format="%Y-%m-%d")
    return games

