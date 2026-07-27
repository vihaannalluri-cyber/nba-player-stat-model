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
BOX_SCORE_COLUMNS = ["MIN", "PTS", "REB", "AST"]


def normalize_games(frame):
    """Clean the game data and add a few columns the model needs."""
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

    for column in BOX_SCORE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=BOX_SCORE_COLUMNS)
    frame = frame.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).drop_duplicates(
        ["GAME_ID", "PLAYER_ID"], keep="last"
    )
    return frame.reset_index(drop=True)


def read_games(path):
    return normalize_games(pd.read_csv(path))


def _load_season(season, cache_dir):
    cached_file = cache_dir / f"player_games_{season}.csv"
    if cached_file.exists():
        games = pd.read_csv(cached_file)
    else:
        try:
            from nba_api.stats.endpoints import playergamelogs
        except ImportError as exc:
            raise RuntimeError("Install the project with: pip install -e .") from exc

        response = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=90,
        )
        games = response.get_data_frames()[0]
        games.to_csv(cached_file, index=False)

    games["SEASON"] = season
    return games


def download_seasons(seasons, output, pause=1.0):
    """Download NBA game logs and save a copy of each season."""
    output = Path(output)
    cache_dir = output.parent / "season_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    season_frames = []
    for index, season in enumerate(seasons):
        season_frames.append(_load_season(season, cache_dir))
        if index < len(seasons) - 1:
            time.sleep(pause)

    games = normalize_games(pd.concat(season_frames, ignore_index=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(output, index=False, date_format="%Y-%m-%d")
    return games
