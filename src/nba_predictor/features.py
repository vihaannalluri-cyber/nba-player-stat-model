import numpy as np
import pandas as pd


TARGETS = ["PTS", "REB", "AST"]
ROLL_WINDOWS = (5, 10, 20)


def _shifted_rolling(series, window, minimum=1):
    # shift(1) makes sure the current game is never included in its own features
    return series.shift(1).rolling(window, min_periods=minimum).mean()


def _add_player_form(df):
    by_player = df.groupby("PLAYER_ID", sort=False, group_keys=False)
    df["DAYS_REST"] = by_player["GAME_DATE"].diff().dt.days.clip(lower=0, upper=10)
    df["IS_BACK_TO_BACK"] = (df["DAYS_REST"] <= 1).astype(int)
    df["CAREER_GAMES_BEFORE"] = by_player.cumcount()

    for stat in ["MIN"] + TARGETS:
        for window in ROLL_WINDOWS:
            df[f"{stat}_ROLL_{window}"] = by_player[stat].transform(
                lambda values, w=window: _shifted_rolling(values, w)
            )
        df[f"{stat}_EXPANDING"] = by_player[stat].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )

    safe_minutes = df["MIN"].replace(0, np.nan)
    for stat in TARGETS:
        rate_name = f"{stat}_PER_MIN"
        df[rate_name] = df[stat] / safe_minutes
        df[f"{rate_name}_ROLL_10"] = by_player[rate_name].transform(
            lambda values: _shifted_rolling(values, 10)
        )
    return df.drop(columns=[f"{stat}_PER_MIN" for stat in TARGETS])


def _add_matchup_history(df):
    by_matchup = df.groupby(["PLAYER_ID", "OPPONENT"], sort=False, group_keys=False)
    df["H2H_GAMES_BEFORE"] = by_matchup.cumcount()
    for stat in TARGETS:
        df[f"{stat}_H2H_5"] = by_matchup[stat].transform(
            lambda values: _shifted_rolling(values, 5)
        )
    return df


def _add_opponent_form(df):
    # Team totals from a game are the numbers that the other team allowed.
    team_games = (
        df.groupby(["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "OPPONENT"], as_index=False)[TARGETS]
        .sum()
        .rename(columns={stat: f"TEAM_{stat}" for stat in TARGETS})
    )
    allowed = team_games.rename(columns={"OPPONENT": "DEF_TEAM"}).sort_values(
        ["GAME_DATE", "GAME_ID"]
    )
    for stat in TARGETS:
        allowed[f"OPP_ALLOWED_{stat}_10"] = allowed.groupby("DEF_TEAM", sort=False)[
            f"TEAM_{stat}"
        ].transform(lambda values: _shifted_rolling(values, 10))
    allowed_columns = ["GAME_ID", "DEF_TEAM"]
    allowed_columns += [f"OPP_ALLOWED_{stat}_10" for stat in TARGETS]
    allowed = allowed[allowed_columns]
    return df.merge(
        allowed,
        left_on=["GAME_ID", "OPPONENT"],
        right_on=["GAME_ID", "DEF_TEAM"],
        how="left",
    ).drop(columns="DEF_TEAM")


def build_features(games):
    """Turn the game log into the values used by the models."""
    df = games.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).copy()
    df = df.reset_index(drop=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = _add_player_form(df)
    df = _add_matchup_history(df)
    df = _add_opponent_form(df)
    return df


def model_feature_columns():
    numeric = [
        "IS_HOME",
        "DAYS_REST",
        "IS_BACK_TO_BACK",
        "CAREER_GAMES_BEFORE",
        "H2H_GAMES_BEFORE",
    ]
    for stat in ["MIN"] + TARGETS:
        numeric.extend([f"{stat}_ROLL_{w}" for w in ROLL_WINDOWS])
        numeric.append(f"{stat}_EXPANDING")
    for stat in TARGETS:
        numeric.extend(
            [f"{stat}_PER_MIN_ROLL_10", f"{stat}_H2H_5", f"OPP_ALLOWED_{stat}_10"]
        )
    categorical = ["TEAM_ABBREVIATION", "OPPONENT"]
    return numeric, categorical


def add_future_matchup(games, player, opponent, date, is_home):
    matches = games[games["PLAYER_NAME"].str.casefold() == player.casefold()]
    if matches.empty:
        raise ValueError(f"Player not found in data: {player}")
    latest = matches.sort_values("GAME_DATE").iloc[-1]
    team = str(latest["TEAM_ABBREVIATION"])
    game_id = f"FUTURE_{latest['PLAYER_ID']}_{date}"
    row = {column: np.nan for column in games.columns}
    row.update(
        {
            "GAME_ID": game_id,
            "GAME_DATE": pd.Timestamp(date),
            "PLAYER_ID": latest["PLAYER_ID"],
            "PLAYER_NAME": latest["PLAYER_NAME"],
            "TEAM_ABBREVIATION": team,
            "OPPONENT": opponent.upper(),
            "IS_HOME": int(is_home),
            "MATCHUP": f"{team} {'vs.' if is_home else '@'} {opponent.upper()}",
        }
    )
    combined = pd.concat([games, pd.DataFrame([row])], ignore_index=True)
    featured = build_features(combined)
    index = featured.index[featured["GAME_ID"] == game_id]
    if len(index) != 1:
        raise RuntimeError("Could not create future matchup row")
    return featured, int(index[0])
