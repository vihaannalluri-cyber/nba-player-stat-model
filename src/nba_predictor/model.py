import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import TARGETS, build_features, model_feature_columns


def make_pipeline(numeric, categorical):
    # Fill missing values first, then prepare the numbers and team names.
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0,
    )
    regressor = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=24,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("preprocess", preprocessor), ("model", regressor)])


def chronological_split(frame, test_fraction=0.2):
    dates = np.sort(frame["GAME_DATE"].dropna().unique())
    if len(dates) < 10:
        raise ValueError("At least 10 distinct game dates are required")
    cutoff = dates[max(1, int(len(dates) * (1 - test_fraction)))]
    train = frame[frame["GAME_DATE"] < cutoff]
    test = frame[frame["GAME_DATE"] >= cutoff]
    return train, test, str(pd.Timestamp(cutoff).date())


def _score_model(model, train, test, target):
    predictions = np.clip(model.predict(test), 0, None)
    baseline = test[f"{target}_ROLL_10"].fillna(train[target].mean())
    actual = test[target]

    return {
        "mae": round(float(mean_absolute_error(actual, predictions)), 4),
        "rmse": round(float(mean_squared_error(actual, predictions) ** 0.5), 4),
        "baseline_rolling_10_mae": round(float(mean_absolute_error(actual, baseline)), 4),
    }


def train_models(games, model_dir, report_path):
    featured = build_features(games)
    featured = featured[featured["CAREER_GAMES_BEFORE"] >= 1].copy()
    numeric, categorical = model_feature_columns()
    columns = numeric + categorical
    train, test, cutoff = chronological_split(featured)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "cutoff_date": cutoff,
        "train_rows": len(train),
        "test_rows": len(test),
        "targets": {},
    }

    for target in TARGETS:
        model = make_pipeline(numeric, categorical)
        model.fit(train[columns], train[target])
        report["targets"][target] = _score_model(model, train, test, target)

        model.fit(featured[columns], featured[target])
        joblib.dump(model, model_dir / f"{target.lower()}.joblib")

    metadata = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "targets": TARGETS,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def load_and_predict(featured, row_index, model_dir):
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())
    columns = metadata["numeric_features"] + metadata["categorical_features"]
    row = featured.loc[[row_index], columns]
    result = {}
    for target in metadata["targets"]:
        model = joblib.load(model_dir / f"{target.lower()}.joblib")
        result[target] = round(max(0.0, float(model.predict(row)[0])), 1)
    return result
