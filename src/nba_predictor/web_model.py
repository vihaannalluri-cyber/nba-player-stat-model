"""Run exported prediction trees without loading scikit-learn."""

import json
import math
from pathlib import Path


def load_web_models(model_folder):
    model_path = Path(model_folder) / "web_models.json"
    return json.loads(model_path.read_text())


def _prepare_row(row, model):
    values = []
    for index, column in enumerate(model["numeric_columns"]):
        value = row.get(column)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            value = model["numeric_fill"][index]
        scaled = (float(value) - model["numeric_mean"][index]) / model["numeric_scale"][index]
        values.append(scaled)

    for index, column in enumerate(model["categorical_columns"]):
        value = row.get(column) or model["categorical_fill"][index]
        values.extend(1.0 if value == category else 0.0 for category in model["categories"][index])
    return values


def _predict_one(row, model):
    values = _prepare_row(row, model)
    prediction = model["baseline"]
    for tree in model["trees"]:
        node_index = 0
        while not tree[node_index][6]:
            node = tree[node_index]
            value = values[node[1]]
            if math.isnan(value):
                node_index = node[4] if node[3] else node[5]
            else:
                node_index = node[4] if value <= node[2] else node[5]
        prediction += tree[node_index][0]
    return round(max(0.0, prediction), 1)


def predict(row, models):
    return {target.upper(): _predict_one(row, model) for target, model in models.items()}
