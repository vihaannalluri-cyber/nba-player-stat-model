"""Export the trained sklearn models in a small, web-friendly format."""

import json
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDER = ROOT / "models"


def export_model(target):
    pipeline = joblib.load(MODEL_FOLDER / f"{target}.joblib")
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    numeric_imputer = numeric_pipeline.named_steps["impute"]
    scaler = numeric_pipeline.named_steps["scale"]

    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    categorical_imputer = categorical_pipeline.named_steps["impute"]
    encoder = categorical_pipeline.named_steps["onehot"]

    trees = []
    for predictors in model._predictors:
        nodes = predictors[0].nodes
        trees.append(
            [
                [
                    float(node["value"]),
                    int(node["feature_idx"]),
                    float(node["num_threshold"]),
                    bool(node["missing_go_to_left"]),
                    int(node["left"]),
                    int(node["right"]),
                    bool(node["is_leaf"]),
                ]
                for node in nodes
            ]
        )

    numeric_columns = preprocessor.transformers_[0][2]
    categorical_columns = preprocessor.transformers_[1][2]
    return {
        "numeric_columns": numeric_columns,
        "numeric_fill": numeric_imputer.statistics_.tolist(),
        "numeric_mean": scaler.mean_.tolist(),
        "numeric_scale": scaler.scale_.tolist(),
        "categorical_columns": categorical_columns,
        "categorical_fill": categorical_imputer.statistics_.tolist(),
        "categories": [values.tolist() for values in encoder.categories_],
        "baseline": float(model._baseline_prediction[0][0]),
        "trees": trees,
    }


def main():
    exported = {target: export_model(target) for target in ["pts", "reb", "ast"]}
    output = MODEL_FOLDER / "web_models.json"
    output.write_text(json.dumps(exported, separators=(",", ":")))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
