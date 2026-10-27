from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_NAMES,
    build_integrated_dataset,
    load_datasets,
)

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
GRAPHS_DIR = ROOT / "graphs"
RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def evaluate(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    labels = [0, 1, 2]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_test, predictions, labels=labels,
            target_names=[TARGET_NAMES[label] for label in labels],
            output_dict=True, zero_division=0,
        ),
    }
    if len(set(y_test)) == len(labels):
        metrics["roc_auc_ovr_weighted"] = roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="weighted",
            labels=labels,
        )
    else:
        metrics["roc_auc_ovr_weighted"] = None
    return metrics


def _save_feature_importance(model: Pipeline) -> list[dict[str, float | str]]:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    importance = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(importance["feature"].iloc[::-1], importance["importance"].iloc[::-1])
    axis.set_title("Selected model feature importance")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(GRAPHS_DIR / "feature_importance.png", dpi=150)
    plt.close(figure)
    return importance.to_dict("records")


def train_and_save() -> dict:
    frames = load_datasets()
    integrated = build_integrated_dataset(frames)
    if integrated.empty:
        raise ValueError("No labeled incident records are available for training")
    x = integrated[FEATURE_COLUMNS]
    y = integrated["target"].astype(int)
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", max_depth=8
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced",
            n_jobs=-1, min_samples_leaf=2,
        ),
    }
    evaluations = {}
    trained = {}
    for name, classifier in models.items():
        pipeline = Pipeline(
            [("preprocessor", make_preprocessor()), ("classifier", classifier)]
        )
        pipeline.fit(x_train, y_train)
        evaluations[name] = evaluate(pipeline, x_test, y_test)
        trained[name] = pipeline
    selected_name = max(
        evaluations,
        key=lambda name: (
            evaluations[name]["f1_weighted"],
            evaluations[name]["roc_auc_ovr_weighted"] or 0,
            evaluations[name]["accuracy"],
        ),
    )
    selected_model = trained[selected_name]

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    GRAPHS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "runwaysafe_model.joblib"
    joblib.dump(
        {
            "model": selected_model,
            "feature_columns": FEATURE_COLUMNS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_names": TARGET_NAMES,
        },
        model_path,
    )
    feature_importance = _save_feature_importance(selected_model)
    results = {
        "training_samples": len(x_train),
        "test_samples": len(x_test),
        "class_distribution_total": {
            TARGET_NAMES[int(key)]: int(value) for key, value in y.value_counts().sort_index().items()
        },
        "class_distribution_train": {
            TARGET_NAMES[int(key)]: int(value)
            for key, value in y_train.value_counts().sort_index().items()
        },
        "class_distribution_test": {
            TARGET_NAMES[int(key)]: int(value)
            for key, value in y_test.value_counts().sort_index().items()
        },
        "decision_tree": evaluations["Decision Tree"],
        "random_forest": evaluations["Random Forest"],
        "selected_model": selected_name,
        "model_path": str(model_path),
        "feature_importance": feature_importance,
        "integrated_rows": len(integrated),
    }
    (RESULTS_DIR / "evaluation_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame(integrated).to_csv(RESULTS_DIR / "integrated_runway_dataset.csv", index=False)
    return results


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2, default=float))
