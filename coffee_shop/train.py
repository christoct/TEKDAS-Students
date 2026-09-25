"""Baseline dan regresi pendapatan harian."""
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from data_prep import BASE_DIR, FEATURES, TARGET

PROCESSED = BASE_DIR / "data" / "processed"
ARTIFACTS = BASE_DIR / "artifacts"


def build_metrics(model, X, y):
    pred = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
    }


def main():
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    models = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, l2_regularization=1.0, random_state=42
        ),
    }
    results, fitted = {}, {}
    for name, estimator in models.items():
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        pipe.fit(X_train, y_train)
        results[name] = build_metrics(pipe, X_test, y_test)
        fitted[name] = pipe
    winner = min((k for k in fitted if k != "baseline_mean"), key=lambda k: results[k]["mae"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[winner], ARTIFACTS / "model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps({
        "target": TARGET, "features": FEATURES, "test_rows": len(test),
        "selection_metric": "MAE (lower is better)", "selected_model": winner,
        "models": results,
        "note": "Metrics are on one random test split; no date column or currency metadata was supplied.",
    }, indent=2), encoding="utf-8")
    predictions = test.copy()
    predictions["predicted_revenue"] = fitted[winner].predict(X_test)
    predictions["residual_actual_minus_predicted"] = predictions[TARGET] - predictions["predicted_revenue"]
    predictions.to_csv(ARTIFACTS / "predictions.csv", index=False)
    try:
        importance = fitted[winner].named_steps["model"].feature_importances_
        pd.DataFrame({"feature": FEATURES, "importance": importance}).sort_values(
            "importance", ascending=False
        ).to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    except AttributeError:
        pass
    print(json.dumps({"selected_model": winner, "models": results}, indent=2))


if __name__ == "__main__":
    main()
