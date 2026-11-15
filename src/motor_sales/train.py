"""Train, evaluate, compare and save models with versioned metadata."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from motor_sales.config import TrainConfig
from motor_sales.data import clean, split_xy, validate
from motor_sales.pipeline import build_model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": round(roc_auc_score(y, proba), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
    }


def feature_importance(model, top_n: int = 10) -> list[dict]:
    clf = model.named_steps["model"]
    if not hasattr(clf, "feature_importances_"):
        return []
    names = model.named_steps["preprocess"].get_feature_names_out()
    pairs = sorted(zip(names, clf.feature_importances_, strict=True), key=lambda p: -p[1])
    return [{"feature": n, "importance": round(float(i), 4)} for n, i in pairs[:top_n]]


def _data_hash(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(raw).hexdigest()[:12]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def train(df: pd.DataFrame, output_dir: str | Path, config: TrainConfig | None = None) -> dict:
    """Train all configured models, pick the best by test AUC, and save it
    together with a metadata file for traceability."""
    config = config or TrainConfig()
    validate(df)
    X, y = split_xy(clean(df))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )

    results = {}
    fitted = {}
    for name in config.models:
        model = build_model(name, config.random_state).fit(X_train, y_train)
        fitted[name] = model
        results[name] = evaluate(model, X_test, y_test)

    best_name = max(results, key=lambda n: results[n]["auc"])
    best = fitted[best_name]
    if results[best_name]["auc"] < config.min_auc:
        raise RuntimeError(
            f"Best model AUC {results[best_name]['auc']} is below the "
            f"minimum of {config.min_auc}; model not saved."
        )

    version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    metadata = {
        "version": version,
        "best_model": best_name,
        "metrics": results,
        "feature_importance": feature_importance(best),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "data_hash": _data_hash(df),
        "git_commit": _git_commit(),
        "config": config.__dict__,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, output_dir / f"model_{version}.joblib")
    joblib.dump(best, output_dir / "model_latest.joblib")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
