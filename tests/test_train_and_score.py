import json

import pytest

from motor_sales.cli import main
from motor_sales.config import FEATURES, TrainConfig
from motor_sales.predict import load_model, score
from motor_sales.train import train


@pytest.fixture(scope="module")
def trained(quotes, tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    return train(quotes, out), out


def test_model_meets_quality_threshold(trained):
    metadata, _ = trained
    best = metadata["best_model"]
    assert metadata["metrics"][best]["auc"] >= TrainConfig().min_auc


def test_artifacts_and_metadata_saved(trained):
    metadata, out = trained
    assert (out / "model_latest.joblib").exists()
    assert (out / f"model_{metadata['version']}.joblib").exists()
    saved = json.loads((out / "metadata.json").read_text())
    for key in ("version", "best_model", "metrics", "data_hash", "git_commit"):
        assert key in saved


def test_training_refuses_weak_model(quotes, tmp_path):
    with pytest.raises(RuntimeError, match="below the minimum"):
        train(quotes, tmp_path, TrainConfig(min_auc=0.999))
    assert not (tmp_path / "model_latest.joblib").exists()


def test_scoring_adds_probability_and_follow_up_flag(trained, quotes):
    _, out = trained
    scored = score(load_model(out / "model_latest.joblib"), quotes[FEATURES])
    assert scored["purchase_probability"].between(0, 1).all()
    flagged = scored[scored["follow_up"]]
    assert flagged["purchase_probability"].between(0.4, 0.6).all()


def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "quotes.csv"
    models = tmp_path / "models"
    main(["generate-data", "--rows", "2000", "--out", str(data)])
    main(["train", "--data", str(data), "--out", str(models)])
    main(["score", "--data", str(data), "--model", str(models / "model_latest.joblib"),
          "--out", str(tmp_path / "scored.csv")])
    assert (tmp_path / "scored.csv").exists()
    assert "flagged for follow-up" in capsys.readouterr().out
