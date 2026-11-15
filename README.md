# Motor Insurance Quote Conversion — MLOps Pipeline

![CI](https://github.com/adamjavid/motor-sales-mlops/actions/workflows/ci.yml/badge.svg)

A production-style machine learning pipeline that predicts whether a motor insurance quote will convert into a sale, and flags borderline quotes for follow-up.

This started as a data science case study (a single exploratory script) completed for a UK motor insurer's interview process. I rebuilt it as a tested, versioned, automated pipeline to show how a notebook-style model can be turned into something reliable enough to run in production.

## What this project demonstrates

| Area | How it's done here |
|---|---|
| **Productionising a model** | Exploratory script refactored into a Python package with a CLI (`generate-data`, `train`, `score`) |
| **No data leakage** | All imputation, scaling and encoding live inside a scikit-learn `Pipeline`, fitted on training data only |
| **Data validation** | Schema, type and range checks run before training and scoring. On the original dataset, this caught a negative licence length the first version had missed |
| **Automated testing** | 18 pytest tests covering validation, cleaning, pipelines, reproducibility, quality gates and an end-to-end CLI run |
| **CI/CD** | GitHub Actions lints and tests on Python 3.11 and 3.12, then trains a model and publishes it as a build artifact |
| **Quality gate** | Training refuses to save a model whose test AUC falls below a configured minimum |
| **Versioning and traceability** | Every model is saved with a timestamped version and `metadata.json` recording metrics, feature importance, data hash, git commit and config |
| **Business output** | Scoring adds a purchase probability and flags quotes in the 40–60% band for targeted follow-up |

## Results

On the original 50,000-quote dataset (not published), the rebuilt leak-free pipeline achieved:

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.933 | 0.938 | 0.853 | 0.893 |
| **Random Forest** | **0.941** | 0.933 | 0.881 | 0.906 |

Price and tax were by far the strongest predictors, followed by driver age and licence length.

The repository runs on **synthetic data** with the same schema, so results from CI will differ from the table above.

## Project structure

```
src/motor_sales/
  config.py      schema, features and training settings
  data.py        loading, validation and cleaning
  pipeline.py    preprocessing and model pipelines
  train.py       training, evaluation, model selection and versioned saving
  predict.py     batch scoring and follow-up flagging
  synthetic.py   synthetic data generator
  cli.py         command-line interface
tests/           pytest suite
.github/workflows/ci.yml   lint, test, train and publish
```

## Running it

```bash
pip install -e ".[dev]"

motor-sales generate-data --rows 20000     # create synthetic data
motor-sales train                          # train, compare and save the best model
motor-sales score --data data/quotes.csv   # score quotes and flag follow-ups

pytest                                     # run the tests
ruff check .                               # lint
```

## Design decisions

- **Imputation inside the pipeline.** The original script filled missing values on the full dataset before splitting, which leaks information from the test set. Moving this into the pipeline fixes that, and means the saved model handles missing values at scoring time too.
- **Credit score sentinel.** The data uses `9999` to mean "unknown". This is converted to missing before imputation rather than treated as a very high score.
- **Strict validation.** Bad data fails loudly rather than silently producing a model trained on errors.
- **Class weighting.** Both models use balanced class weights to handle the imbalance between sales and non-sales.

## Next steps

- Experiment tracking and a model registry with MLflow
- Monitoring for data drift between training and scoring data
- Model explainability with SHAP
- Probability calibration before using the follow-up band operationally
