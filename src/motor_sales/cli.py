"""Command-line interface: generate data, train, and score."""

import argparse
import json
from pathlib import Path

from motor_sales.data import load_quotes
from motor_sales.predict import load_model, score
from motor_sales.synthetic import generate
from motor_sales.train import train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="motor-sales")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-data", help="Create a synthetic dataset")
    gen.add_argument("--rows", type=int, default=10_000)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out", default="data/quotes.csv")

    tr = sub.add_parser("train", help="Train and save the best model")
    tr.add_argument("--data", default="data/quotes.csv")
    tr.add_argument("--out", default="models")

    sc = sub.add_parser("score", help="Score quotes with a saved model")
    sc.add_argument("--data", required=True)
    sc.add_argument("--model", default="models/model_latest.joblib")
    sc.add_argument("--out", default="data/scored.csv")

    args = parser.parse_args(argv)

    if args.command == "generate-data":
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        generate(args.rows, args.seed).to_csv(args.out, index=False)
        print(f"Wrote {args.rows} rows to {args.out}")
    elif args.command == "train":
        metadata = train(load_quotes(args.data), args.out)
        print(json.dumps({k: metadata[k] for k in ("version", "best_model", "metrics")}, indent=2))
    elif args.command == "score":
        scored = score(load_model(args.model), load_quotes(args.data))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.out, index=False)
        flagged = int(scored["follow_up"].sum())
        print(f"Scored {len(scored)} quotes; {flagged} flagged for follow-up")


if __name__ == "__main__":
    main()
