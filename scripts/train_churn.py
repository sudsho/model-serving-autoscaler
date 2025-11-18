"""Tiny churn training script kept here just so the deploy story is end-to-end.
Real training would live in its own repo; this is a stand-in that produces
a model.joblib of the shape the sklearn runtime expects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/churn-v3")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    X, y = make_classification(
        n_samples=20000, n_features=10, n_informative=6, random_state=args.seed
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    pipe = Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=200))]
    )
    pipe.fit(Xtr, ytr)
    acc = pipe.score(Xte, yte)
    print(f"holdout accuracy: {acc:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    np.save(out / "feature_means.npy", Xtr.mean(axis=0))
    print(f"wrote {out / 'model.joblib'}")


if __name__ == "__main__":
    main()
