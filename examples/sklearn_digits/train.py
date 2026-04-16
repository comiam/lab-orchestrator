#!/usr/bin/env python3
"""Fit one sklearn model on the digits dataset for a single seed.

Run standalone or let ``launch.py`` call it as a subprocess::

    python train.py --name svm --seed 42
    python train.py --name rf --seed 43

No GPU required - sklearn is CPU-only.  lab-orchestrator sets
``CUDA_VISIBLE_DEVICES`` automatically; this script simply ignores it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -- Model configs -----------------------------------------------------

MODELS: dict[str, dict] = {
    "svm": {"cls": SVC, "params": {"kernel": "rbf", "gamma": "scale"}},
    "rf": {"cls": RandomForestClassifier, "params": {"n_estimators": 100}},
    "knn": {"cls": KNeighborsClassifier, "params": {"n_neighbors": 5}},
}

RESULTS_DIR = "results"


# -- Training ----------------------------------------------------------


def train(name: str, seed: int) -> dict:
    """Fit model *name* with train/test split seeded by *seed*."""
    cfg = MODELS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(MODELS)}")

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    params = cfg["params"].copy()
    # Not every estimator accepts random_state (e.g. KNeighborsClassifier).
    cls = cfg["cls"]
    if "random_state" in cls().get_params():
        params["random_state"] = seed
    model = cls(**params)
    t0 = time.monotonic()
    model.fit(X_train, y_train)
    elapsed = time.monotonic() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info("%s  seed=%d  accuracy=%.4f  (%.2fs)", name, seed, acc, elapsed)

    return {
        "model": name,
        "seed": seed,
        "accuracy": round(acc, 4),
        "wall_sec": round(elapsed, 2),
    }


def save_results(name: str, seed: int, metrics: dict) -> str:
    """Save metrics to ``results/<timestamp>/<name>/seed_<N>.json``."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(RESULTS_DIR, ts, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved -> %s", path)
    return path


# -- CLI ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one model on digits dataset")
    parser.add_argument("--name", required=True, choices=list(MODELS), help="Model name")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    args = parser.parse_args()

    metrics = train(args.name, args.seed)
    save_results(args.name, args.seed, metrics)


if __name__ == "__main__":
    main()
