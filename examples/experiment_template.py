#!/usr/bin/env python3
"""Experiment template - self-contained launcher + training in one file.

Copy this file into your experiment directory and customise:
  1. Define your model & training logic in ``run_experiment()``.
  2. Adjust ``EXPERIMENTS`` / ``WEIGHTS`` / ``SEEDS`` for your sweep.
  3. Launch:

     python launch.py launch --gpus 0,1 --dry-run
     python launch.py launch --gpus 0,1
     python launch.py train --name baseline --seed 42   # one run

Unlike ``sweep/sweep.py`` (which delegates to a separate ``train.py``), this
template keeps training logic and scheduling in a single script.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

try:
    from lab_orchestrator import (
        build_task_matrix,
        parse_gpu_ids,
        run_schedule,
        scan_completed,
        set_seed,
    )
except ImportError:
    sys.exit("lab_orchestrator not found. Install: pip install -e ../lab-orchestrator")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
#  Experiment configuration - EDIT THIS SECTION
# =====================================================================

EXPERIMENTS = ["baseline", "augmented", "large_model"]
SEEDS = [42, 43, 44, 45, 46]

# Estimated GPU-hours per seed (used for LPT scheduling).
# Calibrate: run one seed, note wall time -> weight = hours.
WEIGHTS = {
    "baseline": 1.0,
    "augmented": 1.5,
    "large_model": 3.0,
}

RESULTS_DIR = "results"


# =====================================================================
#  Training logic - REPLACE WITH YOUR CODE
# =====================================================================


def run_experiment(name: str, seed: int, epochs: int) -> dict:
    """Run a single (experiment, seed) pair.  Returns metrics dict."""
    set_seed(seed, deterministic=True)

    # -- Replace this block with your actual model / training loop --
    logger.info("Training %s  seed=%d  epochs=%d", name, seed, epochs)

    # Placeholder: import your model, dataset, trainer
    # from my_models import build_model
    # from my_data import load_dataset
    # model = build_model(name)
    # dataset = load_dataset()
    # trainer = Trainer(model, dataset, epochs=epochs)
    # metrics = trainer.fit()

    metrics = {
        "experiment": name,
        "seed": seed,
        "epochs": epochs,
        "final_loss": 0.42,  # placeholder
        "final_acc": 0.95,  # placeholder
    }
    # -- End of your training code --

    return metrics


def save_results(name: str, seed: int, metrics: dict) -> str:
    """Save metrics to results/<timestamp>/<name>/seed_<N>.json."""
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(RESULTS_DIR, ts, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved -> %s", path)
    return path


# =====================================================================
#  Entry points
# =====================================================================


def train_single(args: argparse.Namespace) -> None:
    """Called by the scheduler subprocess - runs ONE (name, seed)."""
    metrics = run_experiment(args.name, args.seed, args.epochs)
    save_results(args.name, args.seed, metrics)


def launch(args: argparse.Namespace) -> None:
    """Build task matrix and dispatch across GPUs."""
    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        logger.error("No GPUs detected.  Use --gpus to specify.")
        sys.exit(1)

    seeds = [int(s) for s in args.seeds.split(",")]

    def cmd_factory(name: str, seed: int) -> list[str]:
        return [
            sys.executable,
            __file__,
            "train",
            "--name",
            name,
            "--seed",
            str(seed),
            "--epochs",
            str(args.epochs),
        ]

    tasks = build_task_matrix(
        names=EXPERIMENTS,
        seeds=seeds,
        weights=WEIGHTS,
        cmd_factory=cmd_factory,
    )

    # Resume: filter already-completed seeds.
    pre_done = 0
    pre_done_weight = 0.0
    if args.resume:
        done = scan_completed(RESULTS_DIR)
        remaining = [t for t in tasks if (t.name, t.seed) not in done]
        pre_done = len(tasks) - len(remaining)
        pre_done_weight = sum(t.weight for t in tasks if (t.name, t.seed) in done)
        if pre_done:
            logger.info("--resume: skipping %d completed tasks", pre_done)
        tasks = remaining

    if not tasks:
        logger.info("All tasks already completed.")
        return

    run_schedule(
        gpu_ids,
        tasks,
        dry_run=args.dry_run,
        pre_done=pre_done,
        pre_done_weight=pre_done_weight,
        workers_per_gpu=args.workers_per_gpu,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment launcher")
    sub = parser.add_subparsers(dest="command")

    # -- Sub-command: train (one experiment) --
    p_train = sub.add_parser("train", help="Run single experiment")
    p_train.add_argument("--name", required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--epochs", type=int, default=50)

    # -- Sub-command: launch (schedule all) --
    p_launch = sub.add_parser("launch", help="Schedule experiment sweep")
    p_launch.add_argument("--gpus", type=str, default=None)
    p_launch.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
    )
    p_launch.add_argument("--epochs", type=int, default=50)
    p_launch.add_argument("--dry-run", action="store_true")
    p_launch.add_argument("--resume", action="store_true")
    p_launch.add_argument("--workers-per-gpu", type=int, default=1)

    args = parser.parse_args()

    if args.command == "train":
        train_single(args)
    elif args.command == "launch":
        launch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
