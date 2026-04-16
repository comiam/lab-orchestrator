#!/usr/bin/env python3
"""Schedule an sklearn model*seed sweep via lab-orchestrator.

Run from the ``examples/sklearn_digits/`` directory::

    python launch.py --gpus 0 --dry-run     # preview schedule
    python launch.py --gpus 0               # run all 15 tasks
    python launch.py --gpus 0 --resume      # skip finished seeds
    python launch.py --gpus 0,1 --workers-per-gpu 3

No real GPU is needed - sklearn ignores ``CUDA_VISIBLE_DEVICES``.
Use ``--gpus 0`` (or ``0,1`` for parallel slots) to let the scheduler
distribute tasks across subprocess pools.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

try:
    from lab_orchestrator import (
        build_task_matrix,
        parse_gpu_ids,
        run_schedule,
        scan_completed,
    )
except ImportError:
    sys.exit("lab_orchestrator not found. Install: pip install -e ../../lab-orchestrator")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---- Experiment config ----

MODELS = ["svm", "rf", "knn"]
SEEDS = [42, 43, 44, 45, 46]
WEIGHTS = {
    "svm": 0.01,  # GPU-hours (seconds in practice - sklearn is fast)
    "rf": 0.01,
    "knn": 0.005,
}


def make_cmd(name: str, seed: int) -> list[str]:
    """Build subprocess command for one (model, seed) pair."""
    return [sys.executable, "train.py", "--name", name, "--seed", str(seed)]


# ---- Main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch sklearn digits sweep")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus)
    if not gpu_ids:
        print("No GPUs detected. Use --gpus 0 (sklearn needs no real GPU).", file=sys.stderr)
        sys.exit(1)

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = build_task_matrix(
        names=MODELS,
        seeds=seeds,
        weights=WEIGHTS,
        cmd_factory=make_cmd,
    )

    pre_done = 0
    pre_done_weight = 0.0

    if args.resume:
        done = scan_completed("results")
        remaining = [t for t in tasks if (t.name, t.seed) not in done]
        pre_done = len(tasks) - len(remaining)
        pre_done_weight = sum(t.weight for t in tasks if (t.name, t.seed) in done)
        tasks = remaining

    run_schedule(
        gpu_ids,
        tasks,
        dry_run=args.dry_run,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        pre_done=pre_done,
        pre_done_weight=pre_done_weight,
        workers_per_gpu=args.workers_per_gpu,
    )


if __name__ == "__main__":
    main()
