#!/usr/bin/env python3
"""Minimal training script for use with lab-orchestrator.

Run standalone or let a launcher (``sweep.py`` or
``python -m lab_orchestrator experiments.yaml``) call it as a subprocess::

    python train.py --config big --seed 42
    python train.py --config big --seed 42 --epochs 100
    python train.py --config big --seed 42 --resume   # continue from checkpoint

lab-orchestrator sets ``CUDA_VISIBLE_DEVICES`` automatically; this script
should NOT select the GPU itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -- Experiment configs ------------------------------------------------

CONFIGS: dict[str, dict] = {
    "big": {
        "lr": 1e-3,
        "batch_size": 256,
        "hidden_dim": 512,
    },
    "small": {
        "lr": 3e-4,
        "batch_size": 64,
        "hidden_dim": 128,
    },
}

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"


# -- Checkpointing -----------------------------------------------------


def _ckpt_path(config_name: str, seed: int, checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    return os.path.join(checkpoint_dir, config_name, f"seed_{seed}.json")


def save_checkpoint(
    config_name: str, seed: int, state: dict, checkpoint_dir: str = CHECKPOINT_DIR
) -> None:
    """Save training checkpoint (epoch, metrics, model state, etc.)."""
    path = _ckpt_path(config_name, seed, checkpoint_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    logger.info("Checkpoint saved -> %s", path)


def load_checkpoint(
    config_name: str, seed: int, checkpoint_dir: str = CHECKPOINT_DIR
) -> dict | None:
    """Load checkpoint if it exists, else return None."""
    path = _ckpt_path(config_name, seed, checkpoint_dir)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        state: dict = json.load(f)
    logger.info("Resumed from checkpoint -> %s  (epoch %d)", path, state.get("epoch", 0))
    return state


# -- Training logic (replace with your code) ---------------------------


def train(
    config_name: str,
    seed: int,
    epochs: int,
    *,
    resume: bool = False,
    checkpoint_dir: str = CHECKPOINT_DIR,
) -> dict:
    """Run training for one (config, seed) pair.  Returns metrics."""
    cfg = CONFIGS.get(config_name)
    if cfg is None:
        raise ValueError(f"Unknown config {config_name!r}. Available: {', '.join(CONFIGS)}")

    # --- Seed fixation ---
    from lab_orchestrator.seed import set_seed

    set_seed(seed)

    # --- Resume from checkpoint if requested ---
    start_epoch = 1
    if resume:
        ckpt = load_checkpoint(config_name, seed, checkpoint_dir)
        if ckpt is not None:
            start_epoch = ckpt["epoch"] + 1
            # Restore model/optimizer state here:
            # model.load_state_dict(ckpt["model_state"])
            # optimizer.load_state_dict(ckpt["optimizer_state"])

    if start_epoch > epochs:
        logger.info("Already completed %d/%d epochs - nothing to do.", epochs, epochs)
        return {"config": config_name, "seed": seed, "epochs": epochs, "resumed": True}

    # --- Replace below with your actual training loop ---
    logger.info(
        "Training  config=%s  seed=%d  epochs=%d (start=%d)  lr=%s  bs=%d",
        config_name,
        seed,
        epochs,
        start_epoch,
        cfg["lr"],
        cfg["batch_size"],
    )

    t0 = time.monotonic()
    loss = 0.0
    for epoch in range(start_epoch, epochs + 1):
        # Placeholder: replace with model.train() / optimizer.step()
        loss = 1.0 / (epoch + 1)
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info("  epoch %d/%d  loss=%.4f", epoch, epochs, loss)

        # Periodic checkpoint (every 10 epochs)
        if epoch % 10 == 0 and epoch < epochs:
            save_checkpoint(
                config_name,
                seed,
                {
                    "epoch": epoch,
                    "loss": loss,
                    # Add model/optimizer state_dict here
                },
                checkpoint_dir,
            )

    elapsed = time.monotonic() - t0

    metrics = {
        "config": config_name,
        "seed": seed,
        "epochs": epochs,
        "final_loss": loss,
        "wall_sec": round(elapsed, 2),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "resumed_from_epoch": start_epoch if start_epoch > 1 else None,
    }
    return metrics


def save_results(config_name: str, seed: int, metrics: dict, results_dir: str = RESULTS_DIR) -> str:
    """Save metrics to ``results/<timestamp>/<config>/seed_<N>.json``.

    Layout is compatible with ``lab_orchestrator.resume.scan_completed``.
    """
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(results_dir, ts, config_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved -> %s", path)
    return path


# -- CLI ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Training script template")
    parser.add_argument(
        "--config",
        required=True,
        choices=list(CONFIGS),
        help="Experiment config name",
    )
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help="Base directory for result files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Directory for training checkpoints",
    )

    args = parser.parse_args()

    metrics = train(
        args.config,
        args.seed,
        args.epochs,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )
    save_results(args.config, args.seed, metrics, results_dir=args.results_dir)

    # Clean up checkpoint after successful completion
    ckpt = _ckpt_path(args.config, args.seed, args.checkpoint_dir)
    if os.path.isfile(ckpt):
        os.remove(ckpt)
        logger.info("Removed checkpoint %s", ckpt)


if __name__ == "__main__":
    main()
