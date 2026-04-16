#!/usr/bin/env python3
"""Example: generate a tmux launch script from code.

Outputs a bash script to stdout - does NOT run experiments itself::

    python tmux_example.py > run_experiments.sh
    bash run_experiments.sh
    tmux attach -t experiments
"""

from __future__ import annotations

import sys

try:
    from lab_orchestrator import build_task_matrix, generate_tmux_script
except ImportError:
    sys.exit("lab_orchestrator not found. Install: pip install -e ../lab-orchestrator")

NAMES = ["memorization", "masked_instability", "temporal_limit"]
SEEDS = [42, 43]
WEIGHTS = {
    "memorization": 6.5,
    "masked_instability": 1.7,
    "temporal_limit": 7.3,
}


def make_cmd(name: str, seed: int) -> list[str]:
    return ["python", "main.py", name, "--seed", str(seed), "--epochs", "50"]


tasks = build_task_matrix(
    names=NAMES,
    seeds=SEEDS,
    weights=WEIGHTS,
    cmd_factory=make_cmd,
)

script = generate_tmux_script(
    session="experiments",
    tasks=tasks,
    venv_activate="source ../../.venv/bin/activate",
    cwd="$(dirname $0)",
    gpu_ids=[0, 1, 2, 3],
)

print(script)
