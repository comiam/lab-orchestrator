# Changelog

## v1.0.0 - 2026-04-17

Initial public release.

### Core

- **LPT scheduling** - tasks sorted by weight descending; heavy experiments start first, short ones fill gaps at the end.
- **Load-aware dispatch** - each new task goes to the GPU with the smallest active workload (sum of `Task.weight`).
- **Per-GPU worker pools** - configurable `workers_per_gpu` (default: 1) to overlap CPU-bound preprocessing with GPU kernels.
- **Subprocess isolation** - every task runs in its own process with `CUDA_VISIBLE_DEVICES` set; no shared CUDA contexts.
- **`Task` dataclass** - `(weight, name, seed, cmd)` with comparison by weight for LPT, `.label` property for logging.
- **`build_task_matrix()`** - generates a deduplicated, LPT-sorted task list from a `names * seeds` grid with `cmd_factory` callback; optional `available_names` whitelist.
- **`run_schedule()`** - main entry point; returns `(gpu_id, label, return_code, elapsed_sec)` per task.

### Resume & progress

- **`--resume`** - skip tasks whose `seed_*.json` already exists in the results directory.
- **`scan_completed()`** - walks `results_dir/<timestamp>/<name>/seed_N.json`; supports custom prefix/suffix and optional JSON validation of corrupt/incomplete files.
- **Progress tracking** - `[done/total pct%]` tag injected into every log line via `ProgressFormatter`; ETA based on completed weight; failure count shown as `N!`.

### Robustness

- **`--task-timeout`** - per-task timeout in seconds; timed-out tasks report `rc=-1`.
- **`--max-failures`** - stop scheduling new tasks after N failures, drain in-flight work, report summary.
- **Stall detection** (`stall_warning`) - warns when a GPU has no activity for a configurable duration; logs in-flight task labels.
- **Dead worker recovery** - detects crashed worker processes; marks their in-flight tasks as failed (`rc=-9`) and continues.

### CLI (`python -m lab_orchestrator`)

- Config loading from YAML (`.yaml`/`.yml`, requires `pyyaml`), JSON, or Python (`.py` with `CONFIG` dict).
- Two command formats: `cmd_template` (string with `{name}`/`{seed}` placeholders) and `cmd_parts` (structured `base` + `per_name` + `common`).
- `--gpus 0,1,2,3` - explicit GPU selection; auto-detect via `torch.cuda.device_count()` when omitted.
- `--dry-run` - simulate scheduling with heap-based timeline; print per-GPU task assignments and estimated wall-clock.
- `--workers-per-gpu N` - concurrent workers sharing each GPU.
- `--results-dir` - custom results directory for resume scanning.

### tmux mode

- `--tmux` / `--tmux-session` / `--venv` / `--cwd` - generate a bash script with one tmux window per task instead of running programmatically.
- `generate_tmux_script()` - programmatic API with LPT-based GPU assignment, virtualenv activation, custom `renderer` callback, and `extra_context` passthrough.

### GPU utilities

- `detect_gpus()` - auto-detect CUDA devices via PyTorch; warns about fork-mode CUDA initialization.
- `parse_gpu_ids()` - parse `"0,1,2"` strings or fall back to auto-detect.

### Seed management

- `set_seed()` - fix random state across stdlib `random`, NumPy, PyTorch, TensorFlow, and JAX; auto-detects installed frameworks.
- `deterministic=True` - enable `torch.use_deterministic_algorithms`, set `CUBLAS_WORKSPACE_CONFIG` and `TF_DETERMINISTIC_OPS`.
- `frameworks=` filter and `warn_missing=` flag for selective seeding.

### Logging

- Per-task log files: `logs/<run_timestamp>/gpu<N>_<name>_seed<S>.log` capturing stdout+stderr.
- Summary at end: total GPU-hours, wall-clock time, list of failed tasks with return codes.
- `fmt_duration()` - human-readable duration formatting (`45s`, `2m`, `1h01m`).

### Examples

- `examples/sklearn_digits/` - minimal working example (no GPU needed): 3 models * 5 seeds with SVM, Random Forest, KNN on digits dataset.
- `examples/sweep/` - GPU sweep template with `train.py` + `sweep.py` + `experiments.yaml`.
- `examples/experiment_template.py` - self-contained launcher + training in one file with checkpointing.
- `examples/tmux_example.py` - tmux script generation from code.

### Packaging

- Zero hard dependencies - stdlib `multiprocessing` + `subprocess` only.
- Optional extras: `gpu` (torch >= 2.0), `yaml` (pyyaml >= 6.0), `dev` (ruff, mypy, pytest, flake8).
- `lab-orchestrator` CLI entry point via `pyproject.toml`.
- Python >= 3.10 required.
- MIT license.
- Full type annotations; passes `mypy --disallow-untyped-defs`.
- Test suite: 30+ tests covering task building, GPU parsing, tmux generation, seed reproducibility, resume scanning, progress formatting, dry-run, config loading, and integration.
