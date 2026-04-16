"""Tests for lab-orchestrator package."""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from lab_orchestrator._fmt import fmt_duration
from lab_orchestrator.gpu import parse_gpu_ids
from lab_orchestrator.resume import scan_completed
from lab_orchestrator.seed import set_seed
from lab_orchestrator.task import Task, build_task_matrix

# =====================================================================
#  Task
# =====================================================================


class TestTask:
    def test_label(self):
        t = Task(weight=1.0, name="exp", seed=42, cmd=["echo"])
        assert t.label == "exp[seed=42]"

    def test_ordering_by_weight(self):
        heavy = Task(weight=5.0, name="big", seed=1, cmd=[])
        light = Task(weight=0.5, name="small", seed=1, cmd=[])
        assert heavy > light
        assert sorted([light, heavy], reverse=True) == [heavy, light]

    def test_build_task_matrix_lpt_order(self):
        tasks = build_task_matrix(
            names=["a", "b"],
            seeds=[1, 2],
            weights={"a": 3.0, "b": 1.0},
            cmd_factory=lambda n, s: ["echo", n, str(s)],
        )
        assert len(tasks) == 4
        # LPT: heaviest first
        assert tasks[0].weight == 3.0
        assert tasks[-1].weight == 1.0

    def test_build_task_matrix_dedup(self):
        tasks = build_task_matrix(
            names=["x", "x"],
            seeds=[1, 1],
            weights={"x": 1.0},
            cmd_factory=lambda n, s: ["echo"],
        )
        assert len(tasks) == 1

    def test_build_task_matrix_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown name"):
            build_task_matrix(
                names=["missing"],
                seeds=[1],
                weights={},
                cmd_factory=lambda n, s: [],
                available_names=["valid"],
            )

    def test_build_task_matrix_default_weight(self):
        tasks = build_task_matrix(
            names=["no_weight"],
            seeds=[42],
            weights={},
            cmd_factory=lambda n, s: ["echo"],
        )
        assert tasks[0].weight == 1.0

    def test_cmd_factory_receives_correct_args(self):
        calls = []

        def factory(name, seed):
            calls.append((name, seed))
            return ["run", name, str(seed)]

        build_task_matrix(
            names=["a"],
            seeds=[42, 43],
            weights={"a": 1.0},
            cmd_factory=factory,
        )
        assert calls == [("a", 42), ("a", 43)]


# =====================================================================
#  GPU parsing
# =====================================================================


class TestGpuParsing:
    def test_csv_parse(self):
        assert parse_gpu_ids("0,1,2") == [0, 1, 2]

    def test_csv_with_spaces(self):
        assert parse_gpu_ids(" 0 , 1 , 2 ") == [0, 1, 2]

    def test_none_returns_detect(self):
        # Without torch, returns empty list
        result = parse_gpu_ids(None)
        assert isinstance(result, list)

    def test_single_zero_is_gpu_index(self):
        assert parse_gpu_ids("0") == [0]

    def test_single_number_is_gpu_index(self):
        assert parse_gpu_ids("3") == [3]

    def test_trailing_comma(self):
        assert parse_gpu_ids("0,") == [0]


# =====================================================================
#  tmux
# =====================================================================


class TestTmux:
    @pytest.fixture()
    def sample_tasks(self):
        return [
            Task(weight=2.0, name="big", seed=42, cmd=["python", "train.py", "--big"]),
            Task(
                weight=1.0,
                name="small",
                seed=43,
                cmd=["python", "train.py", "--small"],
            ),
        ]

    def test_default_template_has_session(self, sample_tasks):
        from lab_orchestrator.tmux import generate_tmux_script

        script = generate_tmux_script("test_ses", sample_tasks)
        assert 'SESSION="test_ses"' in script
        assert "tmux new-session" in script
        assert "tmux new-window" in script

    def test_gpu_round_robin(self, sample_tasks):
        from lab_orchestrator.tmux import generate_tmux_script

        script = generate_tmux_script("s", sample_tasks, gpu_ids=[0, 1])
        assert "CUDA_VISIBLE_DEVICES=0" in script
        assert "CUDA_VISIBLE_DEVICES=1" in script

    def test_venv_and_cwd(self, sample_tasks):
        from lab_orchestrator.tmux import generate_tmux_script

        script = generate_tmux_script(
            "s",
            sample_tasks,
            venv_activate="source .venv/bin/activate",
            cwd="/work",
        )
        assert "source .venv/bin/activate" in script
        assert "cd /work" in script

    def test_custom_renderer(self, sample_tasks):
        from lab_orchestrator.tmux import generate_tmux_script

        def renderer(ctx: dict) -> str:
            return "\n".join(w["name"] for w in ctx["windows"])

        result = generate_tmux_script("s", sample_tasks, renderer=renderer)
        assert "big_s42" in result
        assert "small_s43" in result

    def test_custom_renderer_receives_context(self, sample_tasks):
        from lab_orchestrator.tmux import generate_tmux_script

        captured: dict = {}

        def renderer(ctx: dict) -> str:
            captured.update(ctx)
            return ""

        generate_tmux_script(
            "s",
            sample_tasks,
            gpu_ids=[0],
            extra_context={"foo": "bar"},
            renderer=renderer,
        )
        assert captured["session"] == "s"
        assert len(captured["windows"]) == 2
        assert captured["gpu_ids"] == [0]
        assert captured["foo"] == "bar"

    def test_window_count_matches_tasks(self, sample_tasks):
        from lab_orchestrator.tmux import generate_tmux_script

        script = generate_tmux_script("s", sample_tasks)
        assert "2 windows" in script


# =====================================================================
#  Seed
# =====================================================================


class TestSeed:
    def test_set_seed_returns_random(self):
        seeded = set_seed(42)
        assert "random" in seeded

    def test_set_seed_numpy(self):
        pytest.importorskip("numpy")
        import numpy as np

        seeded = set_seed(42, frameworks=["numpy"])
        assert "numpy" in seeded
        a = np.random.rand(3)
        set_seed(42, frameworks=["numpy"])
        b = np.random.rand(3)
        assert list(a) == list(b)

    def test_set_seed_torch(self):
        torch = pytest.importorskip("torch")
        seeded = set_seed(42, frameworks=["torch"])
        assert "torch" in seeded
        a = torch.rand(3)
        set_seed(42, frameworks=["torch"])
        b = torch.rand(3)
        assert torch.equal(a, b)

    def test_set_seed_missing_framework_no_warn(self):
        seeded = set_seed(42, frameworks=["nonexistent_lib"])
        assert "nonexistent_lib" not in seeded

    def test_set_seed_warn_missing(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            set_seed(42, frameworks=["nonexistent_lib"], warn_missing=True)
        assert "Unknown framework" in caplog.text

    def test_deterministic_sets_env(self, monkeypatch):
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        set_seed(42, deterministic=True, frameworks=[])
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"

    def test_stdlib_random_reproducible(self):
        import random

        set_seed(123)
        a = [random.random() for _ in range(5)]
        set_seed(123)
        b = [random.random() for _ in range(5)]
        assert a == b


# =====================================================================
#  Resume / scan_completed
# =====================================================================


class TestResume:
    def test_empty_dir(self, tmp_path):
        assert scan_completed(str(tmp_path)) == set()

    def test_nonexistent_dir(self, tmp_path):
        assert scan_completed(str(tmp_path / "nope")) == set()

    def test_finds_seeds(self, tmp_path):
        # results/20260101/baseline/seed_42.json
        d = tmp_path / "20260101" / "baseline"
        d.mkdir(parents=True)
        (d / "seed_42.json").write_text("{}")
        (d / "seed_43.json").write_text("{}")

        done = scan_completed(str(tmp_path))
        assert done == {("baseline", 42), ("baseline", 43)}

    def test_custom_prefix_suffix(self, tmp_path):
        d = tmp_path / "ts" / "exp"
        d.mkdir(parents=True)
        (d / "run_7.pkl").write_text("")

        done = scan_completed(
            str(tmp_path),
            seed_prefix="run_",
            seed_suffix=".pkl",
        )
        assert done == {("exp", 7)}


# =====================================================================
#  Formatting
# =====================================================================


class TestFmtDuration:
    def test_seconds(self):
        assert fmt_duration(45) == "45s"

    def test_minutes(self):
        assert fmt_duration(120) == "2m"

    def test_hours(self):
        assert fmt_duration(3661) == "1h01m"

    def test_multi_hours(self):
        assert fmt_duration(7200) == "2h00m"


# =====================================================================
#  Progress
# =====================================================================


class TestProgress:
    def test_update_sets_tag(self):
        import time

        from lab_orchestrator.progress import _progress_ctx, update_progress

        t0 = time.monotonic()
        update_progress(
            done=5,
            total=10,
            done_weight=5.0,
            total_weight=10.0,
            wall_t0=t0 - 100,
            failed=0,
        )
        tag = _progress_ctx["tag"]
        assert "5/10" in tag
        assert "50%" in tag

    def test_set_done_tag(self):
        from lab_orchestrator.progress import _progress_ctx, set_done_tag

        set_done_tag()
        assert _progress_ctx["tag"] == "[DONE]"

    def test_failed_count_in_tag(self):
        import time

        from lab_orchestrator.progress import _progress_ctx, update_progress

        update_progress(3, 10, 3.0, 10.0, time.monotonic() - 1, 2)
        assert "2!" in _progress_ctx["tag"]


# =====================================================================
#  Scheduler dry-run
# =====================================================================


class TestSchedulerDryRun:
    def test_dry_run_returns_empty(self):
        from lab_orchestrator.scheduler import run_schedule

        tasks = [
            Task(weight=1.0, name="a", seed=42, cmd=["echo"]),
            Task(weight=2.0, name="b", seed=42, cmd=["echo"]),
        ]
        result = run_schedule([0], tasks, dry_run=True)
        assert result == []


# =====================================================================
#  __main__ config loading
# =====================================================================


class TestConfigLoading:
    def test_load_json_config(self, tmp_path):
        from lab_orchestrator.__main__ import load_config

        cfg = {
            "names": ["a", "b"],
            "seeds": [42],
            "weights": {"a": 1.0, "b": 2.0},
            "cmd_template": "python run.py {name} --seed {seed}",
        }
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps(cfg))
        loaded = load_config(str(p))
        assert loaded["names"] == ["a", "b"]

    def test_load_yaml_config(self, tmp_path):
        pytest.importorskip("yaml")
        from lab_orchestrator.__main__ import load_config

        p = tmp_path / "cfg.yaml"
        p.write_text(
            textwrap.dedent(
                """\
            names:
              - a
              - b
            seeds: [42, 43]
            weights:
              a: 1.0
              b: 2.0
            cmd_template: "python run.py {name} --seed {seed}"
        """
            )
        )
        loaded = load_config(str(p))
        assert loaded["seeds"] == [42, 43]

    def test_load_python_config(self, tmp_path):
        from lab_orchestrator.__main__ import load_config

        p = tmp_path / "cfg.py"
        p.write_text(
            textwrap.dedent(
                """\
            CONFIG = {
                "names": ["x"],
                "seeds": [1],
                "weights": {"x": 0.5},
                "cmd_template": "echo {name} {seed}",
            }
        """
            )
        )
        loaded = load_config(str(p))
        assert loaded["names"] == ["x"]

    def test_unsupported_format_raises(self, tmp_path):
        from lab_orchestrator.__main__ import load_config

        with pytest.raises(ValueError, match="Unsupported"):
            load_config(str(tmp_path / "cfg.toml"))

    def test_build_tasks_from_template_config(self, tmp_path):
        from lab_orchestrator.__main__ import _build_tasks_from_config

        cfg = {
            "names": ["train"],
            "seeds": [42, 43],
            "weights": {"train": 2.0},
            "cmd_template": "python train.py {name} --seed {seed}",
        }
        tasks = _build_tasks_from_config(cfg)
        assert len(tasks) == 2
        # Same weight -> deterministic tiebreak by (name, seed) descending.
        assert tasks[0].cmd == ["python", "train.py", "train", "--seed", "43"]

    def test_build_tasks_from_parts_config(self):
        from lab_orchestrator.__main__ import _build_tasks_from_config

        cfg = {
            "names": ["alpha"],
            "seeds": [1],
            "weights": {},
            "cmd_parts": {
                "base": ["python", "main.py"],
                "per_name": {"alpha": ["--lr", "0.01"]},
                "common": ["--epochs", "10"],
            },
        }
        tasks = _build_tasks_from_config(cfg)
        assert len(tasks) == 1
        assert "--lr" in tasks[0].cmd
        assert "--epochs" in tasks[0].cmd

    def test_missing_cmd_spec_raises(self):
        from lab_orchestrator.__main__ import _build_tasks_from_config

        with pytest.raises(ValueError, match="cmd_template.*cmd_parts"):
            _build_tasks_from_config({"names": ["a"], "seeds": [1]})


# =====================================================================
#  Integration: build -> tmux pipeline
# =====================================================================


class TestIntegration:
    def test_build_to_tmux(self):
        from lab_orchestrator.tmux import generate_tmux_script

        tasks = build_task_matrix(
            names=["train", "eval"],
            seeds=[42, 43],
            weights={"train": 2.0, "eval": 0.5},
            cmd_factory=lambda n, s: ["python", "run.py", n, "--seed", str(s)],
        )
        script = generate_tmux_script(
            "test",
            tasks,
            gpu_ids=[0, 1],
        )
        assert "tmux new-session" in script
        # 4 tasks = 4 windows
        assert "4 windows" in script
        # Heaviest first (LPT)
        assert script.index("train") < script.index("eval")
