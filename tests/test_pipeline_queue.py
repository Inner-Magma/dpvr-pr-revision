"""GPU job queue: expansion, dependency order, GPU assignment, resume."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dpvr.pipeline import Scheduler, expand_jobs, parse_gpu_list  # noqa: E402


def _probe(n=4, used=None):
    used = used or {}
    return lambda: {i: (used.get(i, 0), 81920) for i in range(n)}


def _run(spec, tmp_path, gpus=(0, 1, 2, 3), probe=None, **kw):
    jobs = expand_jobs(spec, tmp_path, kw.pop("overrides", None))
    s = Scheduler(jobs, list(gpus), tmp_path, spec["experiment"], gpu_probe=probe or _probe(),
                  poll_s=0.02, log=lambda *_: None, **kw)
    return s.run()


def test_parse_gpu_list():
    assert parse_gpu_list("0-3,6") == [0, 1, 2, 3, 6]
    assert parse_gpu_list("5") == [5]


def test_matrix_and_placeholders(tmp_path):
    spec = {"experiment": "E9", "vars": {"cfg": "a.yaml"}, "jobs": [
        {"name": "train_s{seed}", "matrix": {"seed": [1, 2]}, "cmd": "train {cfg} seed={seed} ${HOME}"},
        {"name": "eval_s{seed}", "matrix": {"seed": [1, 2]}, "after": ["train_s{seed}"],
         "cmd": "eval --out {out}"},
    ]}
    jobs = {j.name: j for j in expand_jobs(spec, tmp_path)}
    assert set(jobs) == {"train_s1", "train_s2", "eval_s1", "eval_s2"}
    assert jobs["train_s2"].cmd == "train a.yaml seed=2 ${HOME}"
    assert jobs["eval_s1"].after == ["train_s1"]
    assert jobs["eval_s1"].cmd == f"eval --out {tmp_path / 'E9' / 'eval_s1'}"


def test_unknown_dependency_rejected(tmp_path):
    spec = {"experiment": "X", "jobs": [{"name": "a", "cmd": "true", "after": ["nope"]}]}
    with pytest.raises(ValueError):
        expand_jobs(spec, tmp_path)


def test_parallel_jobs_get_distinct_gpus(tmp_path):
    spec = {"experiment": "X", "jobs": [
        {"name": "j{i}", "matrix": {"i": [0, 1, 2]},
         "cmd": "echo $CUDA_VISIBLE_DEVICES > {out}/gpu.txt; sleep 0.3"},
    ]}
    final = _run(spec, tmp_path)
    assert set(final.values()) == {"done"}
    seen = [(tmp_path / "X" / f"j{i}" / "gpu.txt").read_text().strip() for i in range(3)]
    assert len(set(seen)) == 3


def test_busy_gpu_is_skipped_and_multi_gpu_job(tmp_path):
    spec = {"experiment": "X", "jobs": [
        {"name": "big", "gpus": 2, "cmd": "echo $CUDA_VISIBLE_DEVICES > {out}/gpu.txt"},
    ]}
    final = _run(spec, tmp_path, probe=_probe(used={0: 30000}))
    assert final == {"big": "done"}
    assert (tmp_path / "X" / "big" / "gpu.txt").read_text().strip() == "1,2"


def test_dependency_order_and_failure_blocks(tmp_path):
    spec = {"experiment": "X", "jobs": [
        {"name": "a", "cmd": "date +%s%N > {out}/t"},
        {"name": "b", "after": ["a"], "cmd": "date +%s%N > {out}/t"},
        {"name": "bad", "cmd": "exit 3"},
        {"name": "c", "after": ["bad"], "cmd": "true"},
        {"name": "d", "after": ["c"], "cmd": "true"},
    ]}
    final = _run(spec, tmp_path)
    assert final == {"a": "done", "b": "done", "bad": "failed", "c": "blocked", "d": "blocked"}
    ta = int((tmp_path / "X" / "a" / "t").read_text())
    tb = int((tmp_path / "X" / "b" / "t").read_text())
    assert ta < tb
    status = json.loads((tmp_path / "X" / "bad" / "status.json").read_text())
    assert status["returncode"] == 3


def test_resume_skips_done_jobs(tmp_path):
    spec = {"experiment": "X", "jobs": [{"name": "a", "cmd": "echo run >> {out}/count"}]}
    _run(spec, tmp_path)
    _run(spec, tmp_path)
    assert (tmp_path / "X" / "a" / "count").read_text().count("run") == 1
    _run(spec, tmp_path, force=True)
    assert (tmp_path / "X" / "a" / "count").read_text().count("run") == 2


def test_required_var_refuses_to_start(tmp_path):
    spec = {"experiment": "X", "vars": {"ckpt": "REQUIRED"}, "jobs": [
        {"name": "a", "cmd": "touch {out}/ran {ckpt}"},
        {"name": "b", "after": ["a"], "cmd": "true"},
    ]}
    assert _run(spec, tmp_path) == {"a": "failed", "b": "blocked"}
    assert not (tmp_path / "X" / "a" / "ran").exists()
    assert _run(spec, tmp_path, overrides={"ckpt": str(tmp_path / "c")}) == {"a": "done", "b": "done"}


def test_cpu_job_runs_without_gpus(tmp_path):
    spec = {"experiment": "X", "jobs": [
        {"name": "cpu", "gpus": 0, "cmd": "test -z \"$CUDA_VISIBLE_DEVICES\""},
    ]}
    assert _run(spec, tmp_path, probe=_probe(used={i: 50000 for i in range(4)})) == {"cpu": "done"}


def test_repo_job_files_expand(tmp_path):
    from dpvr.pipeline import load_job_file

    root = Path(__file__).resolve().parent.parent / "jobs"
    for f in sorted(root.glob("*.yaml")):
        jobs = expand_jobs(load_job_file(f), tmp_path)
        assert jobs, f
