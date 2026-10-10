"""A small multi-GPU job queue for the revision experiments.

A job file (``jobs/E*.yaml``) lists shell commands. Each job asks for a number
of GPUs and may depend on other jobs. The scheduler starts a job as soon as
its dependencies have succeeded and enough GPUs are free, so independent jobs
(three training seeds, eight evaluations) run side by side on an 8-GPU server.

Job file format::

    experiment: E9
    vars:                      # {name} placeholders; override with --set name=value
      ckpt: REQUIRED           # REQUIRED makes jobs that use it refuse to start
    defaults:
      gpus: 1
      min_free_gb: 70
    jobs:
      - name: train_seed{seed}
        matrix: {seed: [1, 2, 3]}
        cmd: python scripts/train.py ... seed={seed}
      - name: eval_seed{seed}
        matrix: {seed: [1, 2, 3]}
        after: ["train_seed{seed}"]   # quote: {..} inside [..] is YAML syntax
        cmd: python scripts/revision/eval_lmms.py --out {out} ...

Placeholders: ``{out}`` is the job's run directory, ``{experiment}`` the
experiment id, matrix keys and ``vars`` the rest. Unknown ``{...}`` (for
example shell ``${VAR}``) are left untouched. Commands run under ``bash -c``
from the repository root with ``CUDA_VISIBLE_DEVICES`` set to the assigned
GPUs.

Run directory layout (``<run_root>/<experiment>/<job>/``): ``cmd.sh``,
``log.txt``, ``status.json`` and, on success, ``DONE``. Jobs with ``DONE`` are
skipped on the next invocation, so a queue can be re-run after a crash.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

REQUIRED = "REQUIRED"
_PLACEHOLDER = re.compile(r"(?<!\$)\{(\w+)\}")

# GPU probe: returns {gpu_index: (used_mb, total_mb)}.
GpuProbe = Callable[[], dict[int, tuple[int, int]]]


@dataclass
class Job:
    name: str
    cmd: str
    gpus: int = 1
    after: list[str] = field(default_factory=list)
    min_free_gb: float = 0.0
    env: dict[str, str] = field(default_factory=dict)
    missing_vars: list[str] = field(default_factory=list)


def _fill(text: str, mapping: dict[str, object]) -> str:
    return _PLACEHOLDER.sub(
        lambda m: str(mapping[m.group(1)]) if m.group(1) in mapping else m.group(0), text
    )


def parse_gpu_list(spec: str) -> list[int]:
    """'0-3,6' -> [0, 1, 2, 3, 6]."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_job_file(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    if "experiment" not in spec or "jobs" not in spec:
        raise ValueError(f"{path}: job file needs 'experiment' and 'jobs'")
    return spec


def expand_jobs(
    spec: dict,
    run_root: str | Path,
    overrides: dict[str, str] | None = None,
) -> list[Job]:
    """Expand matrices and placeholders into concrete jobs."""
    experiment = str(spec["experiment"])
    variables = {k: v for k, v in (spec.get("vars") or {}).items()}
    variables.update(overrides or {})
    defaults = spec.get("defaults") or {}
    exp_dir = Path(run_root) / experiment

    jobs: list[Job] = []
    for raw in spec["jobs"]:
        matrix = raw.get("matrix") or {}
        keys = list(matrix)
        combos = list(itertools.product(*(matrix[k] for k in keys))) if keys else [()]
        for combo in combos:
            local = dict(variables)
            local.update(dict(zip(keys, combo)))
            local["experiment"] = experiment
            name = _fill(str(raw["name"]), local)
            local["out"] = str(exp_dir / name)
            cmd_template = str(raw["cmd"]).strip()
            used = set(_PLACEHOLDER.findall(cmd_template))
            missing = sorted(k for k in used if str(local.get(k)) == REQUIRED)
            jobs.append(Job(
                name=name,
                cmd=_fill(cmd_template, local),
                gpus=int(raw.get("gpus", defaults.get("gpus", 1))),
                after=[_fill(str(a), local) for a in raw.get("after", [])],
                min_free_gb=float(raw.get("min_free_gb", defaults.get("min_free_gb", 0))),
                env={k: _fill(str(v), local) for k, v in (raw.get("env") or {}).items()},
                missing_vars=missing,
            ))

    names = [j.name for j in jobs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate job names after expansion: {dupes}")
    known = set(names)
    for j in jobs:
        unknown = [a for a in j.after if a not in known]
        if unknown:
            raise ValueError(f"job {j.name}: unknown dependencies {unknown}")
    return jobs


def nvidia_smi_probe() -> dict[int, tuple[int, int]]:
    """Memory in use / total per GPU, as reported by nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    result: dict[int, tuple[int, int]] = {}
    for line in out.strip().splitlines():
        idx, used, total = (int(x.strip()) for x in line.split(","))
        result[idx] = (used, total)
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


class Scheduler:
    """Run jobs on a fixed pool of GPUs, respecting dependencies.

    A GPU counts as free when this scheduler has not assigned it and the probe
    reports less than ``busy_mb`` in use (so GPUs someone else is using are
    left alone) and at least the job's ``min_free_gb`` available.
    """

    def __init__(
        self,
        jobs: list[Job],
        gpus: list[int],
        run_root: str | Path,
        experiment: str,
        *,
        repo_root: str | Path = ".",
        gpu_probe: GpuProbe | None = None,
        busy_mb: int = 2000,
        poll_s: float = 10.0,
        force: bool = False,
        base_env: dict[str, str] | None = None,
        log: Callable[[str], None] = print,
    ):
        self.jobs = {j.name: j for j in jobs}
        self.order = [j.name for j in jobs]
        self.gpus = list(gpus)
        self.exp_dir = Path(run_root) / experiment
        self.repo_root = Path(repo_root).resolve()
        self.probe = gpu_probe or nvidia_smi_probe
        self.busy_mb = busy_mb
        self.poll_s = poll_s
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.log = log
        self.state: dict[str, str] = {}
        self.running: dict[str, tuple[subprocess.Popen, list[int], float]] = {}
        self._stop = False

        for name in self.order:
            done = (self.exp_dir / name / "DONE").exists()
            self.state[name] = "done" if done and not force else "pending"
            if done and force:
                (self.exp_dir / name / "DONE").unlink()

    # ------------------------------------------------------------------ state
    def _write_status(self, name: str, **fields) -> None:
        d = self.exp_dir / name
        d.mkdir(parents=True, exist_ok=True)
        path = d / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(fields)
        path.write_text(json.dumps(status, indent=2))

    def _ready(self, name: str) -> bool:
        return all(self.state[a] == "done" for a in self.jobs[name].after)

    def _propagate_failures(self) -> None:
        changed = True
        while changed:
            changed = False
            for name in self.order:
                if self.state[name] != "pending":
                    continue
                if any(self.state[a] in ("failed", "blocked") for a in self.jobs[name].after):
                    self.state[name] = "blocked"
                    self._write_status(name, state="blocked", at=_now())
                    self.log(f"[queue] {name}: blocked (a dependency failed)")
                    changed = True

    def _free_gpus(self, min_free_gb: float) -> list[int]:
        taken = {g for _, gs, _ in self.running.values() for g in gs}
        mem = self.probe()
        free = []
        for g in self.gpus:
            if g in taken or g not in mem:
                continue
            used, total = mem[g]
            if used >= self.busy_mb:
                continue
            if (total - used) / 1024 < min_free_gb:
                continue
            free.append(g)
        return free

    # --------------------------------------------------------------- running
    def _launch(self, name: str, gpus: list[int]) -> None:
        job = self.jobs[name]
        d = self.exp_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "cmd.sh").write_text(job.cmd + "\n")
        env = dict(self.base_env)
        env.update(job.env)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus) if gpus else ""
        env["DPVR_JOB_NAME"] = name
        env["DPVR_JOB_OUT"] = str(d)
        log_f = open(d / "log.txt", "ab")
        log_f.write(f"\n===== {_now()} start {name} on GPUs {gpus} =====\n".encode())
        log_f.flush()
        proc = subprocess.Popen(
            ["bash", "-c", job.cmd], cwd=self.repo_root, env=env,
            stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log_f.close()
        self.running[name] = (proc, gpus, time.time())
        self.state[name] = "running"
        self._write_status(
            name, state="running", started=_now(), gpus=gpus, cmd=job.cmd,
            host=socket.gethostname(), git_sha=_git_sha(self.repo_root), pid=proc.pid,
        )
        self.log(f"[queue] {name}: started on GPUs {gpus or 'none'} (pid {proc.pid})")

    def _reap(self) -> None:
        for name in list(self.running):
            proc, gpus, t0 = self.running[name]
            rc = proc.poll()
            if rc is None:
                continue
            del self.running[name]
            dur = round(time.time() - t0, 1)
            if rc == 0:
                self.state[name] = "done"
                (self.exp_dir / name / "DONE").write_text(_now() + "\n")
            else:
                self.state[name] = "failed"
            self._write_status(name, state=self.state[name], returncode=rc,
                               finished=_now(), duration_s=dur)
            self.log(f"[queue] {name}: {self.state[name]} (rc={rc}, {dur}s)")

    def _terminate_all(self) -> None:
        for name, (proc, _, _) in list(self.running.items()):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 30
        for name, (proc, _, _) in list(self.running.items()):
            try:
                proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            self.state[name] = "interrupted"
            self._write_status(name, state="interrupted", finished=_now())
        self.running.clear()

    def request_stop(self, *_args) -> None:
        self._stop = True

    def run(self) -> dict[str, str]:
        """Run until every job is done, failed, blocked or unschedulable."""
        for name in self.order:
            job = self.jobs[name]
            if self.state[name] == "pending" and job.missing_vars:
                self.state[name] = "failed"
                self._write_status(name, state="failed", at=_now(),
                                   error=f"unset vars: {job.missing_vars}")
                self.log(f"[queue] {name}: needs --set for {job.missing_vars}")
            if job.gpus > len(self.gpus):
                self.state[name] = "failed"
                self._write_status(name, state="failed", at=_now(),
                                   error=f"needs {job.gpus} GPUs, pool has {len(self.gpus)}")
                self.log(f"[queue] {name}: needs {job.gpus} GPUs, pool has {len(self.gpus)}")
        try:
            while not self._stop:
                self._reap()
                self._propagate_failures()
                for name in self.order:
                    if self.state[name] != "pending" or not self._ready(name):
                        continue
                    need = self.jobs[name].gpus
                    if need == 0:
                        self._launch(name, [])
                        continue
                    free = self._free_gpus(self.jobs[name].min_free_gb)
                    if len(free) >= need:
                        self._launch(name, free[:need])
                if not self.running and not any(
                    s == "pending" and self._ready(n) for n, s in self.state.items()
                ):
                    break
                time.sleep(self.poll_s)
        finally:
            if self.running:
                self._terminate_all()
        return dict(self.state)
