"""collect_results.py, send_results.py and download_assets.py (server hand-off scripts)."""
from __future__ import annotations

import email
import importlib.util
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / "revision" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collect_results = _load("collect_results")
send_results = _load("send_results")
download_assets = _load("download_assets")


def _job(root: Path, exp: str, job: str, state: str, summary: dict | None = None, extra: dict | None = None):
    d = root / exp / job
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "started": "t0", "finished": "t1", "returncode": 0}))
    (d / "log.txt").write_text("\n".join(f"line {i}" for i in range(500)))
    (d / "cmd.sh").write_text("python secret_cmd\n")
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary))
    for rel, data in (extra or {}).items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return d


@pytest.fixture
def run_root(tmp_path):
    root = tmp_path / "revision_runs"
    metrics = {
        "pope_local": {"pope_accuracy,none": 0.8567},
        "mme_local": {"mme_perception_score,none": 1500.123, "mme_cognition_score,none": 300.0},
        "scienceqa_local": {"mc_accuracy,none": 0.6465},
    }
    _job(root, "E0", "reeval_lf7b_seed1", "done",
         {"meta": {"checkpoint": "/ckpt/x", "gpu": "NVIDIA H800", "git_sha": "abc", "limit": None},
          "metrics": metrics},
         {"lmms/samples.jsonl": b"x" * 10, "checkpoint-50/model.json": b"{}",
          "big.json": b"0" * (600 * 1024), "latency.json": b"{\"ms\": 1}"})
    _job(root, "E6", "train", "failed")
    _job(root, "E6", "eval", "blocked")
    return root


def test_collect_tables_and_copies(run_root, tmp_path):
    out = tmp_path / "results"
    stats = collect_results.collect(run_root, out, max_kb=512)
    assert stats["evals"] == 1 and stats["jobs"] == 3
    assert stats["states"] == {"done": 1, "failed": 1, "blocked": 1}

    rows = (out / "summary" / "eval_table.csv").read_text().splitlines()
    header, row = rows[0].split(","), rows[1].split(",")
    rec = dict(zip(header, row))
    assert rec["POPE"] == "85.67" and rec["SQA"] == "64.65"
    assert rec["MME-P"] == "1500.12" and rec["MME-C"] == "300.0"
    assert rec["SEED"] == "" and rec["gpu"] == "NVIDIA H800"
    assert "| E6 | train | failed |" in (out / "summary" / "jobs.md").read_text()

    job = out / "E0" / "reeval_lf7b_seed1"
    assert (job / "summary.json").exists() and (job / "latency.json").exists()
    assert not (job / "big.json").exists()            # over max_kb
    assert not (job / "lmms").exists()                # sample logs stay on the server
    assert not (job / "checkpoint-50").exists()
    assert not (job / "cmd.sh").exists()
    tail = (job / "log_tail.txt").read_text().splitlines()
    assert len(tail) == 200 and tail[-1] == "line 499"


def test_collect_is_rerunnable(run_root, tmp_path):
    out = tmp_path / "results"
    collect_results.collect(run_root, out)
    assert collect_results.collect(run_root, out)["jobs"] == 3


def test_collect_redacts_secrets(tmp_path):
    root = tmp_path / "revision_runs"
    token = "hf_" + "A" * 30
    _job(root, "E0", "r", "done",
         extra={"note.txt": f"using {token}\n".encode(),
                "log.txt": f"start\n{token}\n".encode()})
    out = tmp_path / "results"
    collect_results.collect(root, out)
    dst = out / "E0" / "r"
    assert token not in (dst / "note.txt").read_text()
    assert "hf_**REDACTED**" in (dst / "note.txt").read_text()
    assert token not in (dst / "log_tail.txt").read_text()
    assert "hf_**REDACTED**" in (dst / "log_tail.txt").read_text()


def test_collect_drops_stale_log_copy(tmp_path):
    """源 log.txt 超过 max_kb 时不会覆盖，旧副本必须被删掉而不是留着误导人。"""
    root = tmp_path / "revision_runs"
    job = _job(root, "E0", "r", "done")
    out = tmp_path / "results"
    collect_results.collect(root, out)
    assert (out / "E0" / "r" / "log.txt").exists()

    (job / "log.txt").write_text("y" * 2000)
    collect_results.collect(root, out, max_kb=1)
    assert not (out / "E0" / "r" / "log.txt").exists()   # 旧日志已删
    assert (out / "E0" / "r" / "log_tail.txt").exists()  # 新日志的尾巴还在


@pytest.fixture
def results_tree(tmp_path, monkeypatch, run_root):
    res = tmp_path / "repo" / "results" / "revision"
    collect_results.collect(run_root, res)
    (res / "E0" / "huge.bin").write_bytes(b"1" * (2 * 1024 * 1024))
    ledger = tmp_path / "repo" / "RESULTS.md"
    ledger.write_text("# ledger\nE0 done\n")
    monkeypatch.setattr(send_results, "RESULTS", res)
    monkeypatch.setattr(send_results, "LEDGER", ledger)
    for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO", "MAIL_FROM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(send_results, "_load_env", lambda: None)
    return res


def _parts(msg):
    body = msg.get_body(("plain",)).get_content()
    atts = {p.get_filename(): p.get_payload(decode=True) for p in msg.iter_attachments()}
    return body, atts


def test_message_body_and_zip_cap(results_tree):
    msg = send_results.build_message("a@x", ["b@y"], "subj", max_zip_mb=1.0)
    body, atts = _parts(msg)
    assert "Accuracy runs" in body and "Job states" in body and "E0 done" in body
    assert "1 files left out of the zip" in body
    assert set(atts) == {"eval_table.csv", "jobs.csv", "results_revision.zip"}
    names = zipfile.ZipFile(BytesIO(atts["results_revision.zip"])).namelist()
    assert "revision/summary/eval_table.md" in names
    assert not any(n.endswith("huge.bin") for n in names)


def test_dry_run_writes_eml_and_sends_nothing(results_tree, monkeypatch):
    monkeypatch.setattr(send_results.smtplib, "SMTP_SSL", lambda *a, **k: pytest.fail("sent on dry run"))
    monkeypatch.setattr(sys, "argv", ["send_results.py", "--dry-run"])
    send_results.main()
    eml = results_tree / "summary" / "mail.eml"
    msg = email.message_from_bytes(eml.read_bytes())
    assert msg["Subject"].startswith("[DPVR revision]")


def test_missing_credentials_exit(results_tree, monkeypatch):
    monkeypatch.setenv("MAIL_TO", "b@y")
    monkeypatch.setattr(sys, "argv", ["send_results.py"])
    with pytest.raises(SystemExit, match="SMTP_HOST"):
        send_results.main()


class _FakeSMTP:
    calls: list = []

    def __init__(self, host, port, **kw):
        self.calls.append(("connect", host, port, type(self).__name__))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        self.calls.append(("starttls",))

    def login(self, user, pw):
        self.calls.append(("login", user, pw))

    def send_message(self, msg):
        self.calls.append(("send", msg["To"], msg["Subject"]))


@pytest.mark.parametrize("port,expect_tls", [("465", False), ("587", True)])
def test_send_ssl_and_starttls(results_tree, monkeypatch, port, expect_tls):
    _FakeSMTP.calls = []
    monkeypatch.setattr(send_results.smtplib, "SMTP_SSL", _FakeSMTP)
    monkeypatch.setattr(send_results.smtplib, "SMTP", _FakeSMTP)
    for k, v in {"SMTP_HOST": "smtp.example.com", "SMTP_PORT": port, "SMTP_USER": "me@example.com",
                 "SMTP_PASSWORD": "code", "MAIL_TO": "you@example.com, him@example.com"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(sys, "argv", ["send_results.py", "--subject", "E0 done"])
    send_results.main()
    calls = _FakeSMTP.calls
    assert calls[0][1:3] == ("smtp.example.com", int(port))
    assert (("starttls",) in calls) == expect_tls
    assert ("login", "me@example.com", "code") in calls
    assert calls[-1] == ("send", "you@example.com, him@example.com", "[DPVR] E0 done")


def test_combine_blink(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    raw = tmp_path / "raw"
    for sub, n in (("Art_Style", 3), ("Counting", 2)):
        (raw / sub).mkdir(parents=True)
        pd.DataFrame({"idx": range(n)}).to_parquet(raw / sub / "val-00000-of-00001.parquet")
        pd.DataFrame({"idx": range(9)}).to_parquet(raw / sub / "test-00000-of-00001.parquet")
    (raw / ".cache").mkdir()
    out = tmp_path / "blink_val_combined.parquet"
    assert download_assets.combine_blink(raw, out) == 5
    df = pd.read_parquet(out)
    assert sorted(set(df["sub_task"])) == ["Art_Style", "Counting"]


def test_download_list_and_dry_run(tmp_path):
    env = {"DATA_ROOT": str(tmp_path / "d"), "MODEL_ROOT": str(tmp_path / "m"),
           "CKPT_ROOT": str(tmp_path / "c"), "PATH": "/usr/bin:/bin", "HF_ENDPOINT": ""}
    script = str(REPO_ROOT / "scripts" / "revision" / "download_assets.py")
    out = subprocess.run([sys.executable, script, "--list"], capture_output=True, text=True, check=True).stdout
    assert "Qwen2.5-VL-7B-Instruct" in out and "BLINK-Benchmark/BLINK" in out
    out = subprocess.run([sys.executable, script, "models", "benchmarks", "llava665k", "--dry-run"],
                         capture_output=True, text=True, check=True, env=env).stdout
    assert f"{tmp_path / 'm'}/reference/vicuna-7b-v1.5" in out
    assert f"{tmp_path / 'd'}/benchmarks/blink/raw" in out
    assert "llava_v1_5_mix665k.json" in out
    assert not (tmp_path / "m").exists()


def test_expected_counts_cover_benchmarks():
    assert set(download_assets.EXPECTED) == set(download_assets.BENCHMARKS)
