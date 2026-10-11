#!/usr/bin/env python
"""Email the collected results (run collect_results.py first).

Body: the accuracy table, the job-state table and the results ledger
(docs/revision/RESULTS.md). Attachments: the two CSV tables and a zip of
``results/revision/`` (small files only; capped by ``--max_zip_mb``).

SMTP settings come from the environment or the server's ``.env`` (never
commit them):

    SMTP_HOST      e.g. smtp.qq.com / smtp.gmail.com / smtp.163.com
    SMTP_PORT      465 (SSL, default) or 587 (STARTTLS)
    SMTP_USER      sender login, e.g. the full mailbox address
    SMTP_PASSWORD  the mailbox's SMTP authorisation code / app password
    MAIL_TO        recipient(s), comma separated
    MAIL_FROM      optional, defaults to SMTP_USER

    python scripts/revision/send_results.py --dry-run     # writes results/revision/summary/mail.eml
    python scripts/revision/send_results.py --subject "E0 done"
"""
from __future__ import annotations

import argparse
import io
import os
import smtplib
import ssl
import sys
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "revision"
LEDGER = REPO_ROOT / "docs" / "revision" / "RESULTS.md"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def zip_results(root: Path, max_mb: float) -> tuple[bytes, list[str]]:
    """Zip root, smallest files first, stopping before max_mb; returns skipped paths."""
    buf = io.BytesIO()
    skipped: list[str] = []
    budget = max_mb * 1024 * 1024
    files = sorted((f for f in root.rglob("*") if f.is_file() and f.suffix != ".eml"),
                   key=lambda f: f.stat().st_size)
    used = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            size = f.stat().st_size
            if used + size > budget:
                skipped.append(str(f.relative_to(root)))
                continue
            z.write(f, f.relative_to(root.parent))
            used += size
    return buf.getvalue(), skipped


def build_message(sender: str, to: list[str], subject: str, max_zip_mb: float) -> EmailMessage:
    summary = RESULTS / "summary"
    parts = []
    for name in ("eval_table.md", "jobs.md"):
        p = summary / name
        parts.append(p.read_text(encoding="utf-8") if p.exists()
                     else f"({name} missing: run scripts/revision/collect_results.py)\n")
    if LEDGER.exists():
        parts.append(LEDGER.read_text(encoding="utf-8"))
    zipped = None
    if RESULTS.exists():
        zipped, skipped = zip_results(RESULTS, max_zip_mb)
        if skipped:
            parts.append(f"[{len(skipped)} files left out of the zip (size cap)]\n")
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject
    msg.set_content("\n\n".join(parts))
    for name in ("eval_table.csv", "jobs.csv"):
        p = summary / name
        if p.exists():
            msg.add_attachment(p.read_bytes(), maintype="text", subtype="csv", filename=name)
    if zipped is not None:
        msg.add_attachment(zipped, maintype="application", subtype="zip", filename="results_revision.zip")
    return msg


def send(msg: EmailMessage, host: str, port: int, user: str, password: str) -> None:
    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=60) as s:
            s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=60) as s:
            s.starttls(context=ctx)
            s.login(user, password)
            s.send_message(msg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default=None)
    ap.add_argument("--max_zip_mb", type=float, default=15.0)
    ap.add_argument("--dry-run", action="store_true", help="write the message to summary/mail.eml, send nothing")
    args = ap.parse_args()
    _load_env()

    to = [a.strip() for a in os.environ.get("MAIL_TO", "").split(",") if a.strip()]
    user = os.environ.get("SMTP_USER", "")
    sender = os.environ.get("MAIL_FROM") or user or "dpvr-server@localhost"
    subject = args.subject or f"[DPVR revision] results {datetime.now():%Y-%m-%d %H:%M}"
    msg = build_message(sender, to or ["(MAIL_TO not set)"], f"[DPVR] {subject}" if args.subject else subject,
                        args.max_zip_mb)

    if args.dry_run:
        out = RESULTS / "summary" / "mail.eml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(bytes(msg))
        print(f"dry run: {out} ({out.stat().st_size / 1024:.0f} KB), to={to}")
        return
    missing = [k for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "MAIL_TO") if not os.environ.get(k)]
    if missing:
        sys.exit(f"missing in .env / environment: {', '.join(missing)}")
    send(msg, os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "465")), user,
         os.environ["SMTP_PASSWORD"])
    print(f"sent to {', '.join(to)}: {msg['Subject']}")


if __name__ == "__main__":
    main()
