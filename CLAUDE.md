# CLAUDE.md

This repo is for the Pattern Recognition major revision of the DPVR paper (manuscript PR-D-26-07438). It holds the code from `Inner-Magma/dpvr` plus the revision work. The author writes in Chinese, so reply in Chinese.

Read these first:
- `docs/revision/PLAN.md`: overall plan, code items C1–C10, timeline
- `docs/revision/REVIEWS.md`: reviewer comments, numbered R1.x / R2.x
- `docs/revision/SERVER_HANDOFF.md`: **if you are running on the GPU server, this is your operating manual**. Experiments run through `scripts/revision/run_queue.py jobs/E*.yaml`; accuracy always goes through `scripts/revision/eval_lmms.py`.
- `docs/revision/RESULTS.md`: experiment results ledger

Conventions:
- Code changes go through a PR (one per C-item, with tests). The author reviews and merges.
- The server-side session pushes directly to main only for `docs/revision/RESULTS.md` and small files under `results/revision/`.
- Do not change the evaluation setup (lmms-eval 0.5.0, `use_cache=False` for accuracy).
- `paper/` holds the LaTeX source (`ms.tex`, cas-sc single column, double spaced). It currently compiles to 33 pages; the journal limit is 35.
- Never commit credentials, checkpoints or large logs.
