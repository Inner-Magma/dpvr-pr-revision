# AGENTS.md：GPU 服务器上的执行代理入口

你是 GPU 服务器上的代码代理（Claude Code 或其他读 AGENTS.md 的代理），负责 DPVR 大修（Pattern Recognition 稿件 PR-D-26-07438）的环境搭建、实验运行、问题修复和结果汇报。作者写中文，回复和记录都用中文。

详细操作手册是 [docs/revision/SERVER_HANDOFF.md](docs/revision/SERVER_HANDOFF.md)，本文件只是入口和红线。冲突时以本文件为准，其余以 SERVER_HANDOFF 为准。

## 授权（2026-10-17 作者决定）

作者原话："可以给服务器上的 Claude Code 独立自主决策的权限。最好让它连续不断的工作，这样才是最快的。""怎么合适怎么来，不用刻意追求之前的结果，因为我之前的结果没保存，但是绝对是跑出来的真实数据。""别担心时间问题，我可以要更多的卡。"

所以从现在起：

- **你自己做决定，不等作者。** 排期、分卡、阶段推进、重跑、失败后的排查和修复（代码修复走 `server-fixes` 分支，见下文）、实验之间怎么取舍，都由你判断。决定和理由写进 `docs/revision/RESULTS.md`，然后继续。只有下文"必须停下来问作者"的几种情况才停。
- **连续工作。** 一个队列启动后不要结束工作等人来催：用 Claude Code 的 `/loop`（或同类自检机制）定时检查进度。某个实验一结束就汇总、push，然后立刻在空出来的卡上启动下一个。所有实验结束前不进入空闲。
- **基线以本机为准。** 论文的旧数字已无原始结果可查，不再作为验收或复现的目标。E0 `reeval_vanilla7b`（commit `bcfe97a` 记录的八项）就是新的 vanilla LLaVA-1.5-7B 基线，E0 视为通过。大修所有表格的数字都用本服务器、本仓库的评测链路重新测。和论文的差异只记录，不作为停下的理由。
- **卡可以加。** 作者可以申请更多 GPU。看到新的空闲卡（`nvidia-smi`）就用上，不必等阶段表里写的卡号。

## 红线（任何情况都不做）

1. **评测口径不变：** 精度只用 `scripts/revision/eval_lmms.py`（lmms-eval 0.5.0，`dpvr_hf` / `dpvr_qwen25vl`，`use_cache=False`），不用 `scripts/evaluate.py`，不从论文抄数字。同一张表里的所有行必须来自同一条链路。
2. **不改超参数、不改有效 batch。** 训练单卡有效 batch 4（E6 全量训练是 4 卡×1，见 `jobs/E6.yaml`）。显存有富余也不加 batch。学习率、epoch、分割层候选等实验设定也不改；E6 的分割层按 E2 结论选，属于你的决定。
3. **不提交任何凭据或个人信息。** `.env`（含 SMTP 授权码、收件地址、HF token）、`results/revision/summary/mail.eml`、checkpoint、完整日志都不进仓库。每次 `git add` 前跑一遍 `git grep -nE 'hf_[A-Za-z0-9]{20,}|ghp_|github_pat_'` 并看 `git diff --cached`。commit 作者用 `git -c user.name="Inner-Magma" -c user.email="lsy2002@stu.pku.edu.cn" commit ...`，不要用服务器上别的身份。
4. **不删旧结果和 checkpoint，** 不占用别人正在用的 GPU。
5. **main 上只直接 push** `docs/revision/RESULTS.md` 和 `results/revision/**` 下的小文件，代码改动走下文的 `server-fixes` 分支。不改写任何已推送的历史（不 force push），不碰 `paper/`（论文由云端和作者改）。

## 代码修复

可以自己修 `src/`、`scripts/`、`configs/`、`jobs/`、`tests/` 里的 bug，不用停下来等。修复走分支，main 上的代码仍由作者审核合入：

- 在分支 `server-fixes` 上修（没有就从 main 建，之后的修复都继续加在这个分支上）。只修让实验跑不起来或结果不对的问题，改动尽量小；不借机重构，不改红线 1、2 涉及的口径和设定。
- 改完跑 `pytest tests/`（至少跑相关的测试文件），通过后提交，commit 信息以 `fix(<范围>):` 开头，写清现象和原因，`git push -u origin server-fixes`。第一次推送时开一个 PR（`server-fixes` → `main`），之后的修复都进这个 PR，作者有空再合。
- **实验直接用 `server-fixes` 分支的代码接着跑，不用等合入。** run meta 会记下实际用的 commit。为了让结果和记录分开，可以用 `git worktree add ../dpvr-fixes server-fixes` 在另一个目录检出修复分支跑实验，主目录留在 main 上提交 RESULTS.md。
- 作者合入后，`server-fixes` 跟 main 同步（`git pull origin main`），继续用。
- 在 RESULTS.md 的"问题"一节记一笔：现象、原因、修复 commit。
- 已知待修（RESULTS.md 问题 §8、§10.1、§10.2）：
  - `src/dpvr/lmms_plugin/` 缺 `tasks/__init__.py`。补上后可以去掉服务器侧的 shim。
  - `collect_results.py` 拷出的 `log_tail.txt` 和其他文本文件没有脱敏 token。
  - 超过 `--max_kb` 的 `log.txt` 不覆盖，旧的失败日志会留在 `results/` 里。
- 已经跑完的结果不因代码修复而作废，除非这个 bug 影响了它的数字；受影响的就重跑，并在 RESULTS.md 写明。

## 异常处理

loss NaN、显存或耗时远超估计、同一个任务连续失败时：先停这一项，自己排查。能修就按上一节修了重跑；修不了就记进 RESULTS.md "问题"，跳过它，继续其他不受影响的实验。精度判断以本机的新基线和新训练结果为准：比如 DPVR 比本机 vanilla 低很多，说明有问题，要查；和论文差多少不算异常。

## 必须停下来问作者

只有这几种情况才停下来，发一封主题以 `BLOCKED:` 开头的邮件，并在 RESULTS.md 写明需要作者做什么：

- 需要作者本人才能做的事：申请或续用 GPU、凭据（HF / GitHub / SMTP）失效、磁盘满了而且清不出空间（只能删你自己产生的临时文件）。
- 要做的事违反上面的红线。
- 结果显示论文的核心结论站不住（例如 DPVR-LF 在多数基准上明显差于本机 vanilla），需要作者决定论文怎么写。这时实验照常继续跑，只是把这件事报告出来。

## 工作方式

- 所有实验都通过任务队列：`python scripts/revision/run_queue.py jobs/E*.yaml --gpus <卡> [--set k=v]`。先 `--dry-run`。队列中断后原命令重跑即可，完成的任务自动跳过。
- 长任务放在 tmux 里，每个队列一个窗口，例如 `tmux new -d -s E6 'python scripts/revision/run_queue.py jobs/E6.yaml --gpus 0-4 2>&1 | tee -a $CKPT_ROOT/revision_runs/E6.queue.log'`。
- 进度看 `${CKPT_ROOT}/revision_runs/<实验>/<任务>/status.json` 和 `log.txt`。等待用 `/loop` 定时检查（间隔按任务长短定，例如 30 分钟），不要在一轮里 sleep 几个小时。
- 同时跑多个队列时给它们**不相交**的 `--gpus`。
- 每个结果带 run meta（脚本已自动写入 git commit、GPU 型号、版本）。

## 推荐顺序

这是默认顺序，你可以按实际情况调整。原则是让卡一直有活干，同时保证延迟测量不受干扰。

| 顺序 | 做什么 | 卡 |
| --- | --- | --- |
| 1 | E1 + E5：重测整张延迟表。测的时候整机不跑别的任务，测完马上进入下一步 | 1 张，其他卡空着 |
| 2 | E12（hsf 中转来的 checkpoint 重测，传了哪些跑哪些）、E2（饱和分析）、E9（7B 三个 seed）同时开 | 分开 |
| 3 | E2 出结果后，据 `qwen25vl_7b` 曲线定 Qwen 分割层（不明确就跑 `jobs/E6_split.yaml`），开 E6；E6 冒烟评测没过不开全量训练 | 4 张起 |
| 4 | 空出来的卡按优先级跑：E3_vtw、E4 → E3_baselines → E8 → E7 → E10 | 空出来的卡 |

有多台机器或多出来的卡时，延迟测量（E1/E5）放在一台单独的机器上，其余实验直接并行开。

**时间：** 修改稿 2026-10-25 截止。作者说时间不用担心、可以加卡，所以先不放弃任何实验。到 2026-10-23 还没跑完的，按 E10 → E7 → E8 → E3_baselines → E6 全量训练的顺序停掉，在 RESULTS.md 写明，用已完成的部分汇总。

## 汇报

每个实验（一个 `jobs/E*.yaml`）全部任务结束后：

```bash
python scripts/revision/collect_results.py                     # 汇总到 results/revision/
# 在 docs/revision/RESULTS.md 追加一行：日期 | 实验 | 配置 | 关键数字 | 结果路径 | commit
git add docs/revision/RESULTS.md results/revision
git -c user.name="Inner-Magma" -c user.email="lsy2002@stu.pku.edu.cn" commit -m "results(E9): 7B DPVR-LF s=18 seeds 1-3"
git push origin main
python scripts/revision/send_results.py --subject "E9 done"    # 发到 .env 里的 MAIL_TO
```

除了每个实验结束，开始一个新实验、做了重要决定、修了 bug 时也 push 一次 RESULTS.md，云端靠它跟进度。

要发邮件的时候：每个实验结束、`BLOCKED:`、全部结束时一封总结。邮件发送失败时不要换别的途径，记进 RESULTS.md "问题"后继续。

RESULTS.md 的"服务器信息"一节随硬件变化更新：GPU 型号×数量、驱动/CUDA、`pip freeze` 路径。
