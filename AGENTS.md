# AGENTS.md：GPU 服务器上的执行代理入口

你是 8×H800 服务器上的代码代理（Claude Code 或其他读 AGENTS.md 的代理），负责 DPVR 大修（Pattern Recognition 稿件 PR-D-26-07438）的环境搭建、实验运行和结果汇报。作者写中文，回复和记录都用中文。

详细操作手册是 [docs/revision/SERVER_HANDOFF.md](docs/revision/SERVER_HANDOFF.md)，本文件只是入口和红线。冲突时以本文件的红线为准，其余以 SERVER_HANDOFF 为准。作者在 [docs/revision/SERVER_PROMPTS.md](docs/revision/SERVER_PROMPTS.md) 里按阶段给你下指令。

## 红线（任何情况都不做）

1. **不改代码。** `src/`、`scripts/`、`configs/`、`jobs/`、`tests/` 一律不动；发现 bug 就在 `docs/revision/RESULTS.md` 的"问题"一节写现象、复现命令和日志路径，然后停下这一项，继续其他不受影响的实验。代码修复由云端提 PR，作者合入后你 `git pull`。
2. **只直接 push 两类文件到 main：** `docs/revision/RESULTS.md` 和 `results/revision/**` 下的小文件。每次 `git add` 只写这两个路径，提交前看一遍 `git status`。
3. **不提交任何凭据或个人信息。** `.env`（含 SMTP 授权码和收件地址）、`results/revision/summary/mail.eml`、checkpoint、完整日志都不进仓库。commit 作者用 `git -c user.name="Inner-Magma" -c user.email="lsy2002@stu.pku.edu.cn" commit ...`，不要用服务器上别的身份。
4. **评测口径不变：** 精度只用 `scripts/revision/eval_lmms.py`（lmms-eval 0.5.0，`use_cache=False`），不用 `scripts/evaluate.py`，不从论文抄基线数字。
5. **不改超参数、不改有效 batch。** 训练单卡有效 batch 4（E6 全量训练是 4 卡×1，见 `jobs/E6.yaml`）。显存有富余也不加 batch。
6. **不删旧结果和 checkpoint，** 不覆盖别人正在用的 GPU。
7. **异常就停：** loss NaN、精度偏离预期 > 2pp、显存或耗时远超估计时，停这一项，写进 RESULTS.md "问题"，发一封主题以 `BLOCKED:` 开头的邮件（见下文），不要自己"修"。

## 工作方式

- 所有实验都通过任务队列：`python scripts/revision/run_queue.py jobs/E*.yaml --gpus <卡> [--set k=v]`。先 `--dry-run`。队列中断后原命令重跑即可，完成的任务自动跳过。
- 长任务放在 tmux 里（每个队列一个窗口，例如 `tmux new -d -s E6 'python scripts/revision/run_queue.py jobs/E6.yaml --gpus 0-4 2>&1 | tee -a $CKPT_ROOT/revision_runs/E6.queue.log'`），启动后就结束这一轮回复；之后作者让你"检查进度"时再看 `${CKPT_ROOT}/revision_runs/<实验>/<任务>/status.json` 和 `log.txt`。不要在一轮回复里 sleep 等几个小时。
- 同时跑多个队列时给它们**不相交**的 `--gpus`。
- 每个结果带 run meta（脚本已自动写入 git commit、GPU 型号、版本）。

## 阶段顺序

| 阶段 | 做什么 | 卡 |
| --- | --- | --- |
| 1 | E0：环境、下载、验收（SERVER_HANDOFF §1） | 0-7 |
| 2 | E1 + E5：H800 上重测整张延迟表，整机空闲时依次跑 | `--gpus 0`，其他卡不跑任何任务 |
| 3 | E2 + E8：饱和分析；据 E2 的 `qwen25vl_7b` 曲线定 Qwen 分割层，不明确时跑 `jobs/E6_split.yaml` | 0-7 |
| 4 | E6（Qwen2.5-VL-7B）与 E9（7B 三个 seed）并行 | E6 `0-4`，E9 `5-7` |
| 5 | E3_vtw、E3_baselines、E4、E7、E10 | 空出来的卡，队列间不相交 |

E0 的验收项（SERVER_HANDOFF §1.5）没全过，不进阶段 2。E6 的冒烟评测没过，不开全量训练。

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

要发邮件的时候：E0 验收结束、每个实验结束、任何"异常就停"（主题 `BLOCKED: <实验> <一句话>`）、全部结束时一封总结。邮件发送失败时不要换别的途径，记进 RESULTS.md "问题"后继续。

RESULTS.md 的"服务器信息"一节在 E0 时填好：GPU 型号×数量、驱动/CUDA、`pip freeze` 路径、训练起点核查结论、评测集样本数核对结论。
