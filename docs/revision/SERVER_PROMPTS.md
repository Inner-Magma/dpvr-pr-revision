# 服务器启动步骤与分阶段指令

给作者用：先做"启动前"的人工步骤，然后在服务器的 Claude Code 里按顺序粘贴下面的指令。每条指令跑完、代理回复后再贴下一条。代理的规则在根目录 [AGENTS.md](../../AGENTS.md)（`CLAUDE.md` 已自动引用它），这里的指令只说"现在做哪一段"。

## 启动前（人工，一次）

1. 代码：
   ```bash
   git clone https://github.com/Inner-Magma/dpvr-pr-revision.git && cd dpvr-pr-revision
   ```
   服务器需要能 push 到这个仓库的 main（只推结果）。建议用只授权这一个仓库、权限为 Contents: Read and write 的 fine-grained token（`gh auth login` 或 `git credential` 存起来），不要用账号全权限 token。
2. 配置：`cp .env.example .env`，填写：
   - `DATA_ROOT`、`MODEL_ROOT`、`CKPT_ROOT`：三个目录所在的盘合计至少约 1 TB 空闲；
   - `HF_ENDPOINT`：国内服务器保留 `https://hf-mirror.com`；
   - `SMTP_HOST`、`SMTP_PORT`、`SMTP_USER`、`SMTP_PASSWORD`：发件邮箱和它的 SMTP 授权码（QQ 邮箱：设置 → 账户 → 开启 SMTP 服务后生成授权码，`smtp.qq.com`、端口 465）；
   - `MAIL_TO`：收件地址。
   `.env` 已被 `.gitignore` 忽略，不会被提交。
3. 旧数据经 Hugging Face 私有仓库中转（hsf 与 H800 不能直连；见下一节"在 hsf 上上传"）。传了的部分由代理下载到 `${CKPT_ROOT}/hsf/`，没传的代理按缺失处理，大修照常进行。评测数据不用拷，全部重新下载（hsf 上的 SEED 副本不完整）。
4. 在 `.env` 里再加两行：`HF_TOKEN=<有该私有仓库读权限的 token>`、`HF_CKPT_REPO=<在 hsf 上建的私有数据集仓库，形如 用户名/仓库名>`。
5. 启动 Claude Code：
   ```bash
   npm install -g @anthropic-ai/claude-code   # 已装可跳过
   tmux new -s claude
   cd dpvr-pr-revision && claude
   ```
   在 tmux 里开，断开 SSH 也不影响。

## 在 hsf 上上传（人工，hsf 掉线前尽快）

只传顶层权重（不传 `checkpoint-*` 中间存档），按优先级从上往下，掉线前传到哪里算哪里。合计约 80 GB（按各目录顶层大小估计）。

| 优先级 | hsf 路径（`/nvme/lsy/` 下） | 用途 |
| --- | --- | --- |
| 1 | `Dual-Path-Token-Routing/results/eval_lmms`、`Dual-Path-Token-Routing/reports`（打成一个 tar，约 1 GB） | 旧逐题日志，核对论文每行的 SEED 用的哪套题 |
| 2 | `checkpoints/token_diversion/llava665k_split18_d1` | 训练起点核查（E0）；Table 2 DPVR-PC 行用完整 SEED 重测（E12） |
| 3 | `checkpoints/hsf_lf_13b/llava665k_s24_d1` | 13B DPVR-LF 在训练好的权重上做 KV cache 一致性验证（E12；论文的 s=28 已丢，这是留下的 13B LF） |
| 4 | `checkpoints/token_diversion/llava665k_13b_split28_d1` | 13B 主行重测（E12） |
| 5 | `checkpoints/full_ft` | Table 2 的 FullFT 改成实测行，回应"与全量微调比较"的质疑（E12） |
| 6 | `checkpoints/lora` | Table 2 LoRA 行重测（E12） |

不传：`fastv`（论文 FastV 行是免训练的，E3 会重测）、`token_diversion_13b/*` 和 `token_diversion/ablate_split_*`（分割层扫描表保留旧数字，标注 SEED 题集）。

```bash
pip install -U "huggingface_hub[cli]" && huggingface-cli login     # 用有写权限的 token
cd /nvme/lsy && tar czf dpvr_old_eval.tgz Dual-Path-Token-Routing/results/eval_lmms Dual-Path-Token-Routing/reports
huggingface-cli repo create <仓库名> --repo-type dataset --private -y
huggingface-cli upload <用户名>/<仓库名> dpvr_old_eval.tgz dpvr_old_eval.tgz --repo-type dataset
cd /nvme/lsy/checkpoints
for d in token_diversion/llava665k_split18_d1 hsf_lf_13b/llava665k_s24_d1 \
         token_diversion/llava665k_13b_split28_d1 full_ft lora; do
  huggingface-cli upload <用户名>/<仓库名> "$d" "$d" --repo-type dataset --exclude "checkpoint-*/*" "*/checkpoint-*/*"
done
```

中断后重跑同一条命令即可（已上传的文件会跳过）。`full_ft`、`lora` 下多一层运行目录；如果某个运行的最终权重只在最大的 `checkpoint-N` 里，就单独传那一个 `checkpoint-N`，H800 侧会自动找到。

## 指令 1：环境与验收（E0）

```
读 AGENTS.md 和 docs/revision/SERVER_HANDOFF.md，执行阶段 1（E0）：
1. 按 §1.1 用 scripts/revision/setup_env.sh 建环境（国内镜像按 §1.1 设置），跑 check_env.py。
2. 按 §1.3 用 download_assets.py 下载模型、评测数据和 665k（图像用 download_llava_images.sh），已存在的跳过。
3. 按 §1.3 从 .env 的 HF_CKPT_REPO 下载 hsf 中转的文件到 ${CKPT_ROOT}/hsf/（没有就跳过）。跑 jobs/E0.yaml（先 --dry-run），prov_ckpt=$(python scripts/revision/find_ckpt.py ${CKPT_ROOT}/hsf/token_diversion/llava665k_split18_d1)；没有这个目录就不传，起点核查那一项会拒绝启动，其余照跑，在 RESULTS.md 写明。
4. 跑 pytest tests/。
5. 填 RESULTS.md 的"服务器信息"，按 AGENTS.md 的"汇报"流程汇总、push、发邮件（主题 "E0 done" 或 "BLOCKED: E0 ..."）。
最后告诉我验收清单（§1.5）每一项的结果。
```

## 指令 2：延迟（E1、E5）

```
执行 AGENTS.md 阶段 2：确认 8 张卡上没有任何其他任务后，在 tmux 里依次跑 jobs/E1.yaml 和 jobs/E5.yaml（都用 --gpus 0）。启动后告诉我各自的任务数，然后结束这一轮。
```

## 指令 3：饱和分析与 Qwen 分割层（E2）

```
执行 AGENTS.md 阶段 3：jobs/E12.yaml 用 --gpus 0-2（checkpoint 路径由脚本自动找），jobs/E2.yaml 用 --gpus 3-7。E12 里 lf13b_s24 有/无 KV cache 两次的分数不一致要报告。先只跑 E2 的 llava15_7b，按 SERVER_HANDOFF §3.1 核对它与论文 Figure 2 一致，再放开其余任务。E2 结束后根据 qwen25vl_7b 的曲线给出 Qwen2.5-VL 的分割层建议；不明确就跑 jobs/E6_split.yaml。告诉我建议值和依据。
```

## 指令 4：第二骨干与三个 seed（E6、E9）

```
执行 AGENTS.md 阶段 4：分割层用 <s>（填上一步的结论）。jobs/E6.yaml 用 --gpus 0-4 --set split=<s>，jobs/E9.yaml 用 --gpus 5-7，各开一个 tmux 窗口。E6 冒烟评测通过前不放开全量训练。启动后结束这一轮。
```

## 指令 5：基线与其余实验（E3、E4、E8、E7、E10）

```
执行 AGENTS.md 阶段 5：在空闲的卡上按优先级跑 jobs/E3_vtw.yaml、jobs/E4.yaml，然后 jobs/E3_baselines.yaml、jobs/E8.yaml、jobs/E7.yaml、jobs/E10.yaml，队列之间的 --gpus 不相交，不要占用 E6/E9 还在用的卡。估一下各队列能否在 10/21 前结束，赶不上的按 AGENTS.md 的放弃顺序不启动。启动后告诉我分配方案和预计完成时间，然后结束这一轮。
```

## 随时可用：检查进度

```
检查所有 revision 队列的进度（${CKPT_ROOT}/revision_runs/*/*/status.json 和日志）。有实验全部结束的，按 AGENTS.md 汇总、push、发邮件；有失败或异常的，按红线处理。告诉我每个实验的状态和下一步。
```

想让它定时自查，可以用 `/loop 1h <上面这段>`（Claude Code 的循环命令；不可用时就手动隔几小时贴一次）。

## 指令 6：收尾

```
所有实验都结束了。跑一次 collect_results.py，确认 RESULTS.md 每个实验都有记录、"问题"一节写全，push 后发一封主题为 "ALL DONE" 的总结邮件，正文开头用几句话概括哪些结果支持论文结论、哪些与论文不一致。
```
