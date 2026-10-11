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
3. 旧数据（有就拷，没有就跳过，代理会按缺失处理）：
   - 旧 checkpoint 和 `results.json` 拷到 `${CKPT_ROOT}`，保持原目录结构；至少要 7B DPVR-LF s=18 seed=1 那一个（目录里要有 `training_config.json`）；
   - 旧服务器上的评测数据目录（特别是 5874 条的 SEED 子集）拷到 `${DATA_ROOT}/benchmarks/`。
4. 启动 Claude Code：
   ```bash
   npm install -g @anthropic-ai/claude-code   # 已装可跳过
   tmux new -s claude
   cd dpvr-pr-revision && claude
   ```
   在 tmux 里开，断开 SSH 也不影响。

## 指令 1：环境与验收（E0）

```
读 AGENTS.md 和 docs/revision/SERVER_HANDOFF.md，执行阶段 1（E0）：
1. 按 §1.1 用 scripts/revision/setup_env.sh 建环境（国内镜像按 §1.1 设置），跑 check_env.py。
2. 按 §1.3 用 download_assets.py 下载模型、评测数据和 665k（图像用 download_llava_images.sh），已存在的跳过。
3. 找到旧的 7B DPVR-LF s=18 seed=1 checkpoint，跑 jobs/E0.yaml（先 --dry-run）。没有旧 checkpoint 时用 --only 跳过依赖它的两项，并在 RESULTS.md 写明。
4. 跑 pytest tests/。
5. 填 RESULTS.md 的"服务器信息"，按 AGENTS.md 的"汇报"流程汇总、push、发邮件（主题 "E0 done" 或 "BLOCKED: E0 ..."）。
最后告诉我验收清单（§1.5）每一项的结果。
```

## 指令 2：延迟（E1、E5）

```
执行 AGENTS.md 阶段 2：确认 8 张卡上没有任何其他任务后，在 tmux 里依次跑 jobs/E1.yaml 和 jobs/E5.yaml（都用 --gpus 0）。启动后告诉我各自的任务数，然后结束这一轮。
```

## 指令 3：饱和分析与 Qwen 分割层（E2、E8）

```
执行 AGENTS.md 阶段 3：jobs/E2.yaml 和 jobs/E8.yaml，--gpus 0-7（两个队列分开卡）。先只跑 E2 的 llava15_7b，按 SERVER_HANDOFF §3.1 核对它与论文 Figure 2 一致，再放开其余任务。E2 结束后根据 qwen25vl_7b 的曲线给出 Qwen2.5-VL 的分割层建议；不明确就跑 jobs/E6_split.yaml。告诉我建议值和依据。
```

## 指令 4：第二骨干与三个 seed（E6、E9）

```
执行 AGENTS.md 阶段 4：分割层用 <s>（填上一步的结论）。jobs/E6.yaml 用 --gpus 0-4 --set split=<s>，jobs/E9.yaml 用 --gpus 5-7，各开一个 tmux 窗口。E6 冒烟评测通过前不放开全量训练。启动后结束这一轮。
```

## 指令 5：基线与其余实验（E3、E4、E7、E10）

```
执行 AGENTS.md 阶段 5：在空闲的卡上跑 jobs/E3_vtw.yaml、jobs/E3_baselines.yaml、jobs/E4.yaml、jobs/E7.yaml、jobs/E10.yaml，队列之间的 --gpus 不相交，不要占用 E6/E9 还在用的卡。启动后告诉我分配方案，然后结束这一轮。
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
