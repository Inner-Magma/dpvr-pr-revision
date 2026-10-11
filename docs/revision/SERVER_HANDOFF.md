# 服务器端执行手册（给服务器上的 Claude Code）

你在 GPU 服务器上负责 DPVR 大修的环境搭建与实验运行。代码改动由云端 Claude 以 PR 形式提交，作者合入 main 后你 `git pull` 使用。动手前先读 [PLAN.md](PLAN.md) 和 [REVIEWS.md](REVIEWS.md)。

## 0. 基本规则

- **不要改模型代码或训练代码。** 发现 bug 时在 RESULTS.md 的"问题"一节写清现象和复现命令，然后停下来等。你只直接提交 `docs/revision/RESULTS.md` 和 `results/revision/**` 下的小文件（json / md / csv / 日志摘要），大文件（checkpoint、完整日志）留在服务器上。
- **不覆盖旧结果，不删除旧 checkpoint。** 新结果写到 `results/revision/<实验编号>/<配置名>/`。
- **评测口径不变**：精度一律用 lmms-eval 0.5.0 的现有 handler，`use_cache=False`（与论文已有数字对齐）；新基线必须在同一套评测代码下重测，不从原论文抄数。
- **异常就停**：出现 loss NaN、精度偏离预期 > 2pp、显存或耗时远超估计时，先停止，在 RESULTS.md 记录并报告，不要自行改超参数。
- 长任务用 tmux 或 nohup，日志写进结果目录；不占用别人正在用的 GPU；`CUDA_VISIBLE_DEVICES` 每次显式指定。
- 每条结果都带 run meta：git commit、config、seed、GPU 型号、软件版本（`src/dpvr/utils/run_meta.py`）。

## 1. 环境（E0）

服务器是 8×H20（96 GB）或 8×A800（80 GB），两种卡用同一套环境：CUDA 12.8 的 torch 轮子同时支持 sm_90 与 sm_80，全程 bf16。

### 1.1 安装（版本与论文附录 B 一致，务必锁死）

```bash
git clone https://github.com/Inner-Magma/dpvr-pr-revision.git && cd dpvr-pr-revision
bash scripts/revision/setup_env.sh          # 建 conda 环境 dpvr，按 requirements-lock.txt 安装
conda activate dpvr
```

- 国内镜像：`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`；torch 用 `TORCH_INDEX=<镜像>`（阿里云镜像另加 `TORCH_FIND_LINKS=1`）。
- `requirements-lock.txt` 同时作为约束文件传给 lmms-eval 和 `pip install -e .`。**不要单独 `pip install` 任何包而不带 `-c requirements-lock.txt`**：不带约束装 trl 会把 transformers 升到 5.x，模型构建直接报 `KeyError: 'default'`。
- lmms-eval 直接装 PyPI 的 0.5.0，不再需要克隆 `LMMS_EVAL_PATH`。
- 脚本最后把 `pip freeze` 和 `nvidia-smi` 写到 `results/revision/E0/`。

### 1.2 路径

复制 `.env.example` → `.env`，填写 `DATA_ROOT`、`MODEL_ROOT`、`CKPT_ROOT`（`HF_ENDPOINT` 可选）。评测数据默认在 `${DATA_ROOT}/benchmarks/`，可用环境变量 `BENCH_ROOT` 改。

### 1.3 模型与数据

- `${MODEL_ROOT}/llava-1.5-7b-hf`、`llava-1.5-13b-hf`（llava-hf 格式）
- 训练数据：`${DATA_ROOT}/llava-1.5-665k-instructions/`（JSON + COCO/GQA/OCR-VQA/TextVQA/VG 图像；`scripts/download_llava_images.sh`、`scripts/prepare_data.py`）
- 评测数据（`configs/lmms_tasks/*_local.yaml` 里 `/path/to/benchmarks` 换成 `${BENCH_ROOT}` 后的位置）：`pope/`、`mme/`、`mmbench_en/en/`、`mmbench_cn/cn/dev-00000-of-00001.parquet`、`scienceqa/ScienceQA-IMG/`、`seed-bench/`、`blink/blink_val_combined.parquet`
- 旧服务器的 checkpoint 与 `results.json` 整体迁移到 `${CKPT_ROOT}`，保持原目录结构

装好后跑自检，全部 ✓ 再往下（`!` 是可选项，只提示）：

```bash
python scripts/revision/check_env.py --out results/revision/E0/env_check.json
```

### 1.4 训练起点核查（重要，先做）

作者要求"训练起点以代码为准"。代码行为是：先加载 `${MODEL_ROOT}/llava-1.5-7b-hf`，**如果** `llama_subdir`（默认 `llava1.5-7b-pretrained/vicuna-7b-v1.5`）存在，就把语言模型的 decoder 层和 norm 换成原始 Vicuna；如果 `projector_subdir` 存在，就把投影层换成第一阶段预训练 projector（`token_diversion_x3_fusion.py` 的 `__init__`）。所以起点取决于当时机器上这些目录在不在。

这些模块在 DPVR 训练中是冻结的，checkpoint 里原样保存着它们，**直接比对权重就能确定起点**，不需要靠评测数字去猜：

```bash
python scripts/revision/check_weight_provenance.py --checkpoint <7B DPVR-LF seed=1 checkpoint> \
    --out results/revision/E0/provenance.json
```

它在 CPU 上对比第 0、s-1、s、L-1 层和最终 norm、projector 的张量，输出 `language_model: vicuna | llava_hf` 与 `projector: stage1_projector | llava_hf`。需要 `${MODEL_ROOT}` 下同时有 llava-1.5-7b-hf 与 Vicuna / mm_projector.bin 作对照；缺哪个就从 HF 下载哪个（`lmsys/vicuna-7b-v1.5`、`liuhaotian/llava-v1.5-7b` 的 pretrain projector）。

结论写进 RESULTS.md 的"服务器信息"。之后新服务器上 `llama_subdir` / `projector_subdir` 目录是否存在必须和结论一致（结论是 llava_hf 时就**不要**放这两个目录），否则新训练的起点会和论文不同。

### 1.5 验收（全部通过才进入 §3）

E0 的检查都写在 `jobs/E0.yaml` 里，一条命令并行跑完（用法见 §2.1）：

```bash
python scripts/revision/run_queue.py jobs/E0.yaml --gpus 0-7 --set lf7b_seed1_ckpt=<同上 checkpoint>
```

- [ ] `pytest tests/` 全部通过（无权重时部分测试会 skip，属正常）
- [ ] `check_env.py` 无 ✗
- [ ] `provenance_lf7b_seed1`：起点结论写入 RESULTS.md
- [ ] `reeval_lf7b_seed1`：POPE、SQA 与论文 Table 10 的 .8567 / .6465 差距 < 0.3pp（同时看 `reeval_vanilla7b` 是否与论文 vanilla 行一致，以区分"评测链路不同"和"模型不同"）
- [ ] `latency_vanilla7b` / `latency_lf7b`：B=1、n_img=576、n_txt=128、bf16、10 warmup + 100 次，记录 GPU 型号
- [ ] `smoke_train_lf7b`：50 步 loss 正常下降，显存与论文相近

POPE / SQA 对不上时先停，在 RESULTS.md "问题"里贴 `summary.json` 和 `lmms/` 里的样本日志，等云端排查。

### 1.6 评测入口

所有精度数字都用 `scripts/revision/eval_lmms.py`。它通过 lmms-eval 插件 `dpvr_hf` 调用 lmms-eval 自带的 `llava_hf` handler（论文用的就是它），`use_cache=False`，只把里面的网络换成训练好的 DPVR 模型；`--checkpoint none` 走同一条链路评 vanilla。

```bash
python scripts/revision/eval_lmms.py --checkpoint <dir> --tasks eight --out <dir>   # 八项
python scripts/revision/eval_lmms.py --checkpoint <dir> --tasks pope --limit 20 --out /tmp/x   # 冒烟
```

不要用 `scripts/evaluate.py`：它的 native runner 第一步之后丢掉图像，旧 `--backend lmms_eval` 适配器不套 chat template、也不支持 DPVR-LF。

## 2. 硬件分配原则

### 2.1 任务队列

实验都写成 `jobs/E*.yaml`，由 `scripts/revision/run_queue.py` 分发到空闲 GPU：

```bash
tmux new -s e9
python scripts/revision/run_queue.py jobs/E9.yaml --gpus 0-7 --dry-run   # 先看展开后的命令
python scripts/revision/run_queue.py jobs/E9.yaml --gpus 0-7
```

- 每个任务独占它申请的 GPU 数（默认 1），依赖（`after`）满足后才启动；无依赖的任务并行。
- 显存占用超过 2 GB 的卡视为别人在用，不会分配（`--busy_mb` 可调）；`--gpus` 只写允许用的卡。
- 日志和状态在 `${CKPT_ROOT}/revision_runs/<实验>/<任务>/`（`log.txt`、`status.json`、成功后 `DONE`）。崩溃或中断后**原命令重跑即可**，已完成的任务自动跳过；`--only <任务>` 单跑某项，`--force` 强制重跑。
- 某任务失败时，依赖它的任务标为 blocked，其余继续；队列结束后退出码非 0。
- 训练保持论文设置：单卡、有效 batch 4。8 张卡的用处是让不同 seed / 配置同时跑，不要改成多卡 DDP（会改变有效 batch）。

### 2.2 分配

- 训练任务（E6、E9、E10）每个占一张卡，优先 E6；评测、分析在其余卡上并行。
- 延迟任务要求整卡空闲（任务文件里 `min_free_gb` 接近整卡显存），排在训练之前或训练结束后跑，不要和训练同时在一台机器上测。
- 延迟实验最好在论文用过的型号上测（A800 80G PCIe、RTX 5880 Ada 48G、RTX PRO 6000 Blackwell 97G）。新服务器是 H20 时整张延迟表都要在 H20 上重测（vanilla 与各方法同卡同协议），并在 RESULTS.md 写明。

## 3. 实验顺序

依赖的代码项（C*）未合入 main 前不要开始对应实验。每项实验的任务文件随对应 C 项的 PR 一起加入 `jobs/`；目前已有 `jobs/E0.yaml`、`jobs/E9.yaml`、`jobs/E3_vtw.yaml`（E3 的 VTW 部分与 vanilla 八项）、`jobs/E4.yaml`。

| 顺序 | 实验 | 依赖 | 内容 | 产出对应论文 |
| --- | --- | --- | --- | --- |
| 0 | E0 | — | §1 环境与验收 | — |
| 1 | E6 | C8 | 第二骨干 DPVR-LF 完整训练 + 该骨干 vanilla 评测（骨干待作者确定） | 新表：跨骨干 |
| 1 | E9 | C1 | 修复 seed 后 7B DPVR-LF s=18 三个 seed（1/2/3）重训与八项评测 | Table 2、附录 C |
| 2 | E2 | C6 | 3–4 个模型家族的免训练饱和分析（余弦、注意力质量、logit-lens、残差范数） | Figure 2 扩展 |
| 2 | E4 | C3 | 免训练：从 s 层起冻结图像隐状态；无侧枝 + 末层融合 | §3.1 因果证据 |
| 3 | E3 | C3, C5, C7 | VTW、PruMerge、近期基线：八项精度 + prefill FLOPs + 延迟 | Table 2 |
| 3 | E1 | C2, C7 | KV-cache 解码延迟（7B/13B）与端到端生成延迟（64、256 token） | Table 7 |
| 3 | E5 | C7 | A800 B=1 重测；13B s=28 延迟；解析 FLOPs | Figure 1、Table 6 |
| 4 | E7 | C4 | DPVR-LF + FastV / PruMerge | 新小表 |
| 4 | E8 | C6 | 图像 token 数 144/288/576 下的饱和曲线 | §3.1 / 附录 |
| 5 | E10 | C9 | MLP-only / LoRA 侧枝，30% 数据（`configs/data/llava665k_30pct.yaml`） | Table 4 |
| 5 | E11 | — | 整理已有 K=1–4 的分 benchmark 结果 | Table 5 |

## 4. 汇报方式

每完成一项，在 RESULTS.md 追加一行并 commit + push 到 main（只改 RESULTS.md 和 `results/revision/`）。提交信息格式：`results(E3): VTW 7B eight-bench`。
