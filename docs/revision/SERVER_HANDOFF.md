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

服务器是 **8×H800（80 GB，sm_90）**。环境与 A800 / H20 相同：CUDA 12.8 的 torch 轮子同时支持 sm_90 与 sm_80，全程 bf16，不用 flash-attn（LLaVA 用 eager/sdpa，Qwen2.5-VL 必须用 sdpa），所以不需要编译任何 CUDA 扩展。

- 驱动：`nvidia-smi` 显示的驱动 ≥ 570 时直接用默认 cu128 轮子；更老的驱动若 `torch.cuda.is_available()` 为 False，改用 `TORCH_INDEX=https://download.pytorch.org/whl/cu126`（版本号不变），并在 RESULTS.md 记下。
- 显存：任务文件里 `min_free_gb` 最大 70，80 GB 卡满足；不要调高 batch 来"用满"显存，有效 batch 是论文设置。
- 磁盘：`DATA_ROOT` + `MODEL_ROOT` + `CKPT_ROOT` 合计约需 1 TB（665k 图像约 100 GB，模型约 120 GB，评测数据约 30 GB，其余是新 checkpoint 与日志）。开跑前 `df -h` 确认。

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

`.env` 里还有发结果邮件用的 `SMTP_HOST`、`SMTP_PORT`、`SMTP_USER`、`SMTP_PASSWORD`、`MAIL_TO`（见 §4）。`.env` 已在 `.gitignore` 里，**它和其中的地址、授权码都不得出现在任何 commit、日志摘要或 RESULTS.md 里**。

### 1.3 模型与数据

除旧 checkpoint 外都可以用一条脚本下载到下面的位置（走 `.env` 的 `HF_ENDPOINT`，可断点续传，重跑会跳过已下载的）：

```bash
python scripts/revision/download_assets.py --list                          # 看清单
python scripts/revision/download_assets.py models benchmarks llava665k     # 模型、八项评测数据、665k JSON
bash scripts/download_llava_images.sh ${DATA_ROOT}/llava-1.5-665k-instructions   # 665k 图像（原始站点）
python scripts/revision/download_assets.py --verify                        # 评测集样本数与论文对照
```

`--verify` 把每个评测集的样本数和论文数字背后的样本数（POPE 9000、MME 2374、MMB-EN/CN 4329、SQA 2017、SEED 5874、BLINK 1901）对照，写到 `results/revision/E0/assets.json`。**SEED 论文用的是 5874 条的子集**，新下载的完整版条数不同：旧服务器上有原文件时优先拷过来；拷不到就照常评测，但在 RESULTS.md "问题"里写明 SEED 列与论文不可直接比较。

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

它在 CPU 上对比第 0、s-1、s、L-1 层和最终 norm、projector 的张量，输出 `language_model: vicuna | llava_hf` 与 `projector: stage1_projector | llava_hf`。需要 `${MODEL_ROOT}` 下同时有 llava-1.5-7b-hf 与 Vicuna / mm_projector.bin 作对照。`download_assets.py models` 把后两者放在 `${MODEL_ROOT}/reference/`（`vicuna-7b-v1.5/`、`llava-v1.5-7b-pretrain/mm_projector.bin`），`jobs/E0.yaml` 的 provenance 任务用 `--vicuna` / `--projector` 指向那里。故意不放在 `llama_subdir` / `projector_subdir` 的位置：那两个目录一旦存在就会改变新训练的起点（见下一段）。

结论写进 RESULTS.md 的"服务器信息"。之后新服务器上 `llama_subdir` / `projector_subdir` 目录是否存在必须和结论一致（结论是 llava_hf 时就**不要**放这两个目录），否则新训练的起点会和论文不同。

### 1.5 验收（全部通过才进入 §3）

E0 的检查都写在 `jobs/E0.yaml` 里，一条命令并行跑完（用法见 §2.1）：

```bash
python scripts/revision/run_queue.py jobs/E0.yaml --gpus 0-7 --set lf7b_seed1_ckpt=<同上 checkpoint>
```

- [ ] `pytest tests/` 全部通过（无权重时部分测试会 skip，属正常）
- [ ] `check_env.py` 无 ✗
- [ ] `provenance_lf7b_seed1`：起点结论写入 RESULTS.md
- [ ] `assets_verify`：八项评测集样本数与论文一致；不一致的（多半是 SEED）写进 RESULTS.md "问题"
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
- 训练保持论文设置：单卡、有效 batch 4。8 张卡的用处是让不同 seed / 配置同时跑，不要改成多卡 DDP（会改变有效 batch）。唯一例外是 E6 的全量训练：4 卡 × 每卡 1 条，有效 batch 仍是 4（`jobs/E6.yaml`）。

### 2.2 分配

- 训练任务（E6、E9、E10）每个占一张卡，优先 E6；评测、分析在其余卡上并行。
- 延迟任务（`jobs/E1.yaml` 生成延迟、`jobs/E5.yaml` 预填充延迟与解析 FLOPs）要求整卡空闲，排在训练之前或训练结束后跑，不要和训练同时在一台机器上测；运行时只给队列一张卡（`--gpus 0`），让各项依次测。E1 需要 C2 已合入。
- 论文的延迟是在 A800 80G PCIe、RTX 5880 Ada 48G、RTX PRO 6000 Blackwell 97G 上测的。这台是 H800，所以**整张延迟表都在 H800 上重测**（vanilla 与各方法同卡、同协议、同一次开机），每条结果写明 GPU 型号，RESULTS.md 里注明"H800 重测，与论文旧表的绝对值不可混用，只比相对加速"。

## 3. 实验顺序

依赖的代码项（C*）未合入 main 前不要开始对应实验。任务文件索引见下方 §3.1。

| 顺序 | 实验 | 依赖 | 内容 | 产出对应论文 |
| --- | --- | --- | --- | --- |
| 0 | E0 | — | §1 环境与验收 | — |
| 1 | E6 | C8 | 第二骨干 Qwen2.5-VL-7B-Instruct：vanilla 评测 + DPVR-LF 冒烟训练 + 全量训练与八项评测（`jobs/E6.yaml`；分割层不确定时先跑 `jobs/E6_split.yaml`） | 新表：跨骨干 |
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

### 3.1 任务文件索引

C0–C10 已全部合入 main（2026-10-11），下列任务文件都可以直接跑；先 `--dry-run` 看展开后的命令。

| 任务文件 | 实验 | 内容 | 需要的卡 |
| --- | --- | --- | --- |
| `jobs/E0.yaml` | E0 | 环境与旧 checkpoint 验收（§1） | 多卡并行 |
| `jobs/E6.yaml` | E6 | Qwen2.5-VL-7B：vanilla 评测、冒烟训练、全量训练（4 卡）与评测 | 1 + 4 |
| `jobs/E6_split.yaml` | E6 | Qwen2.5-VL-7B 分割层 s=12/16/20，10% 数据 | 每项 1 |
| `jobs/E9.yaml` | E9 | 7B DPVR-LF s=18 三个 seed 重训与评测 | 每项 1 |
| `jobs/E2.yaml` | E2 | 五个模型家族的饱和分析 | 每项 1 |
| `jobs/E8.yaml` | E8 | LLaVA-1.5 7B/13B 不同图像 token 数的饱和曲线 | 每项 1 |
| `jobs/E4.yaml` | E4 | 免训练因果检查（冻结图像隐状态、无侧枝末层融合） | 每项 1 |
| `jobs/E3_vtw.yaml` | E3 | VTW 与 vanilla 八项 | 每项 1 |
| `jobs/E3_baselines.yaml` | E3 | FastV、PyramidDrop、PruMerge、VisionZip 基线 | 每项 1 |
| `jobs/E1.yaml` | E1 | 端到端生成延迟（KV cache） | 整卡独占，`--gpus 0` |
| `jobs/E5.yaml` | E5 | 预填充延迟与解析 FLOPs | 整卡独占，`--gpus 0` |
| `jobs/E7.yaml` | E7 | DPVR-LF 叠加 token 压缩 | 每项 1 |
| `jobs/E10.yaml` | E10 | 侧枝变体（MLP-only、LoRA、第 s 层初始化） | 每项 1 |

E2 / E8（`jobs/E2.yaml`、`jobs/E8.yaml`，入口 `scripts/revision/saturation.py`）：
- 每个模型 500 个 LLaVA-665k 样本（seed 0），单卡 bf16、eager attention；输出 `curves.npz`（逐样本逐层）、`summary.json`（逐层中位数与 IQR、各指标的转折层）、`saturation.pdf`；`compare` 任务把各模型叠在一张图上（横轴 layer/L）并写转折层表 `compare.md`。
- 先跑 `llava15_7b`，核对与论文 Figure 2 一致（图像余弦 ≥0.92、L18 后注意力质量约 0.04、图像 KL 50% 转折在 L22），不一致先停下排查，再跑其他模型。
- LLaVA-1.5 以外的模型要先下载 HF 格式权重到 `${MODEL_ROOT}/<目录名>`（目录名见任务文件；第二骨干已定为 Qwen2.5-VL-7B-Instruct，即 `qwen25vl_7b` 一项）。Qwen2.5-VL 的图像 token 数用 `--max_pixels` 限制在约 640。
- E8 在 projector 之后把 24×24 图像特征平均池化成 12×24 / 12×12 / 6×6，只支持 LLaVA-1.5 结构。

### E6 说明（第二骨干）

- 模型：`${MODEL_ROOT}/Qwen2.5-VL-7B-Instruct`（HF 格式，transformers 4.57.1 直接加载）。
- 评测走 lmms-eval 0.5.0 自带的 `qwen2_5_vl` handler（插件 `dpvr_qwen25vl`，`use_cache=False`），提示词和图像分辨率（`min_pixels=256·28²`、`max_pixels=2048·28²`）都用 handler 默认值；`eval_lmms.py` 按基座名或 checkpoint 的 method 自动选 handler。vanilla 与 DPVR 同一条链。依赖 `qwen-vl-utils==0.0.14`（已在 lock 文件里）。
- 训练：`configs/method/token_diversion_x3_fusion_qwen25vl.yaml`，和 7B LLaVA 同配方（冻结主干、1 层侧枝、K=1、665k 一个 epoch、lr 1e-4、有效 batch 4），训练图像分辨率与评测一致。注意力必须是 sdpa 或 eager（层接收显式 4D mask）。
- 分割层：默认 s=16（28 层，与 7B 的 18/32 同比例）。先看 E2 里 `qwen25vl_7b` 的饱和曲线；不明确时跑 `jobs/E6_split.yaml`（s=12/16/20，10% 数据），用 `--set split=<s>` 把选定值传给 `jobs/E6.yaml`，并在 RESULTS.md 写明依据。
- 顺序：`vanilla_eval` 与 `smoke_train` 立即开始；冒烟评测通过后才开全量训练。冒烟失败（显存、数据、checkpoint 格式）先修，不要直接跑全量。
- 解码无 KV cache（每步重算全序列），只影响速度，不影响精度协议。

## 4. 汇报方式

每完成一项：

1. 汇总：`python scripts/revision/collect_results.py`。它扫描 `${CKPT_ROOT}/revision_runs/*/*/`，把小文件（≤512 KB 的 json/md/csv/txt/pdf/png/yaml 与日志末 200 行）拷到 `results/revision/<实验>/<任务>/`，并生成 `results/revision/summary/eval_table.{md,csv}`（论文八列，POPE 等为百分数，MME 为原始分）和 `jobs.{md,csv}`（每个任务的状态）。`lmms/` 样本日志与 checkpoint 不拷。
2. 在 RESULTS.md 追加一行（关键数字 + 结果路径 + commit），然后只 `git add docs/revision/RESULTS.md results/revision` 并 push 到 main。提交信息格式：`results(E3): VTW 7B eight-bench`。
3. 发邮件：`python scripts/revision/send_results.py --subject "E3 done"`。正文是两张汇总表加 RESULTS.md，附件是两个 CSV 和 `results/revision/` 的 zip（上限 15 MB）。先 `--dry-run` 看一次（生成 `results/revision/summary/mail.eml`，不发送，也不要提交它）。

什么时候发邮件：E0 验收结束、每个实验（E*）全部任务结束、任何"异常就停"的情况（主题写 `BLOCKED: ...`），以及全部实验结束时一封总结。不要每个任务都发。发送失败（端口不通、认证失败）时不要换别的发送途径，在 RESULTS.md "问题"里记下错误并继续实验。
