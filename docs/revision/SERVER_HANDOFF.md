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

### 1.1 软件版本（与论文附录 B 一致，务必锁死）

```
Python 3.10
torch 2.9.0+cu128
transformers 4.57.1
trl 0.23.0
accelerate 1.11.0
bitsandbytes 0.49.2
lmms-eval 0.5.0
```

```bash
conda create -n dpvr python=3.10 -y && conda activate dpvr
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1 trl==0.23.0 accelerate==1.11.0 bitsandbytes==0.49.2
pip install -e ".[analysis,profile,dev]"
# lmms-eval 0.5.0：克隆到 $LMMS_EVAL_PATH 后 pip install -e
pip freeze > results/revision/E0/pip-freeze.txt
```

`pyproject.toml` 只写了版本下限，不要用它解析出的最新版本。若第二骨干需要更新的 transformers，**另建环境**，不要升级主环境。

### 1.2 路径

复制 `.env.example` → `.env`、`configs/paths.yaml.example` → `configs/paths.yaml`，填写 `DATA_ROOT`、`MODEL_ROOT`、`CKPT_ROOT`、`LMMS_EVAL_PATH`、`HF_ENDPOINT`。

### 1.3 模型与数据

- `${MODEL_ROOT}/llava-1.5-7b-hf`、`llava-1.5-13b-hf`（llava-hf 格式）
- 训练数据：`${DATA_ROOT}/llava-1.5-665k-instructions/`（JSON + COCO/GQA/OCR-VQA/TextVQA/VG 图像；`scripts/download_llava_images.sh`、`scripts/prepare_data.py`）
- 八个评测集本地副本：POPE、MME、MMBench-EN/CN、ScienceQA、SEED-Bench、BLINK（`configs/lmms_tasks/*_local.yaml`）
- 旧服务器的 checkpoint 与 `results.json` 整体迁移到 `${CKPT_ROOT}`，保持原目录结构

### 1.4 训练起点核查（重要，先做）

作者要求"训练起点以代码为准"。代码行为是：先加载 `${MODEL_ROOT}/<base_model>`（`llava-1.5-7b-hf`），**如果** `${MODEL_ROOT}/<llama_subdir>`（默认 `llava1.5-7b-pretrained/vicuna-7b-v1.5`）存在，就把语言模型层替换为原始 Vicuna；如果 `projector_subdir` 存在，就把投影层替换为第一阶段预训练 projector（见 `token_diversion_x3_fusion.py` 的 `__init__`）。评测时 `from_trained` 也走同一路径，所以**结果取决于当时机器上这些目录是否存在**。

请确定论文主实验实际用的是哪种起点，并写进 RESULTS.md：

1. 若能登录旧服务器：检查 `${MODEL_ROOT}/llava1.5-7b-pretrained/vicuna-7b-v1.5` 与 `.../mm_projector.bin` 在训练时是否存在；查看 checkpoint 目录里 `ConfigSaveCallback` 保存的 config。
2. 无论能否登录：在新服务器上分别用"替换"与"不替换"两种方式构建 7B DPVR-LF seed=1 模型，加载 checkpoint，在 POPE 上各评一次，哪种接近论文 Table 10 的 .8567 就是哪种。
3. 后续所有实验沿用核实后的起点；新服务器上目录是否存在要和它一致。

### 1.5 验收（全部通过才进入 §3）

- [ ] `pytest tests/` 全部通过
- [ ] 7B DPVR-LF seed=1 checkpoint 重评 POPE、SQA，与论文 Table 10 的 .8567 / .6465 差距 < 0.3pp
- [ ] vanilla 7B 在 B=1、n_img=576、n_txt=128、bf16、10 warmup + 100 次测量下测一次 forward 延迟，记录 GPU 型号
- [ ] DPVR-LF `max_steps=50` 训练 smoke test：loss 正常下降，显存与论文相近
- [ ] 1.4 的结论写入 RESULTS.md

## 2. 硬件分配原则

- 训练任务（E6、E9、E10）占 80GB 卡，优先 E6。
- 推理任务（评测、分析、延迟）在其余卡上并行。
- 延迟实验最好在论文用过的型号上测（A800 80G PCIe、RTX 5880 Ada 48G、RTX PRO 6000 Blackwell 97G）；型号不同时整张延迟表都要重测，并在 RESULTS.md 写明。

## 3. 实验顺序

依赖的代码项（C*）未合入 main 前不要开始对应实验。

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
