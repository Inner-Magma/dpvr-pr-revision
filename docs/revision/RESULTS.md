# 大修实验结果总表

由服务器上的 Claude Code 维护：每完成一项追加一行，新的在上面。云端 Claude 据此整理论文表格。

## 服务器信息

| 项 | 值 |
| --- | --- |
| GPU 型号 × 数量 | **8 × NVIDIA H20-3e**（单卡 139.7 GB，compute capability 9.0）。⚠️ 不是文档里写的 H800，见"问题" §1 |
| 驱动 / CUDA | 驱动 550.127.08；nvidia-smi 报 CUDA 12.4；torch 自带 CUDA 12.8（`torch==2.9.0+cu128`） |
| 环境（pip freeze 路径） | `results/revision/E0/pip-freeze.txt`（175 包）。关键版本：`torch==2.9.0+cu128`、`torchvision==0.24.0+cu128`、`transformers==4.57.1`、`lmms_eval==0.5.0`、`accelerate==1.11.0`、`bitsandbytes==0.49.2`、`numpy==1.26.4`、`qwen-vl-utils==0.0.14` |
| 环境位置 | conda 前缀环境 `$D/install/conda_env_dpvr`（不在 `~/.conda/envs`） |
| 训练起点核查结论（SERVER_HANDOFF §1.4） | **待补**。checkpoint 已定位到 `${CKPT_ROOT}/legacy/token_diversion/llava665k_split18_d1`（7B DPVR-PC，3 个 safetensors + `training_config.json` + `config.json`，完整）；`provenance.json` 尚未生成，见"问题" §4 |
| 评测集样本数核对结论（SERVER_HANDOFF §1.3） | **7/7 全部与论文一致**，见下表 |

### 评测集样本数核对（`download_assets.py --verify`）

| benchmark | 实测 | 论文 | 结论 |
| --- | --- | --- | --- |
| pope | 9000 | 9000 | ✅ |
| mme | 2374 | 2374 | ✅ |
| mmbench_en | 4329 | 4329 | ✅ |
| mmbench_cn | 4329 | 4329 | ✅ |
| scienceqa | 2017 | 2017 | ✅ |
| seedbench | 17990 | 17990 | ✅ |
| blink | 1901 | 1901 | ✅ |

> SEED-Bench 用完整 `lmms-lab/SEED-Bench`（17,990 题），不是 hsf 上那份 5,874 行的旧副本——与 `SERVER_HANDOFF` 的要求一致。核对通过后 `assets.json` 已重新生成。

## 结果

| 日期 | 实验 | 配置 | 关键数字 | 结果路径 | commit |
| --- | --- | --- | --- | --- | --- |
| 2026-10-13 | E0 验收 | `download_assets.py --verify` | 7/7 benchmark 样本数与论文一致 | `results/revision/E0/assets.json` | 4b4f56a |
| 2026-10-13 | E0 验收 | `check_env.py` | 软件栈/路径/8 卡可用性全过；模型与 665k 图像在跑这条时尚未下载完 | `results/revision/E0/env_check.json` | 4b4f56a |

## 问题

### 1. 卡型与文档不符：拿到的是 8×H20-3e，不是 H800

`AGENTS.md` 全文以及 `SERVER_HANDOFF` 都按 8×H800 写。实际容器里 `nvidia-smi` 是 **8 × NVIDIA H20-3e**（139.7 GB/卡，sm_90）。

- 影响面：E1/E5 的延迟表（Table 7）是**论文里唯一强依赖卡型的实验**，H20 与 H800 的显存带宽差一个量级，这两项数字不能声称是 H800 上的。其余训练/精度实验是跨卡可比的，但 RESULTS.md 记录时必须写明卡型。
- 复现：`cat results/revision/E0/nvidia-smi.txt`
- 待作者决策：延迟表按 H20 如实标注，还是想办法换 H800 队列重跑。

### 2. 作业占用情况（2026-10-13 11:49 查）

| appid | 状态 | 说明 |
| --- | --- | --- |
| `psxmf2vxrs643dy3` | **RUNNING** | 本容器所在作业，8 卡各占约 75 GB |
| `psxl413xrs7fieen` | **INITIALIZED** | 10-11 15:51 起，10-13 11:38 由 RUNNING 退回 INITIALIZED（被抢占/重排）。这是准备替换成 DPVR 作业的那个占卡作业 |

队列 `root.hldy_training_cluster.hadoop-aipnlp.h800_vl_sp`：REMAIN 55 gcores80g、PENDING 8。

### 3. HF datasets 缓存写在根分区，导致 SEED-Bench 生成 ENOSPC（已绕过）

`download_assets.py --verify` 在 `seedbench` 一项上报 `DatasetGenerationError`，根因是：

```
OSError: [Errno 28] No space left on device
```

HF datasets 默认把 arrow 缓存写进 `~/.cache/huggingface/datasets`，落在 **50 GB 的 overlay 根分区**（当时可用 17 GB）。SEED-Bench 的 273 个 parquet 展开成 arrow 后远超 17 GB，写到 16,430/17,990 条时爆盘。

- 绕过办法（**未改任何仓库代码**，纯环境变量）：把缓存指到大盘再跑即可通过。
  ```bash
  export HF_HOME=$CKPT_ROOT/../hf_cache
  export HF_DATASETS_CACHE=$HF_HOME/datasets
  export HF_HUB_CACHE=$HF_HOME/hub
  ```
- 复现：不带上面三个变量直接跑 `python scripts/revision/download_assets.py --verify`。
- 建议（需作者改代码）：`download_assets.py` / `run_queue.py` 应该把 datasets 缓存默认指到 `DATA_ROOT` 或 `CKPT_ROOT` 下的大盘，否则每次新容器都会撞同一堵墙。

### 4. E0 尚未真正跑完

`results/revision/E0/` 目前只有环境与资产核对产物（`env_check.json`、`nvidia-smi.txt`、`pip-freeze.txt`、`assets.json`）。以下 E0 项**还没有产物**：

- `provenance.json`（训练起点核查，需要 `--set prov_ckpt=...`）
- `reeval_vanilla7b`（vanilla 7B 八基准重测）
- `latency_vanilla7b` / `latency_lf7b`（延迟，且受问题 §1 卡型影响）
- `smoke_train_lf7b`（50 step 冒烟训练）

另：665k 图像还缺两项——`coco/train2017`（0 张）、`vg/VG_100K`（0 张，`vg_images1.zip` 在 cs.stanford.edu 上三轮 404），`ocr_vqa/images` 抽取返回 0 张。`DL_IMAGES_DONE 有缺项，未放行`。这三项补齐前不能进阶段 2。

### 5. `.env` 缺 `HF_CKPT_REPO`

`SERVER_PROMPTS.md` 启动前第 4 步要求 `.env` 里有 `HF_CKPT_REPO`，当前没有。checkpoint 实际落在 `${CKPT_ROOT}/legacy/` 而不是文档假定的 `${CKPT_ROOT}/hsf/`，E0 启动时 `--set prov_ckpt=` 要按实际路径传。
