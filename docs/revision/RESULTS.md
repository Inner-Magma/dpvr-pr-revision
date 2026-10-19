# 大修实验结果总表

由服务器上的 Claude Code 维护：每完成一项追加一行，新的在上面。云端 Claude 据此整理论文表格。

## 服务器信息

| 项 | 值 |
| --- | --- |
| GPU 型号 × 数量 | **2026-10-18 起：8 × NVIDIA H20-3e**（143771 MiB/卡，500 W，sm_90），本地容器 `psxob0gxrs643dy3-worker-0`，8 卡全空闲，不经过 hope 队列，直接在容器里跑。<br>**2026-10-16 前**：8 × NVIDIA H100 80GB HBM3（81559 MiB/卡，700 W，sm_90），申请的是 `worker.gcoresh800-80g = 8`，拿到的既不是文档写的 H800、也不是中途出现过的 H20-3e，见"问题" §1。**E0 的延迟数字是在 H100 上测的，引用时按 H100 标注；从 E1 起的延迟表在 H20-3e 上重测。** |
| 驱动 / CUDA | 驱动 550.127.08；nvidia-smi 报 CUDA 12.4；torch 自带 CUDA 12.8（`torch==2.9.0+cu128`） |
| 容器镜像 | `.../aipnlp_base_pytorch2.3.1_cuda12.4.0_python3.9_1_18_glibc_2.28_2532322d`（CentOS 7 壳 + 换过的 glibc 2.28）。原镜像 glibc 2.17 跑不动本环境，见"问题" §6 |
| 环境（pip freeze 路径） | `results/revision/E0/pip-freeze.txt`（175 包）。关键版本：`torch==2.9.0+cu128`、`torchvision==0.24.0+cu128`、`transformers==4.57.1`、`lmms_eval==0.5.0`、`accelerate==1.11.0`、`bitsandbytes==0.49.2`、`numpy==1.26.4`、`qwen-vl-utils==0.0.14` |
| 环境位置 | conda 前缀环境 `$D/install/conda_env_dpvr`（不在 `~/.conda/envs`） |
| 训练起点核查结论（SERVER_HANDOFF §1.4） | **`language_model = llava_hf`，`projector = llava_hf`**（即 `llama_subdir` / `projector_subdir` 当时**不存在**，起点就是 `llava-1.5-7b-hf` 原版）。11 个探针权重与 `llava-1.5-7b-hf` 的距离全为 **0.0**，与 vicuna / stage1_projector 参照均为非零（4e-3 ~ 1.3e-2）。→ **新服务器上这两个目录必须保持不存在**，否则新训练起点会与论文不同。产物 `results/revision/E0/provenance.json` |
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
| 2026-10-20 | **E4 无分支 LF s=18 k=3 / k=4** | `lf_no_branch`：图结构、侧分支换 identity、不训练，融合 3 / 4 份视觉后缀 | **k=3**：85.34 / 1461.07 / 343.57 / 74.10 / 71.61 / 64.20 / **60.23** / 38.98（rc=0，153.7 min）；**k=4**：85.38 / 1470.57 / 328.57 / 74.08 / 71.49 / 64.06 / **60.18** / 39.61（rc=0，156.0 min）。k=3/k=4 彼此几乎重合（最大差 SEED +0.05pp、MME-C +1500）。与 k=1/k=2（SEED 65.44 / 65.39）：**POPE/MME/MMB/SQA/BLINK 全在噪声内，但 SEED 在 k≥3 时一次性掉 5.2pp 并持平**——融合份数从 2 加到 3 时 SEED 出现台阶，之后不再变化。**E4 无分支 k=1..4 全部跑完**：侧分支（含训练）在精度上无可测价值，图结构本身对精度中性；k≥3 的 SEED 台阶是本组唯一非噪声差异 | `results/revision/E4/`；`revision_runs/E4/nobranch7b_s18_k{3,4}/` | 5933ddf（代码 35b5499） |
| 2026-10-20 | **E7 LF+VisionZip 64** | 训练好的 7B DPVR-LF s=18（seed1）叠 VisionZip（dominant=54 + contextual=10，保留 64/576 图 token，keep_cls=True） | **79.36 / 1314.94 / 304.64 / 70.29 / 66.94 / 64.06 / 52.80 / 38.66**，rc=0，96.8 min。与 vanilla+VisionZip-64（79.50 / 1329.23 / 299.29 / 70.99 / 67.52 / 63.91 / 57.35 / 38.24）几乎重合：POPE −0.14pp、MME-P −14.3、MME-C +5.4、MMB-EN −0.70pp、MMB-CN −0.58pp、SQA +0.15pp、SEED **−4.55pp**、BLINK +0.42pp；与 LF+PruMerge（76.28 / 1257.99 / 320.00 / 70.46 / 67.04 / 63.21 / 55.79 / 38.72）相比 POPE +3.08pp、SEED −2.99pp，同属一个档位。两者都远低于 LF 自身（85.40 / 1479.26 / 341.79 / 73.90 / 70.55 / 64.40 / 64.89 / 38.30）→ **与 PruMerge 结论一致：vision-side 显式预算合并本身就是主要损失源（POPE −6pp、SEED −12pp），LF 叠加既不补偿也不加剧**。E7 的 vision-side 支线至此跑完 | `results/revision/E7/`；`revision_runs/E7/lf7b_visionzip_64/` | cc538d9（代码 35b5499） |
| 2026-10-20 | **E7 LF+PruMerge** | 训练好的 7B DPVR-LF s=18（seed1）叠 PruMerge（vision-side，n_keep 每图 IQR 自适应） | **76.28 / 1257.99 / 320.00 / 70.46 / 67.04 / 63.21 / 55.79 / 38.72**，rc=0，114.4 min。与 vanilla+PruMerge（76.53 / 1284.75 / 312.50 / 70.69 / 67.15 / 63.41 / 51.90 / 38.56）几乎重合（POPE −0.25pp、SEED +3.89pp），**但两者都远低于 vanilla 的 85.36 / 65.45**：PruMerge 本身在 POPE −8.8pp、SEED −13.6pp，叠加 LF 不能挽回 → **vision-side 自适应合并在 7B 上本身就是重损失，LF 与它不同源、无互补** | `results/revision/E7/`；`revision_runs/E7/lf7b_prumerge/` | 0ae51ed（代码 35b5499） |
| 2026-10-20 | **E4 无分支 LF s=18 k=2** | `lf_no_branch`：图结构但侧分支换 identity，不训练，融合 2 份视觉后缀 | **85.46 / 1484.73 / 316.43 / 74.01 / 71.52 / 63.61 / 65.39 / 39.03**，rc=0，154.2 min。与 k=1（85.43 / 1462.61 / 336.07 / 74.15 / 71.38 / 63.66 / 65.44 / 39.08）几乎逐项重合（最大差 MME-C −19.6、MME-P +22.1）。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）八项全在噪声内 → **k=1→2 不带来增益，也不掉点**；侧分支即便不训练，图结构本身对精度是中性的 | `results/revision/E4/`；`revision_runs/E4/nobranch7b_s18_k2/` | 1be4aaf（代码 35b5499） |
| 2026-10-20 | **E3_baselines VisionZip 192** | `visionzip7b_192`：dominant=162 + contextual=30（保留 192/576 图 token） | **84.70 / 1425.58 / 349.29 / 73.34 / 70.43 / 64.25 / 62.48 / 38.35**，rc=0，128.5 min。与 VisionZip-128（82.81 / 1393.56 / 336.07 / 72.67 / 69.97 / 64.50 / 60.68 / 38.14）：POPE **+1.89pp**、SEED **+1.80pp**、MME-P +32.0、MME-C +13.2。距 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）只差 POPE −0.66pp、SEED −2.97pp、MME-P −64.2 → **VisionZip 曲线随预算单调逼近 vanilla，192 token 时 POPE 已基本追平，SEED 仍差约 3pp** | `results/revision/E3/`；`revision_runs/E3/visionzip7b_192/` | 1be4aaf（代码 35b5499） |
| 2026-10-19 | **E7 LF+FastV r=0.25** | E7：训练好的 7B DPVR-LF s=18（seed1）叠 FastV k=2，保留 **25%** 图 token，last 打分 | **79.52 / 1392.02 / 348.21 / 73.18 / 69.72 / 64.35 / 56.58 / 39.24**，rc=0，111.7 min。与 r=0.5（83.20 / 1451.40 / 335.00 / 73.97 / 70.46 / 64.45 / 64.36 / 38.82）：POPE **−3.68pp**、MME-P −59.4、MME-C +13.2、MMB-EN −0.79pp、MMB-CN −0.74pp、SQA −0.10pp、SEED **−7.78pp**、BLINK +0.42pp。与 LF 单独相比 POPE −5.88pp、SEED −8.31pp → **预算砍半后 LF+FastV 掉得比 LF 自身在 576 token 时多得多**，SEED/POPE 是主要损失位；与 LF 曲线同一形态（vision-side/LLM-side 压到 1/4 token 时退化集中在计数与多选） | `results/revision/E7/`；`revision_runs/E7/lf7b_fastv_k2_r0.25/` | 35b5499（代码） |
| 2026-10-19 | **E7 LF+FastV r=0.5** | E7：训练好的 7B DPVR-LF s=18（seed1）叠 FastV k=2，保留 50% 图 token，last 打分 | **83.20 / 1451.40 / 335.00 / 73.97 / 70.46 / 64.45 / 64.36 / 38.82**，rc=0，126.2 min。与 LF 单独（85.40 / 1479.26 / 341.79 / 73.90 / 70.55 / 64.40 / 64.89 / 38.30）：POPE **−2.20pp**、MME-P −27.9、MME-C −6.8、MMB-EN +0.07pp、MMB-CN −0.09pp、SQA +0.05pp、SEED −0.53pp、BLINK +0.52pp。与 vanilla+FastV r=0.5 last（83.03 / 1462.3 / 335.7 / 74.08 / 71.01 / 64.80 / 64.50 / 38.93）几乎重合：**LF 与 FastV 叠加没有额外增益也没有额外损失**，说明两者压缩的是同一类冗余 | `results/revision/E7/`；`revision_runs/E7/lf7b_fastv_k2_r0.5/` | 35b5499（代码） |
| 2026-10-19 | **E4 无分支 LF s=18 k=1** | `lf_no_branch`：DPVR-LF 图结构但侧分支换 identity，不训练，k=1 | **85.43 / 1462.61 / 336.07 / 74.15 / 71.38 / 63.66 / 65.44 / 39.08**，rc=0，151.6 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）**逐项在噪声内**；与训练后的 LF（85.40 / 1479.26 / 341.79 / 73.90 / 70.55 / 64.40 / 64.89 / 38.30）也几乎相同 → **训练后的侧分支在精度上不贡献可测增益**，其价值需从效率/稳定性角度论证（k=2/3/4 仍待跑） | `results/revision/E4/`；`revision_runs/E4/nobranch7b_s18_k1/` | 35b5499（代码） |
| 2026-10-19 | **E3_baselines VisionZip 128** | `visionzip7b_128`：dominant=108 + contextual=20（保留 128/576 图 token） | **82.81 / 1393.56 / 336.07 / 72.67 / 69.97 / 64.50 / 60.68 / 38.14**，rc=0，116.6 min。与 VisionZip-64（79.50 / 1329.23 / 299.29 / 70.99 / 67.52 / 63.91 / 57.35 / 38.24）相比：**保留翻倍后 POPE +3.31pp、SEED +3.33pp、MME-P +64.3、MME-C +36.8**；但距 vanilla 仍 POPE −2.55pp、SEED −4.77pp。**vision-side 合并的收益随预算单调，但 128 token 仍未追平 576** | `results/revision/E3/`；`revision_runs/E3/visionzip7b_128/` | 35b5499（代码） |
| 2026-10-19 | **E9 DPVR-LF 7B s=18 seed2 评测** | seed=2 重新训练的同配置 LF（s=18，k=3，全量 665k），八项评测 | **85.73 / 1416.27 / 327.50 / 73.67 / 70.32 / 64.20 / 65.05 / 37.87**，rc=0，145.9 min。与 seed1（85.40 / 1479.26 / 341.79 / 73.90 / 70.55 / 64.40 / 64.89 / 38.30）：POPE +0.33pp、MME-P −63.0、MME-C −14.3、MMB-EN −0.23pp、MMB-CN −0.23pp、SQA −0.20pp、SEED +0.16pp、BLINK −0.43pp（BLINK stderr 1.11，<0.4σ）。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：POPE +0.37pp、MME-P −73.5、MME-C −16.4、MMB-EN −0.46pp、MMB-CN −0.94pp、SQA +0.59pp、SEED −0.40pp、BLINK −1.21pp。**八项仍在噪声内，seed 间离散（MME-P 63 分、BLINK 0.43pp）是主要方差来源** → 汇总时给三 seed 的均值±标准差，不挑最好 seed | `results/revision/E9/eval_seed2/`；`revision_runs/E9/eval_seed2/` | 35b5499（代码） |
| 2026-10-19 | **E9 DPVR-LF 7B s=18 seed1** | 重新训练的 LF 主配置，seed=1，s=18，k=3，全量 665k，单卡有效 batch 4 | **85.40 / 1479.26 / 341.79 / 73.90 / 70.55 / 64.40 / 64.89 / 38.30**，rc=0，148.2 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：POPE **+0.04pp**、MME-P −10.5、MME-C −2.1、MMB-EN −0.23pp、MMB-CN −0.71pp、SQA +0.79pp、SEED −0.56pp、BLINK −0.78pp。**八项全部在噪声内**，主结论成立；这是 E7 的基座 checkpoint | `results/revision/E9/eval_seed1/`；`revision_runs/E9/eval_seed1/` | 35b5499（代码） |
| 2026-10-19 | **E3_baselines VisionZip 64** | `visionzip7b`：vision-side，dominant=54 + contextual=10（保留 64 个图 token），keep_cls=True | **79.50 / 1329.23 / 299.29 / 70.99 / 67.52 / 63.91 / 57.35 / 38.24**，rc=0，104.0 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：**POPE −5.86pp（recall 62.27%、precision 95.02%）**、MME-P −160.6、MME-C −44.6、MMB-EN −3.15pp、MMB-CN −3.74pp、SEED −8.10pp、SQA +0.30pp、BLINK −0.84pp。保留 64/576（11%）时仍好于同属 vision-side 的 PruMerge（POPE 76.53 / SEED 51.90），说明**显式保留 dominant+contextual 比纯自适应 IQR 稳**；但与 FastV 系同预算相比无优势 | `results/revision/E3/`；`revision_runs/E3/visionzip7b_64/` | 35b5499（代码） |
| 2026-10-19 | **E3_baselines PruMerge** | `prumerge7b`：vision-side 合并，n_keep 由每图 IQR 自适应，k=32，feature_layer=-2 / key_layer=-1 | **76.53 / 1284.75 / 312.50 / 70.69 / 67.15 / 63.41 / 51.90 / 38.56**，rc=0，110.9 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：**POPE −8.83pp（recall 55.80%、precision 95.33%）**、MME-P −205.0、SEED −13.55pp、MMB-EN −3.44pp、MMB-CN −4.11pp、MME-C −31.4、SQA −0.20pp、BLINK −0.52pp。**差距最大的一项**：自适应 n_keep 在上限内砍掉的图 token 太多，POPE/SEED 这类细粒度判别任务重伤，文本主导的 SQA/BLINK 基本不动；对比同预算 FastV r=0.5 last（POPE 83.03 / SEED 64.50）明显更差 | `results/revision/E3/`；`revision_runs/E3/prumerge7b/` | 35b5499（代码） |
| 2026-10-19 | **E4 冻结基线 s=28** | `freeze_image`，冻结视觉塔 + projector，`split=28` | **85.36 / 1482.05 / 336.07 / 74.24 / 71.38 / 63.96 / 65.47 / 38.93**，rc=0，183.2 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：**逐项在噪声内**（POPE ±0.00pp、MME-P −7.8、MME-C −7.9，MMB/SQA/SEED/BLINK ±0.35pp）。**冻结扫描 7 个分割层全部跑完（s=8/12/16/18/20/24/28）**：s=8 崩塌，s≥12 全平 → 结论对分割层选择不敏感 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s28/` | 35b5499（代码） |
| 2026-10-19 | **E3_baselines PyramidDrop** | `pdrop7b`：PyramidDrop，阶段边界 8/16/24，每段保留 50% 剩余图 token，按最后一层（last）注意力打分 | **84.91 / 1470.88 / 316.43 / 73.85 / 71.66 / 63.21 / 64.3 / 39.66**，rc=0，136.0 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：POPE **−0.45pp**、MME-P −18.9、MME-C −27.5、MMB-EN −0.28pp、MMB-CN +0.40pp、SQA −0.40pp、SEED −1.15pp、BLINK +0.58pp。**多阶段渐进剪枝比单层 FastV 稳得多**（同 50% 预算下 FastV r=0.5 last 的 POPE 是 83.03、SEED 64.50），几乎贴住 vanilla；代价是需要三次打分，且仍会累积误差 | `results/revision/E3/`；`revision_runs/E3/pdrop7b/` | 35b5499（代码） |
| 2026-10-19 | **E4 冻结基线 s=24** | `freeze_image`，冻结视觉塔 + projector，`split=24` | **85.41 / 1493.32 / 318.93 / 74.10 / 71.43 / 63.71 / 65.34 / 39.14**，rc=0，182.5 min。与 vanilla：POPE +0.05pp、MME-P +3.5、MME-C −25.0、MMB-EN −0.03pp，其余 ±0.2pp。**冻结扫描 s=8/12/16/18/20/24 完成**：s=8 崩塌（POPE 74.92），**s≥12 全部在 vanilla 噪声内、对分割层不敏感**——深层图像 token 不参与更新不损精度 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s24/` | 97a32d6（代码） |
 | 同 `fastv_tf.yaml`，保留 25% 图 token，mean 打分；至此 2×2 网格（r=0.5/0.25 × last/mean）跑齐 | **70.38 / 1189.5 / 365.4 / 67.38 / 62.76 / 62.67 / 54.88 / 37.93**，rc=0，113.8 min。**全表（POPE / MME-P / SEED）**：vanilla 85.36 / 1489.8 / 65.45；r0.5-last 83.03 / 1462.3 / 64.50；r0.5-mean 78.88 / 1364.0 / 61.08；r0.25-last 79.36 / 1402.5 / 61.22；r0.25-mean **70.38 / 1189.5 / 54.88**。**两个因子单调可加：打分方式（last ≫ mean）影响大于保留比例（0.5 > 0.25）**；mean 在 r=0.25 时 recall 只剩 43.6%（precision 93.8%），模型几乎全答 "no"。结论：FastV 类方法里"在哪一层取注意力"是主要矛盾，与本方法用深层（饱和层附近）做分裂/融合的动机一致 | `results/revision/E3/`；`revision_runs/E3/fastv7b_k2_r0.{5,25}_{last,mean}/` | c0193ef（代码） |
| 2026-10-19 | **E4 冻结基线 s=20** | `freeze_image`，冻结视觉塔 + projector，`split=20` | **85.43 / 1472.4 / 336.1 / 74.17 / 71.24 / 63.61 / 65.40 / 38.93**，rc=0，180.0 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：POPE **+0.07pp**、MME-P −17.4、MME-C −7.9，其余五项逐项与 vanilla 相差 ≤0.04pp。s=12→20 全段平：**冻结视觉不损精度，且对分割层不敏感（s≥12）**；s=24 正在跑 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s20/` | 97a32d6（代码） |
| 2026-10-19 | **E3_baselines FastV k=2 r=0.25（last 打分）** | 同 `fastv_tf.yaml`，**保留 25% 图 token（576→144）**，last 打分 | **79.36 / 1402.5 / 365.7 / 73.23 / 69.81 / 64.85 / 61.22 / 38.98**，rc=0，112.8 min。相对 r=0.5 last（83.03 / 1462.3 / 335.7 / 74.08 / 71.01 / 64.80 / 64.50 / 38.93）：POPE −3.67pp（recall 60.67%，precision 96.88%）、MME-P −59.7、MMB-EN −0.85pp、SEED −3.28pp；MME-C +30。**砍到 1/4 后 POPE/SEED 掉得明显，但 SQA/BLINK/MMB-CN 几乎不动**——与"剪枝主要伤细粒度视觉判别、不伤文本主导任务"一致 | `results/revision/E3/`；`revision_runs/E3/fastv7b_k2_r0.25_last/` | a27e612（代码） |
| 2026-10-19 | **E4 冻结基线 s=18** | `freeze_image`，冻结视觉塔 + projector，`split=18`（与 LF 7B 主配置同层） | **85.38 / 1484.1 / 334.3 / 74.17 / 71.31 / 64.06 / 65.33 / 38.98**，rc=0，179.0 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）**逐项在噪声内**（POPE +0.02pp、MME-P −5.7、MME-C −9.6，MMB/SQA/SEED/BLINK ±0.45pp）。s=12/16/18 三行同结论：**在 s≥12 冻结视觉、让文本侧正常前向，精度不掉**——图像 token 的深层更新对最终精度不是必需 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s18/` | 97a32d6（代码） |
| 2026-10-19 | **E3_baselines FastV k=2 r=0.5（mean 打分）** | 同 `fastv_tf.yaml`，保留 50% 图 token，改按各层注意力均值打分 | **78.88 / 1364.0 / 368.9 / 71.56 / 69.16 / 63.96 / 61.08 / 38.45**，rc=0，138.2 min。比 `last` 打分低一大截：**POPE −4.15pp（78.88 vs 83.03）**、MME-P −98.3、MMB-EN −2.52pp、SQA −0.84pp、SEED −3.42pp；recall 62.18%（`last` 69.51%）、precision 93.36%。MME-C 反而 +33（368.9 vs 335.7）。**mean 打分把浅层（视觉细节还没汇聚）的注意力也算进去，丢掉的是信息量更大的 token**——与论文里"用深层打分"的动机一致 | `results/revision/E3/`；`revision_runs/E3/fastv7b_k2_r0.5_mean/` | 8bad590（代码） |
| 2026-10-19 | **E3_baselines FastV k=2 r=0.5（last 打分）** | `fastv_tf.yaml`，训练无关的 token 剪枝基线：保留 50% 图 token（576→288），按最后一层注意力打分；vanilla LLaVA-1.5-7B，七任务（无 MME 之外的额外项） | **83.03 / 1462.3 / 335.7 / 74.08 / 71.01 / 64.80 / 64.50 / 38.93**（POPE / MME-P / MME-C / MMB-EN / MMB-CN / SQA / SEED / BLINK），rc=0，138.1 min（比全量评测快约 25%）。**与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）比：POPE −2.33pp、MME-P −27.5、MME-C −8.2，其余五项 ±0.95pp 内且 SQA +1.19pp**。POPE 的下降全在 recall：precision 95.28% 而 recall 69.51%（`yes_ratio` 仍 0.50）——剪掉一半图 token 后模型更倾向答 "no"。作为"同等图 token 预算下训练无关剪枝"的对照行，精度损失远小于 vtw s=8，但与 DPVR-LF 的差距需在同一 token 预算下比较 | `results/revision/E3/`；`revision_runs/E3/fastv7b_k2_r0.5_last/` | 007fb39（代码） |
| 2026-10-19 | **E4 冻结基线 s=16** | `freeze_image`，冻结视觉塔 + projector，`split=16` | **85.37 / 1488.1 / 316.1 / 74.06 / 71.26 / 64.20 / 65.31 / 39.24**，rc=0，173.6 min。与 vanilla（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）**八项基本重合**：POPE +0.01pp、MME-P −1.7、MME-C −27.8、MMB 两项 ±0.07pp、SQA +0.59pp。冻结视觉在 s≥12 后不损精度（s12 −0.14pp → s16 +0.01pp），s8 的崩塌仅因分割过早 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s16/` | 见本次提交 |
| 2026-10-19 | **E4 冻结基线 s=12** | `freeze_image`，冻结视觉塔 + projector，`split=12` | **85.22 / 1484.1 / 324.6 / 74.22 / 70.94 / 63.86 / 65.46 / 38.77**，rc=0，174.1 min。分割层从 8 提到 12 后**基本追平 vanilla**（POPE −0.14pp、MME-P −5.7、MME-C −19.3、MMB-EN +0.09pp）；s=8 的崩塌（POPE 74.92）是分割过早，不是冻结视觉本身的问题。`s16` 已开跑 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s12/` | 见本次提交 |
| 2026-10-19 | **E12 中转 checkpoint 重测（5/6）** | hsf 中转来的 6 组 checkpoint，本机 lmms-eval 八项重测；代码 `d53c945`（checkpoint_io：`checkpoint-*/training_config.json` 查找 + `/nvme/lsy/...` 旧路径 rebase）、`97a32d6`（LoRA `logits_to_keep`） | 顺序 POPE / MME-P / MME-C / MMB-EN / MMB-CN / SQA / SEED / BLINK：`lora7b` 84.71 / 1484.7 / 337.9 / 72.60 / 70.18 / 62.12 / 65.51 / 40.19；`pc7b_s18` 85.19 / 1474.5 / 338.9 / 73.97 / 71.70 / 63.31 / 65.38 / 40.72；`pc13b_s28` 85.43 / 1537.7 / 287.5 / 76.81 / 73.74 / 68.52 / 67.69 / 41.08；`lf13b_s24` **86.03 / 1536.7 / 270.0 / 76.62 / 73.74 / 68.57 / 67.55 / 40.77**。`lf13b_s24` 跑了两遍（`use_cache=True/False` 两种缓存变体，仅验证加载路径不影响数字）**八项逐位一致**，说明 checkpoint_io 修复后的数值可复现。`fullft7b` **已跑完但结果无效**：rerun 后加载正常（686 tensor，missing=0），但八项 POPE 0.0 / MME-P 0 / MME-C 0 / MMB-EN 5.15 / MMB-CN 0.12 / SQA 8.03 / SEED 5.45 / BLINK 10.78。逐 tensor 检查发现该 checkpoint 的 `multi_modal_projector.linear_1/linear_2` 权重**全为 0**（min=max=0），LM 权重正常（std 0.017），即模型看不到图、只能凭文本作答。属中转 checkpoint 自身缺陷，见“问题” §17，**不计入 E12 有效结果（5/6）**。对照本机 vanilla7b（85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08）：**LF 13B 在 POPE、MME-P、MMB-EN、MMB-CN、SQA、SEED、BLINK 七项更高，MME-C 更低（270.0 vs 343.9）**；7B 各法基本持平，LoRA 略低。中转 checkpoint 的 7B token-diversion（pc7b）与本机训练口径可比 | `results/revision/E12/`；`revision_runs/E12/<job>/` | d53c945, 97a32d6 |
| 2026-10-19 | **E8 图像 token 数 × 饱和层** | LLaVA-1.5-7B / 13B，图 token 576 / 288 / 144 / 36（网格 24×24、12×24、12×12、6×6 双线性缩放），每格 500 样本 seed 0，测 cos / resid / 图文 KL 转折层 | **7B**：kl_img 转折层随图 token **576→288→144→36 = 21→21→22→24**（/32）；kl_txt 恒 22，resid_img/txt 恒 30。**13B**：kl_img **32→34→35→37**（/40）；kl_txt 25→25→26→27，resid 恒 38。**结论：饱和层随图像 token 减少而单调后移**（7B +3 层 / 13B +5 层），与"融合阶段需要处理的图文混合序列越长、饱和越晚"一致；resid 类指标对 token 数不敏感（恒 94~95% 深度） | `results/revision/E8/compare_{7b,13b}/compare.md`（含 8 个单格 run） | 见本次提交 |
| 2026-10-20 | **E6 Qwen2.5-VL-7B DPVR-LF s=22 八项评测** | 4 卡全量训练（split=22、k=1、vision_depth 1、freeze all_but_single，有效 batch 4）后的八项评测 | **89.00 / 1654.35 / 585.36 / 87.50 / 86.05 / 82.50 / 77.11 / 49.03**，rc=0，443.6 min。与 Qwen vanilla（87.44 / 1690.0 / 638.2 / 87.46 / 86.12 / 83.09 / 77.43 / 51.24）：POPE **+1.56pp**、MME-P −35.7、MME-C **−52.8**、MMB-EN +0.04pp、MMB-CN −0.07pp、SQA −0.59pp、SEED −0.32pp、BLINK **−2.21pp**。**除 BLINK 外全部落在噪声内，主结论在 Qwen 架构上同样成立**；BLINK 是唯一一致的系统性低点（LLaVA-7B 上 seed1/2 也各 −0.78/−1.21pp），写论文时应作为跨架构的已知代价单独说明，不用挑 seed 掩盖 | `results/revision/E6/eval/`；`revision_runs/E6/eval/` | 6b6065b（代码 35b5499） |
| 2026-10-19 | **E6 Qwen2.5-VL-7B vanilla 基线** | `vanilla_eval`，lmms-eval 八项，代码 `97a32d6` | **87.44 / 1690.0 / 638.2 / 87.46 / 86.12 / 83.09 / 77.43 / 51.24**（POPE / MME-P / MME-C / MMB-EN / MMB-CN / SQA / SEED / BLINK），rc=0，27487.5 s。**这是 Qwen2.5-VL-7B 在本机链路上的新基线**，E6 训练后的对照行 | `revision_runs/E6/vanilla_eval/` | 97a32d6 |
| 2026-10-19 | **E3_vtw（部分，3 行）** | vanilla 7B 与 VTW 图像 token 合并，`split=8/12`，其余同 LF 配置 | **vanilla7b** 85.36 / 1489.8 / 343.9 / 74.13 / 71.26 / 63.61 / 65.45 / 39.08（与 E0 逐位吻合，链路可复现）；**vtw7b_s12** 81.33 / 1436.2 / 313.9 / 73.76 / 70.46 / 63.71 / 64.32 / 37.82（比 vanilla 低约 4pp，仍可用）；**vtw7b_s8** 51.49 / 788.0 / 358.6 / 55.72 / 48.19 / 62.32 / 45.87 / 37.30（**分割层过早，模型严重退化**）。`s16` 跑到 47%（1h15m，八项进度条 19818/41940）时被主动中止以腾卡（见"问题" §16），`s18/s20/s24` 与 13B 待续 | `results/revision/E3/`；`revision_runs/E3/<job>/` | 见本次提交 |
| 2026-10-19 | **E4 冻结基线（部分，1 行）** | 冻结视觉塔 + projector 的 `freeze_image`，`split=8` | **freeze7b_s8** 74.92 / 1334.8 / 305.7 / 71.68 / 68.01 / 63.16 / 62.78 / 38.56，POPE 比 vanilla 低 10.4pp、MME-P 低 155 —— 可作为"只训浅层/冻结视觉"的下限行。`s12` 正在跑，`s16…s28`、`nobranch` 待续 | `results/revision/E4/`；`revision_runs/E4/freeze7b_s8/` | 见本次提交 |
| 2026-10-18 | **E6 前置关卡（冒烟通过）** | Qwen2.5-VL-7B，`split=22`（E2 结论）；`smoke_train` 20 step / subset 0.001，`smoke_eval` pope + scienceqa limit 20；代码 `97a32d6`（含 checkpoint_io / LoRA generate 两处修复） | **两道关卡全过**：`smoke_train` 20/20 step，rc=0，140.2 s，trainable 233.1M / 8.53B（2.73%）；`smoke_eval` pope_accuracy **0.80**、scienceqa **0.85**（±0.082），rc=0。`vanilla_eval`（Qwen2.5-VL-7B 八项本机基线）与全量训练待跑；全量训练按红线 2 用 4 卡 × batch 1，当前等 4 张空卡 | `revision_runs/E6/{smoke_train,smoke_eval}/`（status.json / summary.json） | 97a32d6 |
| 2026-10-18 | **E1 生成延迟** | 10 个任务，batch 1，576 图 token + 128 文 token，bf16，greedy，新生成 64 / 256 token，KV cache 开/关两套；卡型 **H20-3e** | **7B**：TTFT vanilla 95.4 → LF s=18 **72.8 ms（1.31×）**。无 cache 解码（ms/token）：L64 vanilla 94.3 → LF **73.8（1.28×）**，L256 105.7 → **85.4（1.24×）**。**开 cache** 后两者几乎持平、LF 略慢：L64 14.9 → 15.4，L256 14.9 → 15.2。**13B**：TTFT 170.6 → LF s=24 **129.5（1.32×）**；无 cache L64 172.6 → **133.8（1.29×）**，L256 194.9 → **155.6（1.25×）**；开 cache L64 18.1 → 18.9（LF 慢 4%），s=28 同理。**结论：论文的加速来自省掉深层的整段前向/解码，在 cache 解码（每步只算新 token）下这部分本来就被摊薄，所以优势只出现在 prefill/TTFT 与无 cache 解码**——Table 7 的协议（`use_cache=False`）要如实写明，不能暗示开 cache 也更快 | `results/revision/E1/` | 见本次提交 |
| 2026-10-18 | **E5 prefill 效率** | 解析 FLOPs（与实测无关）+ B=1 / B=4 前向延迟，batch=1 时 n_img 576、n_txt 128、bf16、`use_cache=False`、warmup 10 / iters 100；卡型 **H20-3e** | **实测（B=1，ms）**：7B vanilla 94.36±0.05 → LF s18 K=1 **70.78（1.33×）**、K=2 72.98（1.29×）。**B=4**：vanilla 346.14 → K=1 249.10（1.39×）、K=2 257.04（1.35×）。**13B（B=1）**：vanilla 170.17 → s24 **127.67（1.33×）**、s28 140.10（1.21×）、s34 159.11（1.07×）。**解析 FLOPs（total / LM / 深层栈）**：7B s18 K=1 **−29.0% / −30.9% / −70.5%**，K=2 −26.6%；13B s24 **−27.8% / −28.8% / −71.9%**。峰值显存：7B vanilla 13.26 G / LF 13.64 G，13B vanilla 25.11 G / LF 25.71 G。**结论：分割层越靠后省得越少（13B s34 只剩 7.9% FLOPs、1.07× 实测），实测加速与解析 FLOPs 的 total 降幅基本一致** | `results/revision/E5/`（含 `analytic_flops/analytic_flops.md`） | 见本次提交 |
| 2026-10-18 | **E2 饱和分析** | 5 个模型 × 500 张 LLaVA-665k 样本（seed 0），训练无关，逐层测 cos / resid / 图文注意力 / logit-lens KL | 转折层 = 曲线升到峰值 50% 后再跌回峰值 50% 的那一层（括号内为 / 总层数）：**LLaVA-1.5-7B** kl_img 21(.66) kl_txt 22(.69) resid 30/30(.94)；**LLaVA-1.5-13B** kl_img 32(.80) kl_txt 25(.62) resid 38(.95)；**LLaVA-1.6-Mistral-7B** kl_img 29(.91) kl_txt 26(.81) resid_img 1；**Qwen2.5-VL-7B** kl_img 25(.89) kl_txt 26(.93) resid_img 2、resid_txt 26；**InternVL3-8B** kl_img 25(.89) kl_txt 23(.82) resid_img 3。图像 token 数（500 样本均值）：LLaVA 系 576 / 576，Mistral-1.6 2115.8，Qwen 321.0，InternVL3 2377.7。**结论：Qwen2.5-VL-7B 的 kl_img 转折层 25/28，per-sample IQR [24,25]，kl_txt 26/28、IQR [26,26]——判定明确，不需要跑 `E6_split.yaml`。** 按论文两个 LLaVA 的「分割层 / 转折层」比例（7B 18/21=0.86，13B 28/32=0.88）折算，**E6 取 `split=22`**（0.88×25≈22；若直接用 18/32 的比例则是 16/28，正好是 config 里的默认值，但 E2 显示它的饱和点比 LLaVA-7B 更靠后，所以仍上调到 22）。cos_img 的 tau=0.95 平台在所有模型上都判为 null（曲线单调升不到 0.95），该指标本批不可用，只作记录 | `results/revision/E2/`（5 个模型 + `compare/`） | 见本次提交（代码 `d53c945` 前的 E2 不受两处修复影响：E2 只读 vanilla HF 模型，不经过 checkpoint_io / LoRA） |
| 2026-10-16 | E0 验收 | `reeval_vanilla7b`（vanilla 7B 八基准重测，**首次跑通，本机新基线**） | **2026-10-17 作者决定：本行即新的 vanilla 基线，E0 视为通过**（论文旧结果未保存，不再作为验收目标；详见"问题" §11）。与论文 vanilla 行的差只作记录。实测：POPE .8536 / MME-P 1495.46 / MME-C 328.93 / MMB-EN .7413 / MMB-CN .7138 / SQA .6406 / SEED(`seed_image`) .6540 / BLINK .3898。论文行（`paper/ms.tex:900`）：.857 / 1457 / 318 / .737 / .693 / .627 / .641 / .385。差：**−0.34pp / +38.5 / +10.9 / +0.43pp / +2.08pp / +1.36pp / +1.30pp / +0.48pp**，门槛是 <0.3pp，**八项全在外**，MMB-CN 已过 §7 的 2pp 线。详见"问题" §11 | `results/revision/E0/reeval_vanilla7b/`（`summary.json`、`log_tail.txt`、`lmms/` 样本） | 见本次提交 |
| 2026-10-14 | E0 验收 | `provenance_7b`（训练起点核查） | **verdict: language_model=llava_hf, projector=llava_hf**。11 个探针权重与 `llava-1.5-7b-hf` 的距离全为 **0.0**，与 vicuna / stage1_projector 参照均为非零（4e-3 ~ 1.3e-2）。即该 checkpoint 的 LM 与 projector 都等于 llava-1.5-7b-hf 原版 | `results/revision/E0/provenance.json` | 014ea25 |
| 2026-10-14 | E0 验收 | `latency_vanilla7b` | **28.65 ± 0.40 ms**（p50 28.56），peak 13.26 GB。协议 batch=1, n_img=576, n_txt=128, bf16, use_cache=False, warmup=10, iters=100。卡型 **H100 80GB HBM3** | `results/revision/E0/latency_vanilla7b.json` | 014ea25 |
| 2026-10-14 | E0 验收 | `latency_lf7b` | **25.48 ± 0.93 ms**（p50 24.97），peak 13.64 GB。同上协议；method=token_diversion_x3_fusion, split_layer=18, vision_depth=1, freeze=all_but_single。**相对 vanilla 加速 1.124×** | `results/revision/E0/latency_lf7b.json` | 014ea25 |
| 2026-10-14 | E0 验收 | `smoke_train_lf7b`（50 step 冒烟） | **跑通**：50/50 step，train_loss 2.642，train_runtime 14.4 s（3.47 step/s）。无 NaN（loss 区间 1.05~7.17，grad_norm 0.22~1.27） | `results/revision/E0/smoke_train_lf7b.json`（status.json / log.txt 在 `revision_runs/E0/smoke_train_lf7b/`） | 014ea25 |
| 2026-10-13 | E0 验收 | `download_assets.py --verify` | 7/7 benchmark 样本数与论文一致 | `results/revision/E0/assets.json` | 4b4f56a |
| 2026-10-13 | E0 验收 | `check_env.py` | 软件栈/路径/8 卡可用性全过；模型与 665k 图像在跑这条时尚未下载完 | `results/revision/E0/env_check.json` | 4b4f56a |

> 延迟两项与 `latency_vanilla7b`/`latency_lf7b` 的 `latency.json` 里都写了 `"gpu": "NVIDIA H100 80GB HBM3"`，Table 7 引用时按 H100 标注。
> **E0 六项 2026-10-16 已全部 done**（`reeval_vanilla7b` 于 18:05 跑完，rc=0，耗时 4834.8 s）。
> **2026-10-17 作者决定：验收改为以本机重测为准。** 论文旧结果未保存，原「八项与论文 vanilla 行差 <0.3pp」
> 的门槛取消；`reeval_vanilla7b` 的八项（POPE .8536 / MME-P 1495.46 / MME-C 328.93 / MMB-EN .7413 /
> MMB-CN .7138 / SQA .6406 / SEED .6540 / BLINK .3898，见上表）即**新的 vanilla LLaVA-1.5-7B 基线**，
> **E0 视为通过**，进入阶段 2。与论文的差异只记录、不阻塞（分析见"问题" §11）。
> 大修所有表格的行都必须用本机、本仓库的评测链路重测，不从论文抄数字。

> **进行中（2026-10-18）**：自主连续运行第 3 步开始。本机 8×H20-3e 全空闲，直接在容器内跑，
> 不再提 AFO 作业。按推荐顺序先跑 **E1 + E5**（延迟/生成表，只占 GPU 0，其余卡空着）。
> 环境沿用 `worker.sh` 的导出（conda 前缀环境、`TMPDIR=/workdir/dpvr_tmp`、HF 缓存上盘、
> shim 里的 `device_map` 补丁），代码用 `server-fixes` 分支（worktree `../dpvr-fixes`）。
> E12 的 `--set hsf=` 指向 `${CKPT_ROOT}/legacy`——中转来的 checkpoint 实际落在 `legacy/` 而不是
> 文档假定的 `hsf/`（与 `worker.sh` 里 E0 起点核查的处理一致）。

## 问题

### 1. 卡型与文档不符：申请 H800，拿到的是 8×H100-80G（中途还有一版 H20-3e）

`AGENTS.md` 全文以及 `SERVER_HANDOFF` 都按 8×H800 写。实际拿到的卡型换过两次：

| 时间 | appid | `nvidia-smi` 实测 | 备注 |
| --- | --- | --- | --- |
| 2026-10-13 | `psxmf2vxrs643dy3` | 8 × NVIDIA H20-3e（139.7 GB/卡，sm_90） | 最早那个占卡作业 |
| 2026-10-14 12:03 | `psxncscxrs7fieen` | **8 × NVIDIA H100 80GB HBM3**（81559 MiB，700 W，sm_90） | 当前 E0 用的卡 |

两次申请的都是同一个键 `worker.gcoresh800-80g = 8`，队列 `root.hldy_training_cluster.hadoop-aipnlp.h800_vl_sp`—— **队列名里的 `h800` 不保证发 H800**，平台按当时余量发什么是什么。

- 影响面：E1/E5 的延迟表（Table 7）是**论文里唯一强依赖卡型的实验**。H100 与 H800 同代、同 HBM3、显存带宽量级接近（H800 是 H100 的带宽降级版），比 H20 好得多，但仍**不能声称是 H800 上的数字**。其余训练/精度实验跨卡可比，但 RESULTS.md 记录时必须写明卡型。
- 作者已确认：**H100 也行**，这两项按 H100 如实标注。
- 复现：`grep -n 'nvidia-smi -L' -A 10 dpvr/logs/psxncscxrs7fieen-20261014-120319.dpvr.log`

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

### 4. E0 尚未真正跑完（**2026-10-16 已作废，见下**）

> ⚠️ **本条已过时，保留只为记录当时的判断。** 当时 `results/revision/E0/` 只有环境与资产核对产物，
> `provenance.json` / `reeval_vanilla7b` / 延迟两项 / `smoke_train_lf7b` 全都还没有产物。

**2026-10-16 实际状态：六项全部 `done`、`returncode=0`。**

| 任务 | 状态 | 耗时 |
| --- | --- | --- |
| `assets_verify` | done | — |
| `provenance_7b` | done | — |
| `reeval_vanilla7b` | done | 4834.8 s |
| `latency_vanilla7b` | done | — |
| `latency_lf7b` | done | — |
| `smoke_train_lf7b` | done | — |

**关于 665k 图像：当日「缺项」的判断也是错的，是查错了目录。** 当时脚本与文档假定的根是
`$DATA_ROOT`，而实际落在 `$R/dpvr/datasets/llava-1.5-665k-instructions/`。按正确根复核，六项**全齐、无一缺**：

| 子集 | 实测张数 |
| --- | --- |
| `coco/train2017` | 118287 |
| `gqa/images` | 148854 |
| `ocr_vqa/images` | 166022 |
| `textvqa/train_images` | 25119 |
| `vg/VG_100K` | 64346 |
| `vg/VG_100K_2` | 43903 |

复核命令：

```bash
B=<R>/dpvr/datasets/llava-1.5-665k-instructions
for d in coco/train2017 gqa/images ocr_vqa/images textvqa/train_images vg/VG_100K vg/VG_100K_2; do
  printf '%-24s %s\n' "$d" "$(ls $B/$d | wc -l)"
done
```

**但 E0 的验收仍未过**——不是资产问题，是 §1.5 的精度门槛，见 §11。

### 5. `.env` 缺 `HF_CKPT_REPO`

`SERVER_PROMPTS.md` 启动前第 4 步要求 `.env` 里有 `HF_CKPT_REPO`，当前没有。checkpoint 实际落在 `${CKPT_ROOT}/legacy/` 而不是文档假定的 `${CKPT_ROOT}/hsf/`，E0 启动时 `--set prov_ckpt=` 要按实际路径传。

### 6. 默认镜像 glibc 2.17 太老，作业在 [1/4] 就死（已换镜像绕过）

`SERVER_HANDOFF` 给的默认镜像 `scale-llm/training_pytorch2.5.0_cuda12.4.0_..._6cd026ac:v1.0.0` 是 **CentOS 7 原版 glibc 2.17**（符号表里最低缺 `GLIBC_2.18`）。而本环境的 `conda_env_dpvr` 的 python 来自 **anaconda `pkgs/main`**（不是 conda-forge），且 `torch==2.9.0` 是 **`manylinux_2_28`** wheel —— 两者都要求 **GLIBC_2.28**。

现象：作业活 19 分钟，卡在 [1/4]，日志里

```
/usr/lib64/libc.so.6: version `GLIBC_2.28' not found (required by .../conda_env_dpvr/bin/python)
（另有 GLIBC_2.25/2.26/2.27 来自 python，GLIBC_2.18/2.25/2.26 来自 libstdc++.so.6）
!! torch 探测失败
EXIT rc=1
```

- 复现：`dpvr/logs/psxnadlxrs7fieen-20261014-105649.dpvr.log`
- 绕过办法（**未改任何仓库代码**，只改我自己的 `workspace/dpvr_job/dpvr_e0.hope`，不在红线路径内）：换成 **CentOS 7 壳 + 换过的 glibc 2.28** 的镜像
  ```
  afo.docker.image.name = registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.phxmlp.mtjupyter.singleuser/aipnlp_base_pytorch2.3.1_cuda12.4.0_python3.9_1_18_glibc_2.28_2532322d
  ```
  （`/usr/lib64/libc.so.6` 符号表最高有 `GLIBC_2.28`，libstdc++ 到 `GLIBCXX_3.4.25`。）换镜像是拿 8 卡前先用**单卡探针**验的：`workspace/dpvr_job/probe_glibc.{hope,sh}`，appid `psxo4s4xrs643dy3`，同队列同环境下 `Python 3.10.22 rc=0`、`torch 2.9.0+cu128 cuda 12.8 avail True rc=0`。
- 换后效果：`[1/4]` torch OK → `[2/4] 0 required check(s) failed` → `[3/4] 172 passed in 266.36s` → `[4/4]` 队列起。
- 建议（需作者改文档）：`SERVER_HANDOFF` 里把默认镜像换成这个 glibc 2.28 的，或把环境改成 conda-forge + `manylinux_2_17` wheel，否则每个新容器都要再撞一次。

### 7. E0 六个子任务整齐地 10 秒全灭：宿主机侧没 `conda activate`（已绕过）

`jobs/E*.yaml` 里的命令写成裸 `python`（这是有意的，`SERVER_HANDOFF:26-27` 假定先 `bash scripts/revision/setup_env.sh` 再 `conda activate dpvr`），而 `run_queue.py` 只把 `os.environ` 原样传给子进程、不解析解释器路径。我的入口脚本 `worker.sh` 起初没有等价于 activate 的那一步，于是子进程的裸 `python` 落到系统 **Python 2.7**，六个任务**整齐地 10.0–10.6 秒全灭**（2026-10-14 12:11 那轮）。

```
[queue] provenance_7b:    failed (rc=1, 10.6s)
[queue] assets_verify:    failed (rc=1, 10.6s)
[queue] reeval_vanilla7b: failed (rc=1, 10.5s)
[queue] latency_vanilla7b: failed (rc=1, 10.2s)
[queue] smoke_train_lf7b: failed (rc=1, 10.0s)
[queue] latency_lf7b:     blocked (a dependency failed)
```

子任务日志（`${CKPT_ROOT}/revision_runs/E0/<task>/log.txt`）全是 py2 在解析 py3 代码：

```
check_weight_provenance.py:2  SyntaxError: Non-ASCII character '\xc2' ... no encoding declared
download_assets.py:12         SyntaxError: Non-ASCII character '\xc2' ...
eval_lmms.py:57               def parse_tasks(spec: str) -> list[str]:  SyntaxError: invalid syntax
bench_latency.py:51           def make_inputs(config, batch: int, n_txt: int, device):  SyntaxError
train.py:9                    SyntaxError: Non-ASCII character '\xe2' ...
```

- 复现：`cat dpvr/logs/psxncscxrs7fieen-20261014-120319.dpvr.log`，或直接看上面五个 `log.txt`。
- 绕过办法（**未改任何仓库代码**，同样只改我自己的 `worker.sh`）：在做任何事之前把环境的 bin 排到 `PATH` 最前，等价于 `conda activate dpvr` 且不依赖 conda 的 shell function。
  ```bash
  export PATH="$ENV_PREFIX/bin:$PATH"
  export VIRTUAL_ENV="$CONDA_PREFIX=$ENV_PREFIX"
  unset PYTHONHOME PYTHONPATH
  ```
  并在日志第一段加了一行自证：`裸 python -> <路径> : <版本>`，正常必须打 `Python 3.10.22`，打 `2.7.*` 就说明这条没生效。
- 建议（需作者改文档）：`worker.sh` 属于服务器侧一次性脚本、不进仓库，所以这条更适合写进 `SERVER_HANDOFF`——提醒后来者「容器入口脚本必须自己等价 activate，队列命令里的裸 `python` 不能指望子进程解析」；或者让 `run_queue.py` 用 `sys.executable` 解析 `python`。

### 8. 【挡住所有精度实验】`dpvr.lmms_plugin` 缺 `tasks` 子包，lmms-eval CLI 直接抛 AttributeError

`reeval_vanilla7b` 挂在这里，**根因在仓库侧，不是环境**。所有走 `eval_lmms.py` 的精度实验（E0 回评、E1/E2/E3/E9 全部）都会撞同一堵墙。

**现象**（`${CKPT_ROOT}/revision_runs/E0/reeval_vanilla7b/log.txt`）：

```
File ".../lmms_eval/__main__.py", line 435, in cli_evaluate_single
    package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
AttributeError: 'NoneType' object has no attribute 'submodule_search_locations'
```

**根因**：`scripts/revision/eval_lmms.py:230` 无条件把 `dpvr.lmms_plugin` 加进 `LMMS_EVAL_PLUGINS`；而 lmms-eval 0.5.0 的 CLI 对插件列表里**每一项**都要求存在 `<plugin>.tasks` 子包（它把该目录追加进 `--include_path`）。仓库的 `src/dpvr/lmms_plugin/` 下只有 `models/`（`__init__.py` + `models/__init__.py` + `models/dpvr_hf.py` + `models/dpvr_qwen25vl.py`），没有 `tasks/`，于是 `find_spec("dpvr.lmms_plugin.tasks")` 返回 `None` → 解引用 `.submodule_search_locations` 抛异常。

**上游是同一份代码，不是环境被改坏**：
- 装了上游 `v0.5` 的 `lmms_eval/__main__.py` 对下来，第 432–436 行**逐字相同**（`https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/v0.5/lmms_eval/__main__.py`）；`direct_url.json` 为空，即 pip 正常安装、未被本地改动。
- 也就是说：lmms-eval 的插件协议是「插件包必须同时提供 `models` 和 `tasks` 两个子包」——`src/dpvr/lmms_plugin/__init__.py` 的 docstring 只提了 `models`，少实现了 `tasks`。

**为什么单测没拦住**：`tests/test_revision_eval.py:89 test_lmms_plugin_registers_dpvr_hf` 走的是**库内路径** `from lmms_eval.models import get_model` → `lmms_eval/models/__init__.py:126` 只做 `importlib.import_module(f"{plugin}.models")`，**不碰 `__main__.py`**。172 个测试全过，但 CLI 这条路没被覆盖。

**复现（秒级，不加载任何模型）**：

```bash
cd <repo>
LMMS_EVAL_PLUGINS=dpvr.lmms_plugin PYTHONPATH=<repo>/src \
  <env>/bin/python -m lmms_eval --tasks list
# → AttributeError: 'NoneType' object has no attribute 'submodule_search_locations'
# 反证：去掉 LMMS_EVAL_PLUGINS 前缀即正常列出任务
```

**为什么我没法在服务器侧绕过**（三条路都要动红线内文件，故按 AGENTS.md 第 1 条停在这一项）：
1. 让插件包解析到别的目录 —— `eval_lmms.py:231` 自己把 `PYTHONPATH` 写成 `<repo>/src` **在最前**，我预先设的环境变量会被它压掉；
2. 给 `src/dpvr/lmms_plugin/` 补一个空的 `tasks/__init__.py` —— 改 `src/`，红线禁止；
3. 改 `eval_lmms.py` 让它在找不到 `tasks` 时跳过 —— 改 `scripts/`，红线禁止。

**建议的修法（请云端提 PR，作者定夺）**，任选其一：
- 在 `src/dpvr/lmms_plugin/` 下补 `tasks/__init__.py`（最简单，插件的任务是 `--include_path` 另外给的，空目录即可）；
- 或改 `scripts/revision/eval_lmms.py:229-231`，只在确实有 `tasks` 子包时才把插件追加进 `LMMS_EVAL_PLUGINS`；
- 或把模型注册改成不走 `LMMS_EVAL_PLUGINS`（`lmms_eval.models` 的注册表直接注入）。

**影响面**：E0 的 `reeval_vanilla7b`（vanilla 7B 八基准基线）拿不到；阶段 3/4 的 E1/E2/E3/E9 全部走 `eval_lmms.py`，会一起挂。**这是当前最该先修的一项。**

**邮件**：按 AGENTS.md 第 7 条应发主题 `BLOCKED: E0 reeval_vanilla7b 插件缺 tasks 子包`，但 `.env` 里 SMTP 配置为空（见 §9），**邮件发不出去**。

#### 8.1 补记（2026-10-16）：本条已在服务器侧绕过，根因仍未修

作者 2026-10-16 批准在**红线外**做运行时绕行，于是写了仓库外的 shim
（`workspace/dpvr_job/shim/sitecustomize.py` + 空的 `empty_tasks_stub/`，
由 `worker.sh` 在 `unset PYTHONPATH` **之后** `export PYTHONPATH=<shim>` 挂上，
`eval_lmms.py:231` 会把 `<repo>/src` 前置、原样保留这个尾巴，所以 shim 能穿到子进程）。

修法：在 `sys.meta_path` 最前插一个 finder，**只截 `dpvr.lmms_plugin.tasks` 这一个名字**，
返回 search location 指向真实存在的空目录的 spec，其余名字一律 `return None` 交还 `PathFinder`。

为什么对评测口径零影响：`lmms_eval/__main__.py:391` 已用原始 `include_path` 建好 `TaskManager`，
433–436 那段在其**之后**才改 `args.include_path`、改完无人再读，是死代码。

已双向验证：无 shim 复现（`log.txt` 里完整栈，`AttributeError: 'NoneType' object has no attribute
'submodule_search_locations'`）；有 shim 时同一命令 rc=0、列出 1217 行任务、零异常。

⚠️ **两条正解仍请云端提 PR**（服务器侧只是运行时绕行，根因一字未动）：
1. 在 `src/dpvr/lmms_plugin/` 下补 `tasks/__init__.py`；或
2. 给 `scripts/revision/eval_lmms.py:229-231` 加 `find_spec(...) is not None` 守卫。

**订正**：本条原先写「三条路都要动红线内文件」。**第 1 条是错的** —— 让插件解析到别的目录确实不需要改红线内文件（shim 已证明可行，`PYTHONPATH` 被前置也没关系，只要不靠 `sys.path` 顺序、而是靠 `meta_path` 拦截即可）。

### 9. `.env` 的 SMTP 配置只有端口，其余全空 → `send_results.py` 发不出任何邮件

AGENTS.md 要求「E0 验收结束、每个实验结束、任何『异常就停』、全部结束」各发一封邮件，但本机上邮件通道从头就是断的。

**现象**（`.env`，只报长度、不落值）：

| 键 | 值长度 | 说明 |
| --- | --- | --- |
| `SMTP_HOST` | 0 | 空 |
| `SMTP_PORT` | 3 | 有值（形如 `465`） |
| `SMTP_USER` | 0 | 空 |
| `SMTP_PASSWORD` | 0 | 空（授权码没给） |
| `MAIL_TO` | 0 | 空（收件地址没给） |

**后果**：`scripts/revision/send_results.py` 起不来——主机、账号、授权码、收件人四者缺一不可。所以：

- §8 那条 `BLOCKED:` 邮件**没发出去**；
- E0 验收结束的那封邮件**没发出去**；
- 后续每个实验结束的邮件也不会发。

**我按 AGENTS.md 的处理**：「邮件发送失败时不要换别的途径，记进 RESULTS.md『问题』后继续」——所以没有改用别的通道（不发 IM、不写别处），只在这里记录，并由每次汇报直接告知作者。**请作者补 `.env` 的这四个键**，否则整条邮件链路等于没有，所有「异常就停」的通知都只能靠 RESULTS.md 和汇报。

**日志路径**：无独立日志（脚本在读取 `.env` 阶段就因缺配置退出）。复核命令：

```bash
awk -F= '/^(SMTP|MAIL)/{k=$1; v=substr($0, index($0,"=")+1); gsub(/^[ \t]+|[ \t]+$/,"",v); printf "%s: 长度=%d\n", k, length(v)}' .env
```

### 10. 【凭据泄露，已处理】lmms-eval 把 HF token 原文写进日志，被 GitHub push protection 拦下

第一次提交 E0 结果时 `git push` 被拒：

```
remote: error: GH013: Repository rule violations found for refs/heads/main.
remote:   —— Hugging Face User Access Token —
remote:        - commit: d19023e... path: results/revision/E0/reeval_vanilla7b/log_tail.txt:14
remote:        - commit: d19023e... path: results/revision/E0/reeval_vanilla7b/log_tail.txt:31
```

**链路**：`eval_lmms.py` 通过 `--token` 把 HF token 传给 lmms-eval，而 lmms-eval 在
`__main__.py:400` 把整个 evaluation tracker 的构造参数**原样 INFO 打出来**：

```
Evaluation tracker args: {'output_path': '.../lmms', 'token': 'hf_...'}
```

`collect_results.py` 又把该任务的 `log.txt` / `log_tail.txt` 拷进 `results/revision/E0/<任务>/`，
于是 token 进了待提交目录。

**处理**（本次已做，未污染远端）：本地 `sed` 把 `hf_[A-Za-z0-9]{20,}` 替换成 `hf_**REDACTED**`，
`git commit --amend` 后重推成功（`014ea25..e13be9a`）。被拦的 `d19023e` 从未到达远端。

**每次提交前的硬性检查**（已写进巡检 cron）：

```bash
grep -rn 'hf_[A-Za-z0-9]\{20,\}' results/ docs/    # 必须无输出
```

**给作者的建议（影响所有精度实验）**：`eval_lmms.py` 调 lmms-eval 时把 token 放进命令行参数，
日志必然带出来。建议改成走环境变量（`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`）而不是 `--token`，
或者在 `collect_results.py` 拷日志前统一脱敏。**在改掉之前，每次提交前都要手工 grep 一遍。**

**另注**：该 token 已不可信（写进了本地日志、也被 GitHub 存档在拦截记录里），
**建议作者轮换一次 HF token**。

#### 10.1 补记（2026-10-16）：这条「已处理」是错的，机制还在，而且比 §10 更危险

§10 说「已处理」，那指的是 **那一次提交**处理干净了。这次重新跑完 E0、重跑 `collect_results.py`
之后我把整条链路拆开重算了一遍，发现 **§10 描述的现象当时判断错了，而真正的漏洞没堵**。

**先把当时的判断订正掉。** §10 现在隐含的说法是「token 在 `log.txt` 深处，只是恰好落进了
被拷走的那 200 行」。**这是错的** —— token 根本不在深处，就在**第 14 行**开始的头几行。
实测（本次 `reeval_vanilla7b` 的 `log.txt`，1,967,396 字节）：

| 量 | 值 |
| --- | --- |
| `log.txt` 总行数（Python `splitlines()`） | **23482** |
| 其中 `\r` 字符个数 | **0**（这份日志是干净的 `\n` 行，不存在 CR 膨胀） |
| token 所在行号 | **14 / 31 / 48 / 65 / 106 / 147 / 188** |
| `log_tail.txt` 取的窗口 | 第 **23283** – 23482 行（= `splitlines()[-200:]`） |
| 窗口与 token 的距离 | **相隔 23000+ 行** |

也就是说：`collect_results.py` 那句 `splitlines()[-200:]` 拷的是**日志尾巴**，
而 token 在**日志开头**。两者差了整整两万三千行。**这不是「侥幸擦肩」，是这两个位置在正常
情况下**永远不会**重合** —— 只要日志是「开头打一次配置、中间刷几万行进度、结尾打一次汇总」
这个形状（lmms-eval 的 17990 条 SEED 样本必然刷出这个形状），尾窗口就永远够不到 token。

**所以真正的结论是反过来的：§10 当年的拦截不是「运气好躲过了」，而是那次评测的日志形状
恰好在某个中间状态被抓拍**，而它之所以能被拦截，是因为**它当时确实在窗口里**。
换句话说 —— **能不能拦住 token，取决于日志刷了多少行进度，而不取决于任何一段代码**。
这是个纯粹的巧合依赖，没有任何机制保证。

**现在的风险点**：只要有一次评测的生成阶段刷的行数少于 200（`--limit` 冒烟、任务在早期就
rc≠0 退出、某个 benchmark 的进度条走得慢），`splitlines()[-200:]` 就会一路前移到第 14 行，
`log_tail.txt` **直接带着 token 进 `results/revision/`**。`--limit 20` 的快速冒烟评测
（`eval_lmms.py` 支持、文档里也推荐先跑）是**最可能踩中的那个**：样本少、行数少、尾巴短，
窗口几乎必然覆盖开头。

**三个事实叠起来才是完整的**：
1. `collect_results.py` 对 `log_tail.txt` **没有任何脱敏**（读进来、`splitlines()[-200:]`、
   写出去，全程原样）；
2. 仓库里**没有 `.gitattributes`、没有 git filter**，本地没有兜底；
3. 唯一的防线是提交前手工 `grep`（§10 那条命令），**依赖人去执行**。

**服务器侧的实际做法**：我每次 `git add` 前都跑一遍 §10 的 grep，本次跑完是**空的**——
but 如上所述，这次为空是**因为窗口位置恰好够远**，**不代表这条 grep 有本事在下次也拦住**。
提交这一步我会照旧走 grep，但把它记为「人工兜底，非机制」。

**建议（请云端提 PR，两条都改最稳）**：
1. `collect_results.py` 在写 `log_tail.txt` 之前统一脱敏：
   `re.sub(r'hf_[A-Za-z0-9]{20,}', 'hf_**REDACTED**', text)` —— 一行，覆盖所有形状；
2. `eval_lmms.py` 别把 token 放进 `--token` 命令行参数，改走环境变量
   （`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`），从源头上不让它进日志。

在 1 或 2 落地之前，**每次提交都必须跑一遍 grep，不能省**。

#### 10.2 顺带发现：`collect_results.py` 会让 `results/` 里的 `log.txt` **永久停在上一次失败的那份**

同一个函数里的另一个 bug，本次实测撞上了。

`copy_small()` 的循环有个大小过滤：

```python
if f.suffix not in SMALL_SUFFIXES or f.stat().st_size > max_kb * 1024:
    continue      # 超过 --max_kb（默认 512 KB）就整个跳过
```

而 `log.txt` 是**无条件**单独写的：

```python
log = src_dir / "log.txt"
if log.exists():
    lines = log.read_text(errors="replace").splitlines()[-200:]
    (dst_dir / "log_tail.txt").write_text("\n".join(lines) + "\n")
```

于是结果目录里出现**两个日志文件、语义完全相反**：

| 文件 | 谁写的 | 本次实际内容 |
| --- | --- | --- |
| `log.txt` | `copy_small()` 的循环，**受 512 KB 限制** | **旧文件**：2026-10-14 那次失败运行的（6,332 字节，末尾是 `'NoneType' object has no attribute 'submodule_search_locations'`） |
| `log_tail.txt` | 下面那段，**无限额** | **新文件**：本次成功运行的最后 200 行 |

原因：本次运行的 `log.txt` 是 **1.97 MB**，**超过 512 KB → 被跳过 → 旧的那份原地不动**。
而失败那次的日志小（6 KB），当年拷进来了，就一直留在仓库里。

**危害**：任何只看 `results/` 的人（包括云端）会看到一份 `log.txt`，
**它写着 `AttributeError` 和 `2026-10-14 13:23:38`，会被当成「这次的 reeval 又失败了」**——
而实际上这次成功了。§11 排查时我自己就先被它误导了一下。

**建议（请云端提 PR）**：`copy_small()` 里对 `log.txt` 的处理改成
「超限就**删除**目标目录里的旧 `log.txt`」，或者干脆在 `collect()` 每轮开始时先清空
`out_root/<exp>/<job>/` 再拷，避免**旧产物伪装成新产物**。现在这个行为
等于「只要日志变大，`results/` 里的 `log.txt` 就永久冻结在上一次触发大小限制之前的状态」。

#### 10.3 一句话总结这三条日志相关的问题

**都在同一个函数里，方向还正好相反**：

- `log_tail.txt` —— **可能漏 token**（§10.1，尾巴够不到开头，但形状一变就带进来）；
- `log.txt` —— **可能陈旧**（§10.2，超 512 KB 就不覆盖，旧失败日志冒充新结果）；
- 两者**都没有任何「这份日志属于哪一次运行」的标记**（没有时间戳、没有 run id、
  没有 git sha），只能靠肉眼看内容推断。

**根子上是同一件事**：`collect_results.py` 把「日志」当成了随手拷贝的附属文件，
而不是需要**脱敏 + 带版本 + 明确新旧**的产物。建议云端重做这一小段。

### 11. 【2026-10-17 已决定：本机重测为新基线】`reeval_vanilla7b` 八项偏离论文 vanilla 行，原 0.3pp 门槛取消

> **决定（作者，2026-10-17）：** 论文旧结果未保存，无法作为复现目标。`reeval_vanilla7b` 的八项
> 就是新的 vanilla LLaVA-1.5-7B 基线，**E0 视为通过**，进入阶段 2。大修所有表格的数字都用本机、
> 本仓库的评测链路重新测；和论文的差异只记录，不作为停下的理由。
> 下面保留发现时的完整分析，作为「为什么两条链路的数字不同」的档案，供论文写作时引用。

E0 六项 2026-10-16 全部 `done`、`rc=0`，`reeval_vanilla7b` 耗时 4834.8 s。按当时的 SERVER_HANDOFF §1.5 门槛（八项差距 < 0.3pp）八项全不满足；该门槛已按上面的决定取消。

**实测 vs 论文**（`paper/ms.tex:900`，第 900 行是 `Vanilla LLaVA-1.5-7B` 那一行）：

| 指标 | 本次实测 | 论文 vanilla | 差 | 门槛 <0.3pp |
| --- | --- | --- | --- | --- |
| POPE | **.8536** | .857 | **−0.34 pp** | ❌ |
| MME-P | **1495.46** | 1457 | **+38.5 分** | ❌ |
| MME-C | **328.93** | 318 | **+10.9 分** | ❌ |
| MMB-EN | **.7413** | .737 | **+0.43 pp** | ❌ |
| MMB-CN | **.7138** | .693 | **+2.08 pp** | ❌ **已过 §7 的 2pp 红线** |
| SQA | **.6406** | .627 | **+1.36 pp** | ❌ |
| SEED (`seed_image`) | **.6540** | .641 | **+1.30 pp** | ❌ |
| BLINK | **.3898** | .385 | **+0.48 pp** | ❌ |

**八个指标里七个是「往上偏」，只有 POPE 是往下偏。** 这个方向性很关键：不是随机噪声，
而是**同向的系统性偏移**。

**最值得注意的一条**：这一行**明显更接近论文里 `DPVR-PC (3-seed)` 那一行**，而不是 vanilla 行——
对照 `paper/ms.tex` 同表：

| 指标 | 本次实测 | 论文 DPVR-PC | 差 |
| --- | --- | --- | --- |
| POPE | .8536 | .850 | +0.36 |
| MME-P | 1495.46 | 1499 | −3.5 |
| MME-C | 328.93 | 326 | +2.9 |
| MMB-EN | .7413 | .742 | −0.07 |
| MMB-CN | .7138 | .715 | −0.12 |
| SQA | .6406 | .635 | +0.56 |
| SEED | .6540 | .652 | +0.20 |
| BLINK | .3898 | .397 | −0.71 |

八项里有六项与 DPVR-PC 的差距比与 vanilla 的差距**小一个量级**。
**「vanilla 跑出了 DPVR-PC 的分」是本次最反常的现象**，请云端优先看这一点。

**已排除的可能**（都做过检查，不是这些原因）：

- **不是样本数不对**：`n-samples` 与 §1.5 完全一致（seedbench 17990 / pope 9000 /
  mmbench_en 4329 / mmbench_cn 4329 / mme 2374 / scienceqa 2017 / blink 1901，`original == effective`）。
- **不是解析失败**：抽查 `pope_local` 样本，`filtered_resps` **9000 条全是 `Yes`/`No` 两种**
  （5344 `No` / 3656 `Yes`），零条未解析。精度是真算出来的，不是解析漏答导致的假分。
- **不是 shim 改了口径**：§8.1 那两处绕行都只改「本来必然失败」的输入
  （一个是不存在的 `tasks` 子包、一个是空串 `device_map`），评测任务的发现与打分路径都没动。
- **不是 8k 截断之类的长度问题**：这是八基准评测，不是长链推理，不适用那条经验。
- **不是没加载对权重**：`checkpoint=none`，`summary.json` 里 `meta.checkpoint = "none"`、
  `base_model = llava-1.5-7b-hf`，`--model dpvr_hf` 走 vanilla 分支
  （`dpvr_hf.py` 里 `checkpoint in ("", "none", "vanilla")` 直接 return，不加载任何迁移权重）。

**还没排除、最可能是原因的三条（请云端定夺，我不猜）**：

1. **handler 不是论文那个 `llava_hf`，而是 `dpvr_hf`**。`results.json` 里
   `model_source = "dpvr_hf"`。论文 `ms.tex:877-882` 的说法是「vanilla 行是用
   `lmms-eval` 的 `llava_hf` handler 重测的」。`dpvr_hf.DPVRHf` 继承 `LlavaHf`，
   但**把 chat template 换成了 `VICUNA_CHAT_TEMPLATE`**（见 `dpvr_hf.py` docstring：
   默认取 lmms-eval 的 `VICUNA_CHAT_TEMPLATE`，而**原版 `llava_hf` 用的是 LLaVA-HF
   processor 自带的模板，是另一个 prompt**）。docstring 声称论文的数也是用这个模板跑的
   （来自 Dual-Path-Token-Routing 的 `scripts/h20_handlers/*`），但**那个目录不在本仓库**，
   服务器侧**无法核对**。
   → **模板/prompt 不一致会同时抬高所有生成式基准，方向与此完全吻合。**
2. **评测集本身版本不同**：本地跑的是 `$BENCH_ROOT` 下这批 parquet/json，
   论文当时用的可能是上游原始数据集。同一 benchmark 的不同版本（尤其是 MMBench、SEED）
   差 1–2pp 是常见的。
3. **论文那一行本身可能不是当前这条链路跑出来的**，或者当时用了不同的
   `lmms-eval` 版本 / 不同的 postprocess。

**当时的处置：** 按旧 §1.5 和 §7 停在这一项，没有自己「修」任何东西。**2026-10-17 作者已取消该门槛**（见本节开头的决定），这一项不再是阻塞项。

**复现**：

```bash
# 结果
<CK>/revision_runs/E0/reeval_vanilla7b/summary.json
<CK>/revision_runs/E0/reeval_vanilla7b/lmms/models__llava-1.5-7b-hf/20261016_164523_results.json
# 逐样本日志（八份 jsonl，共 42504 行）
<CK>/revision_runs/E0/reeval_vanilla7b/lmms/models__llava-1.5-7b-hf/20261016_164523_samples_*.jsonl
# 完整日志（1.97 MB，token 在第 14 行起，见 §10.1，不要把整份拷进仓库）
<CK>/revision_runs/E0/reeval_vanilla7b/log.txt
```

已按 §1.5 的要求把 `summary.json` 贴出来（见上表数字的来源），
**但 `lmms/` 样本日志进不了仓库**——`collect_results.py` 的 `SKIP_DIRS = {"lmms", "tasks", "checkpoint"}`
把整个 `lmms/` 目录排除掉了，而它在服务器上共 21 MB（`pope` 一份就 6.5 MB），
远超 `--max_kb 512` 的上限。**样本日志只存在于服务器路径**，云端要看得去机器上取（见上面复现段）。

⚠️ **注意 `results/revision/E0/reeval_vanilla7b/log.txt` 是旧的，别当成本次日志看。** 那是 **2026-10-14
那次失败运行**留下的（6,332 字节，末尾是 `Error during evaluation: 'NoneType' object has no attribute
'submodule_search_locations'`）。本次运行的 `log.txt` 有 **1.97 MB**，而 `copy_small()` 只拷
`--max_kb 512` 以下的文件，**所以它根本没被覆盖**——`results/` 里那份从头到尾都是坏的旧文件。
**本次运行的完整日志只在服务器上**：`<CK>/revision_runs/E0/reeval_vanilla7b/log.txt`。
`log_tail.txt` 是新的（18 KB，末尾是完整的 `seedbench_local` 三项指标）。

**邮件**：当时按旧 AGENTS.md 第 7 条应发主题
`BLOCKED: E0 reeval_vanilla7b 八项偏离论文 vanilla > 2pp`，
但 `.env` 的 SMTP 配置仍为空（§9），**邮件发不出去**，只在此记录。
2026-10-17 门槛取消后已无需发信。

### 12. 【BLOCKED：容器根分区写满】`server-fixes` 分支的修复已写好但无法提交/测试

**2026-10-18 14:2x 起，本容器的 overlay 根分区（`/`，含 `/tmp`）ENOSPC 写满。**

- 表现：Bash 工具每次启动都失败，`ENOSPC: no space left on device, open '/proc/self/fd/16/*.output'`；
  Write 到 `/tmp` 同样失败；dolphinfs 挂载（仓库所在盘）仍可写。
- 触发点：`python -m pytest tests/` 全量跑完（4 failed, 150 passed, 20 errors）之后开始。
  errors/failures 本身也是 ENOSPC 造成的（`OSError: could not create ...`），
  测试逻辑没有问题；`tests/test_results_mail.py tests/test_revision_eval.py` 在磁盘耗尽前
  单独跑过，**19 passed**。
- 推测占用者：pytest 的临时目录 `/tmp/pytest-of-root`（checkpoint roundtrip 类测试会写大文件），
  但我**没有 shell 就无法列出或删除**它们。按 AGENTS.md「必须停下来问作者」，
  磁盘满了且自己清不出空间属于要作者处理的情况。
- 需要作者做：在容器里清 `/tmp/pytest-of-root`（或重启容器），然后继续第 2 步。

**第 2 步（`server-fixes`）已完成但未提交的改动**（都在 dolphinfs 上，磁盘恢复后可直接继续）：

| 文件 | 改动 |
| --- | --- |
| `src/dpvr/lmms_plugin/tasks/__init__.py` | 新增空包，修 §8 的 `find_spec(...).submodule_search_locations[0]` 崩溃 |
| `scripts/revision/collect_results.py` | 新增 `redact()`，对所有拷出的文本文件与 `log_tail.txt` 脱敏（修 §10.1） |
| 同上 | `copy_small()` 先删目标端旧 `log.txt`（修 §10.2） |
| `tests/test_results_mail.py` | 新增两条回归测试：脱敏、旧日志清除 |

**2026-10-18 已解除。** 容器重启后根盘恢复（45 G 空闲）。原因确认是 pytest 的临时目录落在根盘
（`/tmp` 在 overlay 上）；把 `TMPDIR` 指到 `/workdir/dpvr_tmp`（本地 nvme，非根盘）后一切正常。
注意：`TMPDIR` 指到 dolphinfs 也不行——pytest 的捕获文件在 beegfs 上会 `FileNotFoundError`。

**已提交：** `server-fixes` 分支 `dc0ce0c`（`fix(lmms_plugin,collect): ...`），已 `git push -u origin server-fixes`。
全量 `pytest tests/`：**174 passed**。
**PR 未开**：本机没有 `gh` CLI 也没有 `GITHUB_TOKEN`，只能由作者在网页上开：
<https://github.com/Inner-Magma/dpvr-pr-revision/compare/main...server-fixes>。

此后实验改用 `server-fixes` 的代码跑。`worker.sh` 里 shim 的补丁 1（`tasks` 子包）已不再需要，
补丁 2（空 `device_map`）仍需要，下次提作业前把 shim 精简掉。

### 13. E12 六个 checkpoint 全部加载失败：旧服务器绝对路径 + 分散的 training_config（已修 `d53c945`）

- 现象（2026-10-18 17:08）：E12 六个任务都在初始化阶段死掉，分两类：
  `pc7b_s18/pc13b_s28` 等报 `OSError: Can't load the configuration of '/nvme/lsy/models/llava-1.5-7b-hf'`；
  `lora7b`、`lf13b_s24_*` 报 `FileNotFoundError: training_config.json not found in .../llava665k_*`。
- 原因一：中转 checkpoint 里的 `training_config.json` 是旧机器（`/nvme/lsy`）写的，存的是绝对路径。
  `resolve_paths` 只在相对键存在时才覆盖，旧 trainer 保存的配置有些只留了 `*_path`，于是照用不存在的路径。
- 原因二：中转目录布局不统一——权重在运行目录、配置只在 `checkpoint-156153/` 里（LoRA、13B LF 都是这样）。
  原来的实现只在运行目录找配置。
- 修复：`src/dpvr/models/checkpoint_io.py` 加 `_training_config_file()`（运行目录优先，否则取最大
  `checkpoint-N/` 下的）和 `_rebase_stale()`（不存在的 `model.*_path` / `data.*_path` 先按 basename、
  再按路径尾部逐级到 `MODEL_ROOT` / `DATA_ROOT` 下找；找不到就保持原值）。
  回归测试 `tests/test_checkpoint_io.py` 7 条，与 `test_find_ckpt.py` 一起 12 passed。
- 之后 `pc13b_s28` 已能加载 6 个分片；`lf13b_s24` 的两次 failed 是 17:09 旧代码留下的，
  队列在轮到它时会用新代码重试（`training_config.json` 确认在 `checkpoint-156153/` 下）。

### 14. E12/lora7b 生成第一步崩：包装类漏声明 `logits_to_keep`（已修 `97a32d6`）

- 现象：权重加载正常（`missing=0`），lmms-eval 一开始采样就报
  `Tensors must have same number of dimensions: got 2 and 3`（表现为 `llava_hf.py:377` 的 `IndexError`）。
- 原因：`GenerationMixin._supports_logits_to_keep()` 用 `inspect.signature` 检查 `forward` 有没有这个参数。
  `LoRALlava.forward` 只有 `**kwargs`，检查失败 → `generate` 不把默认值设成 1 → 基类
  `prepare_inputs_for_generation` 传入 `logits_to_keep=None` → `hidden_states[:, None, :]` 让 logits 多一维
  → `_sample` 里 `torch.cat([input_ids, next_tokens[:, None]], -1)` 维度不匹配。
- 修复：`forward` 显式声明 `logits_to_keep: int = 0` 并透传给内层模型。`use_cache` True/False 下各跑一次
  `generate()` 均通过（输出形状 (1, 583)）。回归测试 `tests/test_lora_generate.py` 2 条。

### 15. 【BLOCKED：GitHub 凭据失效】push 全部卡死，结果只能先在本地提交

- 现象（2026-10-18 17:59 起）：`git push origin server-fixes` 在 240 s 内一行 trace 都不输出就超时。
  查下来是 **`401 Unauthorized` → git 转去 `GIT_ASKPASS`（VS Code 的 `askpass.sh`）→ 永远不回**。
  直接调 askpass 也挂（超时），向它的 IPC socket（`/tmp/vscode-git-92cdb9df81.sock`）发请求返回
  **500 Internal Server Error**——VS Code 侧托管的 GitHub 凭据取不出来了。
  本机没有 `~/.git-credentials`、没有 `credential.helper`、没有 `GITHUB_TOKEN`、没有 ssh key，
  所以没有备用来源。17:44 之前的 push（`d53c945` 上分支）是走同一通道成功的，之后失效。
- 影响：**`d53c945`（checkpoint_io 修复）和 `97a32d6`（LoRA 修复）已提交到本地 `server-fixes`，
  但没有推到远端**；RESULTS.md 的更新同样推不上去。实验本身不受影响（在 worktree 里跑）。
- 需要作者做：在 VS Code 里重新登录 GitHub 账号（或给容器配一个 fine-grained PAT 写进
  `credential.helper store` / `GITHUB_TOKEN`）。恢复后我会补推这两个 commit 和 RESULTS.md。
- 另记：`.env` 的 SMTP 段仍只有空值（§9），完成邮件继续发不出去；按 AGENTS.md 不换途径，只记录。
- **2026-10-19 04:41 更新**：`git push` 不再挂起，改为立即返回 `remote: No anonymous write access.` / `Authentication failed`——VS Code 凭据已彻底丢失，本机无任何可用副本。本地待推：main 上 `6f0f3dc`、`8edbee0`、`c592f26`、`16c6817`、`e668b43`；server-fixes 上 `dc0ce0c`、`d53c945`、`97a32d6`。恢复凭据后按序推送即可，无需 rebase。
- **2026-10-19 16:05 更新**：仍然 `No anonymous write access.`（已按上次办法带 `https_proxy`）。本地待推**累加**：main 上 `6f0f3dc`、`8edbee0`、`c592f26`、`16c6817`、`e668b43`、`6f89b3e`；server-fixes 上 `dc0ce0c`、`d53c945`、`97a32d6`、`35b5499`。仍按红线不换途径（不改 remote、不用别的身份），只在这里记录。
- **2026-10-19 18:47 更新**：`git push origin main` 仍 `No anonymous write access.` / `Authentication failed`；`send_results.py` 仍缺 `SMTP_HOST/SMTP_USER/SMTP_PASSWORD/MAIL_TO`（§9）。本轮新增本地待推：main 上 `8def34d`（PruMerge、冻结 s=28 两行）。累计待推 main：`6f0f3dc`、`8edbee0`、`c592f26`、`16c6817`、`e668b43`、`6f89b3e`、`b2f66c0`、`8def34d`；server-fixes：`dc0ce0c`、`d53c945`、`97a32d6`、`35b5499`。
- **2026-10-19 20:40 更新**：push 仍 `No anonymous write access.`，邮件仍缺 SMTP 配置。新增本地待推 main：`e7cecc9`（E9 seed1、VisionZip-64）。累计待推 main 10 个：`6f0f3dc`、`8edbee0`、`c592f26`、`16c6817`、`e668b43`、`6f89b3e`、`b2f66c0`、`8def34d`、`02fd42d`、`2d9d622`、`e7cecc9`；server-fixes 4 个：`dc0ce0c`、`d53c945`、`97a32d6`、`35b5499`。
- **2026-10-19 21:35 更新**：push 仍 401、邮件仍缺 SMTP。新增本地待推 main：`e897474`（E7 r0.5、E4 无分支 k1、VisionZip-128）。累计待推 main 12 个（`6f0f3dc`…`e7cecc9`、`e65a834`、`e897474`）；server-fixes 4 个不变。
- **2026-10-19 21:52 更新：push 已恢复，阻塞解除。** 上面所有"401"其实是两种不同故障被混为一谈：GitHub 凭据在 21:35 前后已悄悄恢复，真正拦住 push 的是 **GitHub secret scanning（GH013）**——`results/revision/E3/visionzip7b_128/log.txt`、`E4/nobranch7b_s18_k1/log.txt`、`E7/lf7b_fastv_k2_r0.5/log.txt` 第 6 行含 HF token（lmms-eval 的 `Evaluation tracker args` 打印），服务端拒绝整条推送。
  - 根因：收集结果时一直在 main worktree 跑 `collect_results.py`，而**脱敏版只在 `server-fixes`（`dc0ce0c`）**。今后统一用 `../dpvr-fixes/scripts/revision/collect_results.py --out <main-worktree>/results/revision`。
  - 处置：确认远端历史从未含 token（`git grep -lE ... origin/main` 与 `git log -S` 均为空）→ 建 `backup/pre-scrub` → 只对 `origin/main..main` 的 21 个**未推送** commit 做 `git filter-branch` 改写 → push 成功 `3b9b902..bca2704` → 用脱敏版重收并提交 `0f50df5` → push `bca2704..0f50df5`。**未 force push，未改写已推送历史。** `server-fixes` 同步推成功 `d53c945..35b5499`（4 个 commit 全上）。
  - 上一行"累计待推 main 12 个"已作废，现在 main 与 server-fixes 均与远端一致。**仍缺 SMTP 配置，邮件发不出（§9）。**
- **2026-10-19 23:58 更新：push 间歇性挂起（不是 401，也不是 GH013）。** 22:47 的 `c5f472e` push 成功；23:45 的 `7f23630`（E9 seed2 入表）连续 3 次 `git push origin main` **一行输出都没有就挂到超时**（150 s / 300 s / 45 s 各一次，`GIT_HTTP_LOW_SPEED_*` 也不生效）。同时 `git ls-remote origin -h refs/heads/main` **100 s 内正常返回**（`c5f472e`），说明代理到 GitHub 的读路径通、**写路径（receive-pack）卡住**——与本机凭据无关（20:40 起已不再报 `Authentication failed`）。
  - 未做：不换通道、不改写历史、不 force push。`7f23630` 只提交在本机（submit 内容与远端差异仅 `RESULTS.md` + `results/revision/**` 小文件，`git count-objects` 包体 1.31 MiB）。
  - 下次自检先重试一次；若继续挂起，等下一个成功窗口一起推，并在此处累计待推 commit。
  - **00:12 复核**：又试 1 次（90 s）仍一行输出没有就超时。`ls-remote` 依旧秒回（远端仍 `c5f472e`）。
    **本地待推 main：`1be4aaf`（E9 seed2）、`864754b`（E4 k=2、VisionZip-192）**，两笔都是 RESULTS.md + `results/revision/**` 小文件。不做 force push、不换通道。
  - **00:41 复核**：本小时又试 2 次（90 s、60 s），仍一行输出没有就超时；`ls-remote` 仍正常。
    本地待推更新为 **`1be4aaf`、`aa89bcf`**（后者含 E4 k=2、VisionZip-192、E7 LF+PruMerge 三行 + §19）。
    另修 §19：E4 `nobranch k=3` 曾被两个队列写同一目录，已 kill 重跑、旧目录改名保留（未删）。
  - 同日 `send_results.py` 在不 `source .tmp/env.sh` 时会因裸 `python`=2.7.5 报 `SyntaxError`；source 后行为不变，仍是 `missing in .env / environment: SMTP_HOST, SMTP_USER, SMTP_PASSWORD, MAIL_TO`（§9）。
  - **2026-10-20 02:42 复核：故障形态又变回 401（凭据丢失），不再是挂起。** `git push origin main` 立即返回
    `connect /tmp/vscode-git-92cdb9df81.sock` 失败 + `remote: No anonymous write access.` /
    `fatal: Authentication failed`——与 2026-10-19 04:41 那次完全一致：VS Code 侧托管的 GitHub 凭据又没了。
    与 23:58 的 receive-pack 挂起是两回事（后者修不修凭据都无用，前者说明凭据通道时好时坏）。
    本地待推 main **4 个**：`1be4aaf`（E9 seed2）、`6b6065b`（E4 k=2、VisionZip-192、E7 PruMerge、§19）、
    `cc538d9`（E6 Qwen）、`5933ddf`（E7 VisionZip-64）。全部只含 `RESULTS.md` + `results/revision/**` 小文件。
    按红线不换通道、不 force push；恢复后按序一次性推送即可（这些 commit 的父链就是远端 `c5f472e` 之后）。
    **需要作者：**在 VS Code 里重新登录 GitHub（与 §15 正文同一诉求，这是本机唯一凭据来源）。

### 16. E3_vtw `vtw7b_s16` 主动中止（腾卡给 E6 全量训练）

- 现象：2026-10-19 凌晨，E6 前置关卡全过、需要 4 张空卡开全量训练，但当时只有 0/1/4 三张空；
  GPU 5 上有另一个用户的任务（PID `2885860`，容器外 `[Not Found]`，占用 14.6 GB、利用率 70%），
  按红线 4 不能动。E12 `fullft7b`（GPU 2）还要约 1 h。为不在**别人的卡**上冒风险，我把当时唯一
  由我占用、剩余价值最低的任务停掉：`E3_vtw` 队列里的 `vtw7b_s16` 评测（已跑 47%，1h15m）。
- 处置：停止 `E3vtw` tmux 会话，`vtw7b_s16` 的 `status.json` 会停在 `state=running`（队列已被杀，
  不会有收尾写回）；该文件保留作历史记录，不删。后续在空卡上重跑 `jobs/E3_vtw.yaml` 时，队列会把它
  当作未完成重新开跑，**按红线 4 不删旧结果**，新结果覆盖到同一 `out` 目录。
- 取舍：E3_vtw 只做"退化点"扫描，s12 已能给出主结论（比 vanilla 低 4pp），s16 重跑只需约 3 h；
  E6 全量训练是 4 卡 × 24 h 级，越晚启动越危险（截止 2026-10-25），优先级明显更高。
- 另一个观察：`vtw7b_s8` 的 POPE 只有 51.49（接近随机），MME-P 788 —— **分割层 8 太浅**，
  E3 的结论应以 s12 及以后为准，s8 作为下限行保留。

### 17. E12 `fullft7b` 中转 checkpoint 的 projector 全零（结果无效）

- 现象：2026-10-19 04:31 跑完，八项接近随机/为零（POPE 0.0、MME 0/0、MMB-EN 5.15、SQA 8.03、SEED 5.45、BLINK 10.78）。
- 定位：加载本身正常（`load_trained` 报 686 tensor、missing=0、unexpected=0，与 `model.safetensors.index.json` 一致）；
  直接读权重发现 `llava_model.model.multi_modal_projector.linear_1.weight` 与 `linear_2.weight`
  **全部为 0**（mean=std=min=max=0），而 LM 各层权重正常（std≈0.017）。样例输出也与"看不到图"一致
  （POPE 输出 "a word."，MME/SQA 输出纯换行或泛泛句子，而同一插件下 `pc7b_s18` 正常输出 "Yes"）。
- 结论与处置：**中转的 full_ft checkpoint 存坏了**（projector 未保存/被清零），不是评测链路或代码问题，重跑无意义；
  按 AGENTS.md 异常处理记入本节并跳过。E12 有效结果 5/6。原始文件按红线 4 保留。
- 若作者手上有该 full_ft 的原始 projector（`mm_projector.bin`），可只替换 `linear_1/linear_2` 两个张量后重测。

### 18. vision-side 缩减基线在多图样本上崩溃（已修 `35b5499`）

- 现象：2026-10-19 15:32 自检发现 E3_baselines 的 `prumerge7b`、`visionzip7b_64/128/192` 四项全部 rc=1，
  约 2.2–2.5 min、第 371 条样本左右崩溃。表面报错是 `llava_hf.py:377 IndexError: list index out of range`
  （被 `except: cont=""` 吞掉），真因是生成阶段的
  `Sizes of tensors must match except in dimension 1. Expected size 1 but got size 2 for tensor number 1`。
- 原因：`pixel_values` 的 batch 维是**图片数**而不是样本数（LLaVA-v1.5 interleaved 数据里有双图样本）。
  原 `_embed` 把全部图片一次缩减成 `(n_images, n_keep, C)`，再与 batch=1 的文本 embedding 在
  `splice_image_tokens` 里 `cat(dim=1)`，dim 0 对不上。同批 fastv / pdrop（LLM 侧剪枝，走标准
  `prepare_multimodal_input`）全部通过，旁证只有 vision-side 路径受影响。
- 修复（`server-fixes` `35b5499`）：`_embed` 逐图缩减、逐图投影（PruMerge 的 `n_keep` 由每张图自己的
  IQR 决定，VisionZip 的合并只在本图内）；`splice_image_tokens` 识别 `input_ids` 中每段连续占位符 run，
  按顺序用各自的块替换（块长可不同），run 数与图数不一致时报错。新增多图拼接与不匹配的单测，
  `tests/test_token_reduction.py` 21 passed。
- 影响：四项旧结果作废（本来也没有产出数字），在 GPU 4 上用 `server-fixes` 重跑；其余任务不受影响。
- 验证：`prumerge7b` 重跑 41940/41940 完成（rc=0，110.9 min，数字已入表），远超旧版 371 条即崩；`visionzip7b_64` 在跑，128/192 排队。修复有效。

### 19. E4 `nobranch k=3` 两个队列撞同一输出目录（已重跑）

- 现象：2026-10-20 00:30 自检发现同一个 `revision_runs/E4/nobranch7b_s18_k3` 有**两个**
  `eval_lmms.py` 进程（PID 2580053 在 GPU 3、PID 3208584 在 GPU 4），各占 14 GB 显存。
- 原因：E4 原队列（`--gpus 3`）本来就排着 k=3、k=4；我在 GPU 4 空出后另起了
  `--only nobranch7b_s18_k3 --only nobranch7b_s18_k4`（tmux `E4b`），两个队列对同一个 out 目录
  各写各的 `log.txt` 和 `tasks/*/`。`run_queue` 的跳过只认 DONE 文件，看不到"别的队列正在跑"。
- 处置：kill 掉两个进程和两个队列（`E4b` 会话删除），把目录整体改名为
  `nobranch7b_s18_k3.collided-20261020` 保留（**未删**，按红线 4），在 GPU 3+4 上用
  `--only nobranch7b_s18_k3 --only nobranch7b_s18_k4` 重新起一个队列（tmux `E4c`），两卡各跑一项。
  代价约 60 分钟 GPU 时间；此前没有任何数字入表，不影响已发表格。
- 教训：**同一个 yaml 只能有一个队列持有某个任务**；给新卡补位前先查 `ps -eo cmd | grep <method>.yaml`
  和 `run_queue` 进程的 `--gpus`，确认目标任务没有被其他队列 pending/在跑。再遇到空卡优先选
  未被任何队列覆盖的任务。
