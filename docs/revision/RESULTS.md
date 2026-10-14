# 大修实验结果总表

由服务器上的 Claude Code 维护：每完成一项追加一行，新的在上面。云端 Claude 据此整理论文表格。

## 服务器信息

| 项 | 值 |
| --- | --- |
| GPU 型号 × 数量 | **8 × NVIDIA H100 80GB HBM3**（81559 MiB/卡，700 W，sm_90）。⚠️ 申请的是 `worker.gcoresh800-80g = 8`，拿到的既不是文档写的 H800、也不是中途出现过的 H20-3e，见"问题" §1 |
| 驱动 / CUDA | 驱动 550.127.08；nvidia-smi 报 CUDA 12.4；torch 自带 CUDA 12.8（`torch==2.9.0+cu128`） |
| 容器镜像 | `.../aipnlp_base_pytorch2.3.1_cuda12.4.0_python3.9_1_18_glibc_2.28_2532322d`（CentOS 7 壳 + 换过的 glibc 2.28）。原镜像 glibc 2.17 跑不动本环境，见"问题" §6 |
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
| 2026-10-14 | E0 验收 | `provenance_7b`（训练起点核查） | **verdict: language_model=llava_hf, projector=llava_hf**。11 个探针权重与 `llava-1.5-7b-hf` 的距离全为 **0.0**，与 vicuna / stage1_projector 参照均为非零（4e-3 ~ 1.3e-2）。即该 checkpoint 的 LM 与 projector 都等于 llava-1.5-7b-hf 原版 | `results/revision/E0/provenance.json` | 014ea25 |
| 2026-10-14 | E0 验收 | `latency_vanilla7b` | **28.65 ± 0.40 ms**（p50 28.56），peak 13.26 GB。协议 batch=1, n_img=576, n_txt=128, bf16, use_cache=False, warmup=10, iters=100。卡型 **H100 80GB HBM3** | `results/revision/E0/latency_vanilla7b.json` | 014ea25 |
| 2026-10-14 | E0 验收 | `latency_lf7b` | **25.48 ± 0.93 ms**（p50 24.97），peak 13.64 GB。同上协议；method=token_diversion_x3_fusion, split_layer=18, vision_depth=1, freeze=all_but_single。**相对 vanilla 加速 1.124×** | `results/revision/E0/latency_lf7b.json` | 014ea25 |
| 2026-10-14 | E0 验收 | `smoke_train_lf7b`（50 step 冒烟） | **跑通**：50/50 step，train_loss 2.642，train_runtime 14.4 s（3.47 step/s）。无 NaN（loss 区间 1.05~7.17，grad_norm 0.22~1.27） | `results/revision/E0/smoke_train_lf7b.json`（status.json / log.txt 在 `revision_runs/E0/smoke_train_lf7b/`） | 014ea25 |
| 2026-10-13 | E0 验收 | `download_assets.py --verify` | 7/7 benchmark 样本数与论文一致 | `results/revision/E0/assets.json` | 4b4f56a |
| 2026-10-13 | E0 验收 | `check_env.py` | 软件栈/路径/8 卡可用性全过；模型与 665k 图像在跑这条时尚未下载完 | `results/revision/E0/env_check.json` | 4b4f56a |

> 延迟两项与 `latency_vanilla7b`/`latency_lf7b` 的 `latency.json` 里都写了 `"gpu": "NVIDIA H100 80GB HBM3"`，Table 7 引用时按 H100 标注。
> E0 六项里 **5 项 done、1 项 failed**（`reeval_vanilla7b`，见"问题" §8），因此 E0 验收**尚未全过**，按 AGENTS.md 阶段门不进阶段 2。

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

### 4. E0 尚未真正跑完

`results/revision/E0/` 目前只有环境与资产核对产物（`env_check.json`、`nvidia-smi.txt`、`pip-freeze.txt`、`assets.json`）。以下 E0 项**还没有产物**：

- `provenance.json`（训练起点核查，需要 `--set prov_ckpt=...`）
- `reeval_vanilla7b`（vanilla 7B 八基准重测）
- `latency_vanilla7b` / `latency_lf7b`（延迟，且受问题 §1 卡型影响）
- `smoke_train_lf7b`（50 step 冒烟训练）

另：665k 图像还缺两项——`coco/train2017`（0 张）、`vg/VG_100K`（0 张，`vg_images1.zip` 在 cs.stanford.edu 上三轮 404），`ocr_vqa/images` 抽取返回 0 张。`DL_IMAGES_DONE 有缺项，未放行`。这三项补齐前不能进阶段 2。

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
