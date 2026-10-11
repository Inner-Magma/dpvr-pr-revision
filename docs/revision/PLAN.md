# DPVR 大修总体规划（PR-D-26-07438）

本仓库专门用于这次 Pattern Recognition 大修，内容 = `Inner-Magma/dpvr` main 的完整历史 + 大修期间的代码、实验结果与论文源文件。原 `dpvr` 仓库保持不动。

- 审稿意见原文与编号：[REVIEWS.md](REVIEWS.md)
- 服务器端执行手册：[SERVER_HANDOFF.md](SERVER_HANDOFF.md)
- 实验结果总表：[RESULTS.md](RESULTS.md)
- 论文源文件：[`paper/`](../../paper/)（`ms.tex`，cas-sc 单栏双倍行距，当前 33 页，上限 35 页）
- 详细分析（Claude Docs，作者可在线批注）：审稿意见分析与修改计划、大修执行规划

## 四个阶段

1. **环境**：新服务器复现论文已有数字后才进入实验（见 SERVER_HANDOFF §1）。
2. **代码**：每项改动一个 PR（C1–C10），带单元测试；作者 review 后合入 main。
3. **实验**：服务器 Claude Code 按 SERVER_HANDOFF §3 的顺序运行 E0–E11，结果写入 RESULTS.md。
4. **论文**：先改不依赖实验的部分，实验出结果后填表；用 latexdiff 生成标红版本。

## 代码改动清单

| 编号 | 改动 | 主要文件 | 验收 | 服务实验 |
| --- | --- | --- | --- | --- |
| C0 | 服务器流水线：锁版本的环境脚本与自检；8 卡任务队列；统一评测入口（lmms-eval `llava_hf` + DPVR 插件，`use_cache=False`）；训练起点权重比对；延迟基准；E0/E9 任务文件 | `scripts/revision/`、`jobs/`、`src/dpvr/pipeline/`、`src/dpvr/lmms_plugin/` | 单测通过；服务器上 E0 全部通过 | E0 及之后全部 |
| C1 | seed 传入 `TrainingArguments`（`seed`、`data_seed`）；锁定依赖版本 | `src/dpvr/training/trainer.py`、`requirements-lock.txt` | 不同 seed 的前 10 个 batch 样本 id 不同 | E9 |
| C2 | DPVR-LF KV-cache 解码：深层（s..L-K-1）只缓存文本 KV，融合层和浅层缓存全序列；支持左 padding | `src/dpvr/models/token_diversion_x3_fusion.py` | 有/无 cache 的 greedy 输出一致，logits 误差 < 1e-3 | E1 |
| C3 | 免训练变体：VTW；无侧枝 + 末层融合；从第 s 层起冻结图像隐状态 | 新 `src/dpvr/models/vtw_baseline.py` 等，注册到 `registry.py` | 复现 VTW 原文至少一个数字 | E3, E4 |
| C4 | DPVR-LF + FastV / PruMerge | 仿 `td_fastv.py` 新增 `lf_fastv.py` | 剪枝比例为 0 时与 DPVR-LF 输出一致 | E7 |
| C5 | 外部基线接入（PruMerge + 2–3 个近期方法） | `src/dpvr/models/` | 各复现原文一个数字 | E3 |
| C6 | 饱和分析通用化：去掉写死的 576 / image token id 32000；残差范数指标；图像 token 池化 | `scripts/preliminary*.py`、`src/dpvr/analysis/` | LLaVA-1.5-7B 上重画出 Figure 2 | E2, E8 |
| C7 | 效率测量：解析 FLOPs（独立求 ρ）；统一 B=1；端到端生成延迟；去掉写死的 `cuda:5` | `src/dpvr/profiling/flops.py`、`scripts/profile_efficiency.py` | 解析 FLOPs 与 fvcore 误差 < 5% | E1, E5 |
| C8 | 第二骨干支持：Qwen2.5-VL-7B-Instruct（2026-10-11 定，见下）；解除 `T_txt = S - max_img` 的"batch 内图像 token 数相同"假设 | `routing.py`、`token_diversion_x3_fusion.py`、`token_diversion_x3_fusion_qwen.py`、`data/qwen.py`、插件 `dpvr_qwen25vl`、新 config | 变长图像 token 的单测；smoke 训练 | E6 |
| C9 | 侧枝变体：MLP-only、LoRA 侧枝；可选用第 s 层权重初始化 | `token_diversion_x3_fusion.py` | 参数量正确；梯度可达 | E10 |
| C10 | README 降调；统一论文中的代码链接 | `README*.md` | 链接可匿名访问 | — |

### 第二骨干：Qwen2.5-VL-7B-Instruct

作者 2026-10-11 要求用"最好的"第二骨干。选 Qwen2.5-VL-7B-Instruct 的理由：

- R2.2 点名的 Qwen-VL 系列；与 LLaVA/Vicuna 无关（Qwen2.5 LLM、自研 ViT、M-RoPE），能回应 R1.14 "LLaVA/Vicuna 特有"的质疑。
- lmms-eval 0.5.0 有官方 `qwen2_5_vl` handler，vanilla 数字可与公开结果对照；Qwen3-VL 虽更强，但 0.5.0 没有 handler（要自写评测，审稿人难核对），且有 DeepStack 往前几层注入视觉特征。
- 动态分辨率，每张图 token 数不同，正好覆盖 C8 的变长路由和 R2.7 "分割层与图像 token 数的关系"。
- 7B 档与论文主实验同规模，单卡训得动；32B/72B 训练成本高一个量级且无同规模 LLaVA 对照。

## 时间（2026-10-25 截止，不延期）

| 日期 | 内容 |
| --- | --- |
| 10/11–10/13 | H800 环境与验收（E0）；E1、E5 延迟（整机空闲时） |
| 10/13–10/14 | E2 饱和分析，定 Qwen 分割层；E12 旧 checkpoint 重测；E6、E9 开训 |
| 10/14–10/21 | E3（VTW 优先）、E4，之后 E3 其余基线、E8、E7、E10；E6/E9 评测；论文改写不依赖实验的部分 |
| 10/21 | 结果冻结 |
| 10/22–10/25 | 结果填入论文、回复信、合作者审阅、提交 |

时间不够时按此顺序放弃（先放弃的排前面）：E10 → E7 → E8 → E3 非 VTW 基线 → E6 全量训练。最低配置：C1–C3、C7；E1、E3（仅 VTW）、E4、E5；全文降调。
