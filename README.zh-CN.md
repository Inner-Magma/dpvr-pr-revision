# DPVR:双路径视觉 Token 路由 (`dpvr`)

[English](README.md) | **简体中文**

> **仅靠末层 Fusion 就足够 (Late-Layer Fusion is Enough)。** 视觉 token 的表征在
> MLLM 的最后几层之前就已饱和。我们在 LLaVA-1.5 的饱和点处将视觉 token 分流到
> 单个可训练的侧支模块,深层网络只做**纯文本前向**,并**仅在最后一层**重新融合
> 视觉流与文本流。仅用约 3% 的可训练参数,即可在标准基准上达到或超过全量微调,
> 同时将 A800 实测延迟降低约 28%。

本仓库是 **双路径视觉 Token 路由 (Dual-Path Vision Token Routing, DPVR)** 方法及
其核心实例 **DPVR-LF(Late-Layer Fusion,末层融合)** 的参考实现(*Pattern
Recognition* 期刊投稿)。基座模型为 LLaVA-1.5(Vicuna + CLIP-ViT-L/14-336);仅训练
一个解码块(约占参数量的 3%,7B 模型为 2.02 亿)。

---

## 方法

### DPVR-LF(末层融合)—— 本文提出的方法 ★

```
输入 ─► 层 0…17(冻结,共享)  ─► 在 s=18 处分流
          │
          ├─► single_transformer(1 个可训练侧支块)
          │        └─ 生成图像表征
          │
          ├─► 层 18…30(冻结):纯文本前向
          │     图像位置被完全跳过 —— 这 13 个深层不为图像 token
          │     计算任何 Q/K/V/O/FFN  → 获得完整算力节省
          │
          └─► 层 31(冻结):单次图文融合
                图像位置由侧支输出重新装配回去,
                再用一个全注意力块融合两条流 ─► lm_head
```

- 图像 token 在 13 个深层被跳过,只在最后一层重新融合 → 相比逐层路由的基线,
  获得完整的算力节省。
- 7B 模型(32 层)在最后一层(索引 **31**)做融合;13B 模型分流点为 `s = 24`。
- 融合层数在 **K = 1** 时即已饱和 —— 第二个融合层仅带来约 0.18 个百分点的提升,
  因此单个末层融合就足够。
- 实现:[`src/dpvr/models/token_diversion_x3_fusion.py`](src/dpvr/models/token_diversion_x3_fusion.py)
- 配置:[`configs/method/token_diversion_x3_fusion.yaml`](configs/method/token_diversion_x3_fusion.yaml)

所有 DPVR 变体都共享冻结的浅层栈(`L0`–`L17`)和这一层可训练侧支,区别仅在于
深层栈中如何处理图像位置。DPVR-LF 是用于主结果的配置。

### 路由基线 / 消融

这是两种前置的路由策略,作为架构消融对照保留(对应论文图 3):

| 变体 | 深层栈对图像 token 的处理 | 算力节省 | 代码 |
|------|--------------------------|----------|------|
| **DPVR-PC**(逐层覆盖) | 每个深层都把图像位置重置为侧支输出;仍跑全注意力 | 无 | [`token_diversion.py`](src/dpvr/models/token_diversion.py) |
| **DPVR-KV**(KV 缓存替换) | 图像位置在深层中**仅贡献 K/V** | 部分(约 25–35%) | [`token_diversion_substitution.py`](src/dpvr/models/token_diversion_substitution.py) |
| **DPVR-LF**(末层融合)★ | 图像位置在 13 层中**完全跳过**,仅末层融合 | 完整 | [`token_diversion_x3_fusion.py`](src/dpvr/models/token_diversion_x3_fusion.py) |

### 共同设置

- 基座:LLaVA-1.5(Vicuna + CLIP-ViT-L/14-336)
- 可训练:一个 `LlamaDecoderLayer` 侧支(约 3% 参数;7B 为 2.02 亿)
- 精度:bf16;标签会屏蔽用户轮次与图像 patch token
- 数据:LLaVA-1.5-mix-665k(各变体使用完全相同的数据集以保证公平对比)

---

## 安装

```bash
git clone https://github.com/Inner-Magma/DPVR.git
cd DPVR
pip install -e .            # 可附加 [eval,analysis,profile] 安装可选依赖
```

为你的机器配置路径(代码中所有绝对路径都从这些根目录解析):

```bash
cp .env.example .env                                # 或:
cp configs/paths.yaml.example configs/paths.yaml    # 填入 DATA_ROOT / MODEL_ROOT / CKPT_ROOT
```

## 数据

LLaVA-1.5-mix-665k 指令 + 图像。用以下命令下载图像:

```bash
bash scripts/download_llava_images.sh /path/to/llava-1.5-665k-instructions
```

## 训练

```bash
# DPVR-LF(本文提出,末层融合)—— 主模型:
python scripts/train.py --config configs/method/token_diversion_x3_fusion.yaml

# 路由基线 / 消融:
python scripts/train.py --config configs/method/token_diversion.yaml                # DPVR-PC
python scripts/train.py --config configs/method/token_diversion_substitution.yaml   # DPVR-KV
```

可在命令行覆盖任意配置项(断点续训 / 消融示例):

```bash
python scripts/train.py --config configs/method/token_diversion_x3_fusion.yaml \
    method.split_layer=24 \
    training.output_subdir=token_diversion_x3_fusion/ablate_split_24 \
    training.resume_from_checkpoint=/path/to/checkpoint-NNNNN
```

13B 变体以及融合层数 / 分流层的消融配置一并放在 [`configs/method/`](configs/method/)
(例如 `token_diversion_x3_fusion_13b_split24_d1.yaml`、
`token_diversion_x3_fusion_last2.yaml`);其他基线(`full_ft`、`lora`、`fastv`)也在
这里。消融扫描脚本在 [`sweeps/`](sweeps/)。

## 评测

评测使用 [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval)(用 `eval`
可选依赖安装,并设置 `LMMS_EVAL_PATH`)。任务配置在
[`configs/lmms_tasks/`](configs/lmms_tasks/);其中 `*_local.yaml` 是模板——把它们的
`dataset_path` 指向你本地的基准数据副本。

```bash
python scripts/evaluate.py --checkpoint /path/to/checkpoints/token_diversion_x3_fusion/... \
    --data_root /path/to/benchmarks
```

> 注意:所有 lmms-eval 运行都要传 `use_cache=False`。ScienceQA 的指标键名是
> `mc_accuracy,none`。

---

## 仓库结构

```
src/dpvr/
  models/        token_diversion_x3_fusion (DPVR-LF, 本文提出)、
                 token_diversion (DPVR-PC)、token_diversion_substitution (DPVR-KV)、
                 td_fastv,以及基线 (full_ft, lora, fastv)
  training/      训练器、冻结逻辑、回调
  evaluation/    lmms-eval 适配器、原生 runner
  analysis/      KL 散度、CKA、隐状态探针
  profiling/     FLOPs、参数量、显存、GPU 小时
  visualization/ 注意力图、t-SNE
configs/         _base + data / method / eval / lmms_tasks 消融配置
scripts/         train.py、evaluate.py、prepare_data.py、性能分析与分析入口
sweeps/          分流层 / 视觉深度 / 数据规模 消融驱动脚本
tests/           前向一致性、collator、checkpoint 往返
```

---

## 引用

如果你使用了本代码,请引用论文(录用后更新引用信息):

```bibtex
@article{liu2026dpvr,
  title   = {Late-Layer Fusion is Enough: Dual-Path Vision Token Routing for
             Multimodal Large Language Models under Visual Saturation},
  author  = {Liu, Siyuan and Wu, Jinyang},
  journal = {Pattern Recognition (under review)},
  year    = {2026}
}
```

## 许可证

[MIT](LICENSE)。
