# PR-D-26-07438 审稿意见（原文，带编号）

- 期刊：Pattern Recognition；决定：**Major Revision**（2026-10-05）
- 修改稿截止：**2026-10-25**（已计划申请延期，结果见 PLAN.md）
- 通讯作者：Jinyang Wu；修改稿须上传源文件（LaTeX）
- 编号 R1.x / R2.x 与 PLAN.md、回复信一致。原文摘自编辑部邮件，未改动。

## Associate Editor

> We appreciate the authors for their submission. The paper makes interesting contribution, while both reviewers raised concerns on (1) claims exceed the evidence; (2) validation covers too few model families; (3) key baseline comparisons are missing; (4) KV-cache limitations slow decoding; (5) robustness and design ablations are incomplete; (6) accuracy and speed reporting are inconsistent; etc. I would encourage the authors to revise their paper accordingly.

## Reviewer #1

- **R1.1** The phrase "vision tokens saturate" is stronger than the current operational evidence warrants. High adjacent-layer cosine similarity does not by itself imply that deeper updates are functionally unnecessary. Expressions such as "deep updates approach a no-op" appear repeatedly, but a cosine similarity above 0.92 can coexist with functionally significant residual updates.
- **R1.2** The title and conclusion use "Late-Layer Fusion is Enough" and "a single fusion layer is enough for perceptual competence in modern MLLMs." The experiments are primarily on LLaVA-1.5-7B/13B, while the manuscript itself admits that LLaVA-Next is only a smoke test and Qwen-VL-family models have not been trained.
- **R1.3** The statement that DPVR-LF "matches or exceeds full fine-tuning across eight standard benchmarks" is not adequately supported by Table 2. The cited FullFT row does not contain results for all eight benchmarks and is numerically above DPVR-LF on some reported metrics such as MME-P and MME-C.
- **R1.4** "Bit-identical" and "numerical noise" should be used more cautiously around the frozen-shallow-stack verification. A median cosine >0.99989 is extremely similar, but it is not literally bit-identical to vanilla.
- **R1.5** Figure 1 emphasizes the −28% latency result, but this specific value comes from the 7B A800 experiment at B=4. Other settings show substantially smaller gains: for example, 13B DPVR-LF at s=24 saves about 14.8% on Blackwell and 23.1% on 5880 Ada, while s=34 saves only about 1.7% and 7.8%, respectively.
- **R1.6** The paper repeatedly refers to 25-30% forward savings, yet Table 6 demonstrates that wall-clock savings depend strongly on split point and hardware.
- **R1.7** The so-called "theoretical prediction −26.8%" is calibrated using the measured A800 result through ρ=0.70. This makes it less convincing as an independent theoretical validation. It is effectively a fitted cost model.
- **R1.8** The paper reports an eight-benchmark evaluation but frequently summarizes performance using a six-benchmark mean, excluding MME-P and MME-C because they use different scales. This is defensible, but the distinction needs to be more prominent in headline plots and claims.
- **R1.9** The strongest practical drawback appears only later in §4.8: DPVR-LF currently cannot reuse the normal KV cache during decoding, producing approximately 188 ms/token versus 35 ms/token for vanilla 13B in Table 7. This is sufficiently important that it should appear near the main efficiency result.
- **R1.10** The main 13B accuracy configuration and efficiency configurations are not fully aligned. Table 3 highlights s=28, while Table 6 profiles s=24 and s=34. This complicates accuracy-efficiency comparison.
- **R1.11** Some tables mix N=3 means, historical single-seed values, and single-seed ablations. Although footnotes disclose this, the presentation is difficult to interpret quickly.
- **R1.12** VTW is particularly close conceptually because it already withdraws visual tokens and performs text-only computation after a shallow stage. The manuscript distinguishes DPVR on trainability and late fusion, but does not provide a direct VTW comparison.
- **R1.13** FastV is included, but only an inference-time token-pruning baseline. A direct comparison to the closest visual-withdrawal architecture would better establish the benefit of the side branch and final fusion.
- **R1.14** The novelty claim is evaluated only in the LLaVA-1.5 architecture family. Without a second independently trained backbone, it is difficult to establish that "modality-asymmetric depth" is a general architectural phenomenon rather than a LLaVA/Vicuna-specific property.
- **R1.15** The claimed advantage over LoRA mixes two concepts: trainable-parameter efficiency and inference compute efficiency. LoRA is not designed to remove inference FLOPs; DPVR changes the compute graph. These should be presented as different axes rather than direct substitutes.

## Reviewer #2

Summary / Strengths（略，见邮件）：认可视觉饱和发现、轻量设计、多维实验。

- **R2.0** Please refer the recent papers: [Suggested references removed by Associate Editor-in-Chief]. [Please include references that are relevant for this work and chosen by the authors.]
- **R2.1** Insufficient comparison with state-of-the-art visual token compression & MLLM acceleration methods. The paper only selects FastV, VTW, LLaVA-PruMerge as baselines, missing recent advanced routing architectures (e.g. layer-wise token skip, dynamic fusion MLLMs). It cannot fully demonstrate the efficiency superiority of late single-fusion design against existing multi-layer cross-modal optimization schemes.
- **R2.2** Limited model generalization verification: Only validated on Vicuna-based LLaVA-1.5, no comprehensive tests on Qwen-VL, LLaVA-Next, Mistral multi-modal and other mainstream backbones. Only a simple compatibility test is mentioned without quantitative accuracy & latency data, unable to prove DPVR-LF's universal adaptability across diverse MLLM frameworks.
- **R2.3** Defect of decoding-stage efficiency: DPVR-LF cannot reuse KV cache during autoregressive decoding, all test speed gains only exist in prefill phase. The paper lacks targeted optimization scheme and auxiliary ablation for cache-adaptive routing, limiting practical deployment value of chat-style multi-modal dialogue scenarios.
- **R2.4** Incomplete robustness & task ablation: Performance drops obviously on multi-image reasoning and cross-lingual benchmarks (BLINK/MMBench-CN). The paper does not design adjustable multi-fusion layer strategies for complex vision-language tasks, nor conducts ablation on varying image resolutions and image token quantities to test saturation layer stability.
- **R2.5** Missing ablation of alternative vision branch structures: Only adopts single-layer transformer as side branch, without comparing multi-layer vision branch, MLP-only branch and other lightweight variants; cannot confirm one-layer structure is the optimal capacity configuration for saturated visual representation extraction.
- **R2.6** Lack joint optimization experiment with orthogonal acceleration techniques: The paper mentions token pruning, knowledge distillation as future work, but does not carry out combined ablation of DPVR-LF + token compression, unable to verify the upper limit of model acceleration after stacking multiple optimization methods.
- **R2.7** Insensitive hyperparameter exploration for saturation split layer: The paper only verifies fixed split intervals for 7B/13B models, does not analyze the correlation between saturation layer position and image token count, backbone depth, pre-training data distribution, cannot provide quantitative split-layer selection guidance for arbitrary MLLM scales.

## EiC 格式与文献要求

- **EiC.a** 参考文献覆盖近两年 SOTA；少引 arXiv，改引正式发表版本；不要 "[1,2,3,4,5,6]" 式打包引用；最终 35–55 条。
- **EiC.b** 引用模式识别领域的近期工作（不限于 Pattern Recognition 期刊）。
- **EiC.c** 双倍行距单栏，正式论文 ≤ 35 页（当前源文件编译为 33 页）。
