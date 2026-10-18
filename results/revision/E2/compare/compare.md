| run | L | image tokens | cos_img | kl_img | kl_txt | resid_img | resid_txt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLaVA-1.5-7B | 32 | 576.0 | — | 21 (0.66) | 22 (0.69) | 30 (0.94) | 30 (0.94) |
| LLaVA-1.5-13B | 40 | 576.0 | — | 32 (0.80) | 25 (0.62) | 38 (0.95) | 38 (0.95) |
| LLaVA-1.6-Mistral-7B | 32 | 2115.836 | — | 29 (0.91) | 26 (0.81) | 1 (0.03) | 30 (0.94) |
| Qwen2.5-VL-7B | 28 | 320.996 | — | 25 (0.89) | 26 (0.93) | 2 (0.07) | 26 (0.93) |
| InternVL3-8B | 28 | 2377.728 | — | 25 (0.89) | 23 (0.82) | 3 (0.11) | 26 (0.93) |

Cells: layer (layer / L). kl_*, resid_*: 50%-of-peak transition; cos_img: first layer from which the median stays ≥ tau.
