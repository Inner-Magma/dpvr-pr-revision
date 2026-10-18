# Accuracy runs (lmms-eval 0.5.0, use_cache=False)

POPE/MMB/SQA/SEED/BLINK in %, MME as scores. `limit` set = smoke run, not a result.

| experiment | job | POPE | MME-P | MME-C | MMB-EN | MMB-CN | SQA | SEED | BLINK | limit | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | reeval_vanilla7b | 85.36 | 1495.46 | 328.93 | 74.13 | 71.38 | 64.06 | 65.4 | 38.98 |  | none |
| E12 | fullft7b |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/full_ft/llava665k |
| E12 | lf13b_s24_cache_False | 86.03 | 1536.66 | 270.0 | 76.62 | 73.74 | 68.57 | 67.55 | 40.77 |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/hsf_lf_13b/llava665k_s24_d1 |
| E12 | lf13b_s24_cache_True | 86.03 | 1536.66 | 270.0 | 76.62 | 73.74 | 68.57 | 67.55 | 40.77 |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/hsf_lf_13b/llava665k_s24_d1 |
| E12 | lora7b | 84.71 | 1484.72 | 337.86 | 72.6 | 70.18 | 62.12 | 65.51 | 40.19 |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/lora/llava665k_r16 |
| E12 | pc13b_s28 | 85.43 | 1537.74 | 287.5 | 76.81 | 73.74 | 68.52 | 67.69 | 41.08 |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/token_diversion/llava665k_13b_split28_d1 |
| E12 | pc7b_s18 | 85.19 | 1474.51 | 338.93 | 73.97 | 71.7 | 63.31 | 65.38 | 40.72 |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/token_diversion/llava665k_split18_d1 |
| E3 | vanilla7b | 85.36 | 1489.82 | 343.93 | 74.13 | 71.26 | 63.61 | 65.45 | 39.08 |  | none |
| E3 | vtw7b_s12 | 81.33 | 1436.23 | 313.93 | 73.76 | 70.46 | 63.71 | 64.32 | 37.82 |  | config:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/revision_runs/E3/vtw7b_s12/method_config.json |
| E3 | vtw7b_s8 | 51.49 | 788.04 | 358.57 | 55.72 | 48.19 | 62.32 | 45.87 | 37.3 |  | config:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/revision_runs/E3/vtw7b_s8/method_config.json |
| E4 | freeze7b_s8 | 74.92 | 1334.77 | 305.71 | 71.68 | 68.01 | 63.16 | 62.78 | 38.56 |  | config:/mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/revision_runs/E4/freeze7b_s8/method_config.json |
| E6 | smoke_eval | 80.0 |  |  |  |  | 85.0 |  |  | 20 | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/revision/E6/smoke_s22 |
| E6 | vanilla_eval | 87.44 | 1690.0 | 638.21 | 87.46 | 86.12 | 83.09 | 77.43 | 51.24 |  | none |
