# Accuracy runs (lmms-eval 0.5.0, use_cache=False)

POPE/MMB/SQA/SEED/BLINK in %, MME as scores. `limit` set = smoke run, not a result.

| experiment | job | POPE | MME-P | MME-C | MMB-EN | MMB-CN | SQA | SEED | BLINK | limit | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | reeval_vanilla7b | 85.36 | 1495.46 | 328.93 | 74.13 | 71.38 | 64.06 | 65.4 | 38.98 |  | none |
| E12 | fullft7b |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/full_ft/llava665k |
| E12 | lf13b_s24_cache_False |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/hsf_lf_13b/llava665k_s24_d1 |
| E12 | lf13b_s24_cache_True |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/hsf_lf_13b/llava665k_s24_d1 |
| E12 | lora7b |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/lora/llava665k_r16 |
| E12 | pc13b_s28 |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/token_diversion/llava665k_13b_split28_d1 |
| E12 | pc7b_s18 |  |  |  |  |  |  |  |  |  | /mnt/dolphinfs/hdd_pool/docker/user/hadoop-speech-dolphinfs/hadoop-speech/users/fengmingkuan/dpvr/checkpoints/legacy/token_diversion/llava665k_split18_d1 |
