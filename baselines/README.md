# Baselines

Here we provide code to send uniformly sampled frames and transcripts to a multimodal LLM (Gemini 2.5 Pro) for VQA, focusing on EgoLifeQA. We also support evaluating "oracles", i.e. using the provided ground-truth annotated (`target_day`, `target_time`) for each question in EgoLifeQA. We use this information to center the discrete timestamps around which we sample frames or transcripts to send to the MLLM (GPT-4.1, Gemini 2.5 Pro).

## Gemini 2.5 Pro Uniform Sampling
We use the Gemini Batch API to evaluate uniformly sampling 3000 frames and audio transcript from all video prior to query time on each question in EgoLifeQA. This script runs the full dataset in one pass. You may optionally parallelize this by batching across EgoLifeQA questions.
```
python gemini_mllm_uniformsample.py
```

## Oracle Baselines
Note: In our paper, we use report results using two multimodal LLMs (mllm): GPT-4.1 and Gemini 2.5 Pro.

Use audio transcript oracle (entire target_day):
```
python baselines.py --mllm gemini-2.5-pro --use_dt --use_dt_oracle
```

Use only previous 3 days of audio transcripts:
```
python baselines.py --mllm gpt-4.1 --use_dt --num_prev_days 3
```

Use only visual oracle (50 frames centered around GT moments):
```
python baselines.py --mllm gemini-2.5-pro --use_visual_oracle
```

Use DT previous 3 days and visual oracle:
```
python baselines.py --mllm gpt-4.1 --use_dt --num_prev_days 3 --use_visual_oracle
```

Use both oracles:
```
python baselines.py --mllm gemini-2.5-pro --use_dt --use_dt_oracle --use_visual_oracle
```

Use only captions:
```
python baselines.py --mllm gpt-4.1 --use_captions --captioner llava-video-7b_summarized
```

## Results
All results are written to `egagent/egolife_results/`.