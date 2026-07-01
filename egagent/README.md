## EGAgent Inference

<p align="center">
<img src="./../figs/egagent_pipeline.png" width="1024"/>
</p>

First, update the following values in [langgraph_agent.py](langgraph_agent.py)
1. `dataset`: `videomme` or `egolife`
2. `agent_backbone`: e.g. `gpt-4.1` or `gemini-2.5-pro`

The planner decomposes each question into steps and selects which search tools to run per step (`visual`, `audio`, `eg`). After each step, a grader may early-exit to answer generation only if it is very confident the working memory already has all the relevant information needed to answer.

To run EGAgent on EgoLifeQA (500 MCQ pairs):
```
python egagent/run_egagent_on_egolife.py --tscript-search llm
```

Optional flags:
- `--tscript-search {llm,bm25}`: transcript retrieval backend used inside the agent graph
- `--remove-diarization`: strip diarization tags from EgoLife transcripts
- `--example-id N`: run a single EgoLifeQA question by ID (for debugging)
- `--verbose`: print state after each graph node (use with `--example-id`)

Debug one example and save the trace:
```
python egagent/run_egagent_on_egolife.py --example-id 1 --verbose --tscript-search bm25 2>&1 | tee egagent_trace.txt
```

To run EGAgent on VideoMME-long (900 MCQ pairs):
```
python egagent/run_egagent_on_videomme.py --tscript-search llm
```

Debug one VideoMME-long example (set `dataset = 'videomme'` in `langgraph_agent.py` first):
```
python egagent/run_egagent_on_videomme.py --example-id 601-1 --verbose --tscript-search bm25 2>&1 | tee egagent_videomme_trace.txt
```

## [Optional] Batch + Merge results
If using batching, once inference is complete on all batches, merge results to a single json:

```
from utils import merge_batched_results

dataset = 'egolife' # egolife, videomme
agent_backbone = 'gpt-4.1' # gpt-4.1, gemini-2.5-pro, qwen-2.5-vl-7b
config = f'{dataset}_agentic-{agent_backbone}_visual+entitygraph-dtonly-and-dtcaptionfuse+dt-llmsearch_results'

merge_batched_results(config, agent_backbone)
```