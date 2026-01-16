## EGAgent Inference

<p align="center">
<img src="./../figs/egagent_pipeline.png" width="1024"/>
</p>

First, update the following values in `langgraph_agent.py`: 
1. `dataset`: videomme or egolife
2. `agent_backbone`: e.g. GPT 4.1 or Gemini 2.5 Pro
3. `dataset_root`: path containing EgoLife and Video-MME datasets (HuggingFace)

We divide all MCQ pairs into 5 batches to parallelize inference. For EgoLife (500 questions), each batch has 100 pairs, and for Video-MME (300 videos * 3 questions per video), each batch has 180 pairs.

To run EGAgent on EgoLifeQA on question IDs from 100 to 200:
```
python run_egagent_on_egolife.py 100
```

Similarly to run EGAgent on VideoMME-long from question ID 180 to 360:
```
python run_egagent_on_egolife.py 180
```

## Merge results
Once inference is complete on all batches, merge results to a single json:

```
from utils import merge_results

dataset = 'egolife' # egolife, videomme
agent_backbone = 'gpt-4.1' # gpt-4.1, gemini-2.5-pro, qwen-2.5-vl-7b
config = f'{dataset}_agentic-{agent_backbone}_visual+entitygraph-dtonly-and-dtcaptionfuse+dt-llmsearch_results'

merge_results(config, agent_backbone)
```