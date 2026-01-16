# [Agentic Very Long Video Understanding]()
_Aniket Rege, Arka Sadhu, Yuliang Li, Kejie Li, Ramya Korlakai Vinayak, Yuning Chai, Yong Jae Lee, Hyo Jin Kim_

<p align="center">
<img src="./figs/egagent_teaser.png" width="768"/>
</p>

Here we provide code for our agentic framework for very long video understanding powered by entity scene graphs, EGAgent. EGAgent consists of a planning agent equipped for multi-hop cross-modal reasoning by querying three tools: a visual search tool, an audio transcript search tool, and an entity graph search tool. We structure this repository as follows:
1. Create Data Sources for Tool Querying
2. Agent Inference
3. Baselines and Evaluation

## Installation
Install prerequisite packages with conda.
```
conda env create -f environment.yml
conda activate egagent
```

## Create Data Sources for Tool Querying
We create data sources for the visual search tool and entity graph in `prepare_datasources/`. The audio transcripts are queried on the fly and do not require an explicit data source.

## EGAgent Inference
We provide code for EGAgent inference on EgoLife and Video-MME in `egagent/`.

## EGAgent Inference
We provide code to evaluate simple baselines on very long video understanding, i.e. multimodal LLMs that uniformly sample frames and transcripts in `baselines/`.

## Citation
If you find this project useful in your research, please consider citing:
```
@misc{rege2025agentic,
  title     = {Agentic Very Long Video Understanding},
  author    = {Rege, Aniket and Sadhu, Arka and Li, Yuliang and Li, Kejie and Vinayak, Ramya Korlakai and Chai, Yuning and Lee, Yong Jae and Kim, Hyo Jin},
  month     = {December},
  year      = {2025},
}
```