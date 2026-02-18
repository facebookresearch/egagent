## Visualize category-wise MCQ Accuracy on EgoLifeQA
This replicates Figure 4 from our paper, also shown below.
```
python plot_accuracy.py --out figs/egolifeqa_categorywise_accuracy.png
```
<p align="center">
<img src="../figs/egolifeqa_categorywise_accuracy.png" width="768"/>
</p>

## EGAgent Tool Retrieval Recall
Here we compute tool-wise retrieval recall of EGAgent which searches over the entity graph, visual embedding database, and audio transcripts (EG + F + T) with Gemini 2.5 Pro. We first extract retrieved timestamps from the working memory $\mathcal{M}$ of each tool. We compute recall for each tool with respect to time windows W around timestamps from the final VQA Agent's reasoning trace, i.e. the justification for its predicted answer to the multiple-choice question. We record a 'hit' if any timestamp selected by the search tool lies in the window W. To get recall across W = {10 sec, 30 sec, 1 min, 2 min, 10 min, 1 hour} as reported in Table 6 (Appendix E) of our paper, run:
```
bash "run_all_recall_configs.sh"
```