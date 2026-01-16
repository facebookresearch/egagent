## Sample Videos at 1 FPS
First, we sample image frames from the raw long videos at 1 FPS for captioning and entity graph creation.
```
python sample_egolife_1fps.py
```

## Fuse Audio Transcripts and Captions for Entity Graph Extraction

<p align="center">
<img src="./../figs/eg_creation.png" width="512"/>
</p>

First update the `dataset` and `mllm` values in `summarize_and_fuse_captions.py`. By default we use GPT-4.1 as the multimodal LLM to fuse captions and audio transcripts.

To fuse transcripts and captions on videomme-long with LlaVA-Video 7B:
```
python summarize_and_fuse_captions.py 0
```
To fuse transcripts and captions on day 4 of egolife:
```
python summarize_and_fuse_captions.py 4
```

## Create Entity Graph from Fused Transcript + Caption
First update the `dataset` value in `create_entity_graph.py`.

To create entity graphs on all 300 videos in videomme-long:
```
python create_entity_graph.py 0
```
To create entity graph on day 6 of egolife:
```
python create_entity_graph.py 6
```

## Prepare Visual Database
We now embed the frames sampled at 1 fps with a strong vision encoder (we use SigLIP 2), and add these embeddings to a table for EGAgent to query during inference. First, update the `model_root` and `dataset_root` values in `create_db_visual_frames.py`, and optionally change the embedding model `retriever` to one of your choice.
```
python create_db_visual_frames.py
```