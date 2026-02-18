# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Retrieval/embedding model and processor. Import this module only when you need
to run frame or text embedding (e.g. frame_retriever_sql or create_db_visual_frames).
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from paths import MODEL_ROOT

retriever = "siglip2-giant-opt-patch16-384"
ckpt = f"{MODEL_ROOT}/{retriever}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)


def embed_frames_batch(images, device: torch.device = None, batch_size: int = 256) -> np.ndarray:
    """Embed a batch of images using the embedding model.

    Args:
        images: list of PIL Image objects
        device: device to use for the embedding model (default: module-level device)
        batch_size: batch size for the embedding model

    Returns:
        embeddings of the images
    """
    if device is None:
        device = globals()["device"]
    all_embeddings = []
    num_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="Embedding batches"):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=inputs.pixel_values)
            embs = outputs.pooler_output.cpu().numpy()
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)


def embed_texts_batch(texts: list, device: torch.device = None, batch_size: int = 128) -> np.ndarray:
    """Embed a batch of text using the embedding model.

    Args:
        texts: list of text strings
        device: device to use for the embedding model (default: module-level device)
        batch_size: batch size for the embedding model

    Returns:
        embeddings of the text
    """
    if device is None:
        device = globals()["device"]
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = processor(
            text=batch,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model.text_model(input_ids=inputs.input_ids)
            embs = outputs.pooler_output.cpu().numpy()
        all_embs.append(embs)
    return np.vstack(all_embs)
