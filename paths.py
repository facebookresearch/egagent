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
Centralized filesystem paths for the project.
Update these values to match your local setup.
"""

# Path helpers
from pathlib import Path

# Project root (this file lives at repo root)
EGAGENT_ROOT = Path(__file__).resolve().parent

# Dataset roots
EGOLIFE_ROOT = "path/to/EgoLife"  # path to EgoLife dataset root (https://huggingface.co/datasets/lmms-lab/EgoLife)
VIDEO_MME_ROOT = "path/to/VideoMME"  # path to VideoMME dataset root (https://huggingface.co/datasets/lmms-lab/Video-MME)
MODEL_ROOT = EGAGENT_ROOT  # path containing multimodal embedding model

# API key file locations
GOOGLE_GENAI_KEY_PATH = "path/to/google-genai-key.txt"
OPENAI_API_KEY_PATH = "path/to/openai-api-key.txt"

# Captions and transcripts
CAPTION_ROOT = EGAGENT_ROOT / "captions"  # egolife and videomme captions root
VMME_ASR_DIR = VIDEO_MME_ROOT + "subtitle" 

# Precomputed embeddings
VMME_EMBS_PATH = "path/to/videomme_embeddings" # path to .npy files of embeddings of each video in Video-MME (Long)

# Outputs and derived data (repo-relative, do not change)
RESULTS_ROOT = EGAGENT_ROOT / "egolife_results"
DB_ROOT = EGAGENT_ROOT / "dbs"
ENTITYGRAPH_DB_ROOT = EGAGENT_ROOT / "entitygraph_db"
TIMESTAMP_EPISODES_ROOT = EGAGENT_ROOT / "timestamp_episodes"