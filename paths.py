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
REPO_ROOT = Path(__file__).resolve().parent

# Dataset roots
EGOLIFE_ROOT = "path/to/EgoLife"  # path to EgoLife dataset root
VIDEO_MME_ROOT = "path/to/VideoMME"  # path to VideoMME dataset root
MODEL_ROOT = "path/to/models"  # path containing embedding model checkpoints

# API key file locations
GOOGLE_GENAI_KEY_PATH = "path/to/google-genai-key.txt"
OPENAI_API_KEY_PATH = "path/to/openai-api-key.txt"

# Captions and transcripts
EGOLIFE_CAPTION_ROOT = "path/to/egolife/captions"
RAW_CAPTION_ROOT = "path/to/captions"  # egolife and videomme captions root
VMME_ASR_DIR = "path/to/video-mme/subtitle" 

# Precomputed embeddings
VMME_EMBS_PATH = "path/to/videomme_embeddings" # path to .npy files of embeddings of each video in Video-MME (Long)

# Original EgoLife dataset directory used for sampling 1 fps frames in prepare_datasources/sample_egolife_1fps.py
# Dataset available at https://huggingface.co/datasets/lmms-lab/EgoLife
EGOLIFE_DATA_DIR = "path/to/EgoLife"

# Outputs and derived data (repo-relative, do not change)
RESULTS_ROOT = REPO_ROOT / "egolife_results"
DB_ROOT = REPO_ROOT / "dbs"
FRAMES_DB_ROOT = REPO_ROOT / "frames_db"
ENTITYGRAPH_DB_ROOT = REPO_ROOT / "entitygraph_db"
TIMESTAMP_EPISODES_ROOT = REPO_ROOT / "timestamp_episodes"
PROCESSED_CAPTION_ROOT = REPO_ROOT / "captioning"
