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
Creates a SQLite table 'frames' with columns:
  day(str) timestamp (str), path (str), embedding (blob)
"""

import argparse
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
import sqlite3
import torch
import time
from tqdm import tqdm
from typing import List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import DB_ROOT, VMME_EMBS_PATH, EGOLIFE_ROOT, VIDEO_MME_ROOT
from retrieval_model import device, embed_frames_batch

if not os.path.exists(DB_ROOT):
    os.makedirs(DB_ROOT)

def np_to_blob(array: np.ndarray) -> bytes:
    """Convert a numpy array to a blob."""
    return array.astype(np.float32).tobytes()

def process_egolife_day(
    day_num: int,
    frames_dir: str,
    device: torch.device,
    batch_size: int = 256,
    max_ram_gb: float = 40.0,
) -> None:
    """Process a single day of EgoLife frames and embed them into a SQLite DB.
    
    Args:
        day_num: day number (1-7)
        frames_dir: directory containing the frames
        device: device to use for the embedding model
        batch_size: batch size for the embedding model

    Returns:
        None
    """
    start = time.time()
    db_path = DB_ROOT / f"egolife/egolife_jake_frames_day{day_num}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table = "frames"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            path TEXT,
            embedding BLOB
        )
    """)
    conn.commit()

    # collect frame file names
    files = sorted(fn for fn in os.listdir(frames_dir) if fn.endswith(('.jpg', '.png')))
    if not files:
        print(f"No frames found for DAY{day_num} in {frames_dir}")
        conn.close()
        return

    # Roughly estimate how many frames we can hold in CPU RAM at once.
    # We assume a single RGB image uses ~W*H*3 bytes and add a safety factor.
    sample_path = os.path.join(frames_dir, files[0])
    sample_img = Image.open(sample_path).convert("RGB")
    w, h = sample_img.size
    bytes_per_image = w * h * 3  # RGB bytes
    safety_factor = 1.5  # account for Python/PIL overhead, lists, etc.
    est_bytes_per_frame = bytes_per_image * safety_factor
    max_bytes = max_ram_gb * (1024 ** 3)

    # Ensure at least one frame per chunk, but cap to avoid extremely large chunks.
    max_frames_per_chunk = max(1, int(max_bytes // est_bytes_per_frame))
    max_frames_per_chunk = min(max_frames_per_chunk, len(files))

    print(
        f"DAY{day_num}: estimated ~{est_bytes_per_frame/1e6:.2f} MB per frame; "
        f"using chunk size up to {max_frames_per_chunk} frames to stay under ~{max_ram_gb} GB CPU RAM."
    )

    # Process in chunks so we never have all frames in memory at once.
    num_files = len(files)
    chunk_size = max_frames_per_chunk
    num_chunks = (num_files + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_files)
        chunk_files = files[start_idx:end_idx]

        images, paths, timestamps = [], [], []
        for fn in tqdm(
            chunk_files,
            desc=f"DAY{day_num} loading frames ({chunk_idx+1}/{num_chunks})",
            leave=False,
        ):
            ts = int(os.path.splitext(fn)[0])  # filename -> timestamp
            path = os.path.join(frames_dir, fn)
            im = Image.open(path).convert("RGB")
            images.append(im)
            paths.append(path)
            timestamps.append(ts)

        # batch embeddings for this chunk only
        embs = embed_frames_batch(images, device, batch_size=batch_size).astype("float32")

        # insert
        for ts, path, emb in zip(timestamps, paths, embs):
            cursor.execute(
                f"""
                INSERT INTO {table} (timestamp, path, embedding)
                VALUES (?, ?, ?)
                """,
                (ts, path, np_to_blob(emb)),
            )

        conn.commit()

        # free references for this chunk before moving on
        del images, paths, timestamps, embs

    conn.close()
    print(f"Finished DAY{day_num}, saved to {db_path} in time = {time.time() - start: .2f}sec")

# merge all EgoLife day-wise tables into single table across days
def merge_day_dbs(output_db: str) -> None:
    """Merge all EgoLife day-wise tables into a single table across days."""
    conn_out = sqlite3.connect(output_db)
    cur_out = conn_out.cursor()

    cur_out.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id INTEGER PRIMARY KEY,
            day TEXT,
            timestamp TEXT,
            path TEXT,
            embedding BLOB
        )
    """)

    day_dbs = [DB_ROOT / f"egolife/egolife_jake_frames_day{d}.db" for d in range(1, 8)]
    for day_db in day_dbs:
        day_name = os.path.splitext(os.path.basename(day_db))[0]
        day_label = day_name.split("_")[-1]

        conn_in = sqlite3.connect(day_db)
        cur_in = conn_in.cursor()

        for row in cur_in.execute("SELECT timestamp, path, embedding FROM frames"):
            ts, path, emb = row
            cur_out.execute("INSERT INTO frames (day, timestamp, path, embedding) VALUES (?, ?, ?, ?)",
                            (day_label, ts, path, emb))

        conn_out.commit()
        conn_in.close()
        print(f"Finished merging {str(day_db).split('/')[-1]}")

    conn_out.close()
    print(f"All merged into {output_db}")
    
def process_videomme_video(vidname: str, dataset_root: str) -> None:
    """Process a single video from Video-MME and embed its frames into a SQLite DB."""
    start = time.time()
    frames_dir = f'{dataset_root}/video-mme/video_1fps/{vidname}/'
    db_path = DB_ROOT / f"videomme/videomme_frames_{vidname}.db"
    if os.path.exists(db_path):
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table = "frames"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY,
            day TEXT,
            timestamp TEXT,
            path TEXT,
            embedding BLOB
        )
    """)
    conn.commit()

    # collect frames
    files = sorted(fn for fn in os.listdir(frames_dir) if fn.endswith(('.jpg','.png')))
    paths, timestamps = [], []
    
    for fn in tqdm(files, desc=f"Loading Video {vidname}"):
        frame_num = int(fn.split(".")[0])
        hours = frame_num // 3600
        minutes = (frame_num % 3600) // 60
        seconds = frame_num % 60
        ts = f"{hours:02d}{minutes:02d}{seconds:02d}00"
        path = os.path.join(frames_dir, fn)
        paths.append(path)
        timestamps.append(ts)

    # batch embeddings
    embs = np.load(f'{VMME_EMBS_PATH}/{vidname}_{retriever}.npy').astype('float32')
    assert embs.shape[0] == len(files)
    
    # insert timestamps, filepaths and embeddings into table
    for ts, path, emb in zip(timestamps, paths, embs):
        cursor.execute(f"""
            INSERT INTO {table} (day, timestamp, path, embedding)
            VALUES (?, ?, ?, ?)
        """, ('day1', ts, path, np_to_blob(emb)))

    conn.commit()
    conn.close()
    print(f"Finished Video {vidname}, saved to {db_path} in time = {time.time() - start: .2f}sec")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process EgoLife or VideoMME frames into a SQLite DB with visual embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["egolife", "videomme"],
        default="egolife",
        help="Which dataset to process (default: egolife).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for the embedding model (EgoLife only).",
    )
    parser.add_argument(
        "--max-ram-gb",
        type=float,
        default=40.0,
        help="Approximate maximum CPU RAM to use in GB when loading EgoLife frames.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    """Process the frames of EgoLife and VideoMME and embed them into a SQLite DB."""
    args = parse_args()

    dataset = args.dataset
    batch_size = args.batch_size
    max_ram_gb = args.max_ram_gb

    dataset_root = EGOLIFE_ROOT if dataset == "egolife" else VIDEO_MME_ROOT

    if dataset == "egolife":
        for selected_day in range(1, 8):
            frames_dir = f"{dataset_root}/image_1fps_A1_JAKE/DAY{selected_day}"
            process_egolife_day(
                selected_day,
                frames_dir,
                device,
                batch_size=batch_size,
                max_ram_gb=max_ram_gb,
            )
        
        merge_day_dbs(DB_ROOT / "egolife/egolife_jake_frames.db")

    elif dataset == "videomme":
        df_videomme = json.loads(pd.read_parquet(f"{VIDEO_MME_ROOT}/videomme/test-00000-of-00001.parquet").to_json(orient='records'))
        videomme_long = [e for e in df_videomme if e["duration"] == "long"]
        for v in videomme_long:
            selected_video = v["videoID"]
            process_videomme_video(selected_video, dataset_root)