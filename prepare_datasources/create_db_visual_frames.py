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

import numpy as np
import os
from PIL import Image
import torch
import time
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
import sqlite3

model_root = "" # path to embedding model checkpoints
dataset_root = "" # path to egolife and videomme datasets

# Load embedding model
retriever = "siglip2-giant-opt-patch16-384"
ckpt = f"{model_root}/{retriever}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

def embed_frames_batch(images: list, device, batch_size=256):
    all_embeddings = []
    num_batches = (len(images) + batch_size - 1) // batch_size  # ceiling division
    for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="Embedding batches"):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding="max_length").to(device)
        with torch.no_grad():
            outputs = model.vision_model(pixel_values=inputs.pixel_values)
            embs = outputs.pooler_output.cpu().numpy()
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)

def np_to_blob(array: np.ndarray) -> bytes:
    return array.astype(np.float32).tobytes()

def process_egolife_day(day_num, frames_dir, device, batch_size=256):
    start = time.time()
    db_path = f"dbs/egolife/egolife_jake_frames_day{day_num}.db"
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

    # collect frames
    files = sorted(fn for fn in os.listdir(frames_dir) if fn.endswith(('.jpg','.png')))
    images, paths, timestamps = [], [], []

    for fn in tqdm(files, desc=f"Loading DAY{day_num}"):
        ts = int(os.path.splitext(fn)[0])  # filename -> timestamp
        path = os.path.join(frames_dir, fn)
        im = Image.open(path).convert("RGB")
        images.append(im)
        paths.append(path)
        timestamps.append(ts)

    # batch embeddings
    embs = embed_frames_batch(images, device, batch_size=batch_size).astype('float32')

    # insert
    for ts, path, emb in zip(timestamps, paths, embs):
        cursor.execute(f"""
            INSERT INTO {table} (timestamp, path, embedding)
            VALUES (?, ?, ?)
        """, (ts, path, np_to_blob(emb)))

    conn.commit()
    conn.close()
    print(f"Finished DAY{day_num}, saved to {db_path} in time = {time.time() - start: .2f}sec")

# merge all EgoLife day-wise tables into single table across days
def merge_day_dbs(output_db, day_dbs):
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

    for day_db in day_dbs:
        day_name = os.path.splitext(os.path.basename(day_db))[0]
        day_label = day_name.split("_")[-1]

        print(f"Merging {day_db} → {output_db}")
        conn_in = sqlite3.connect(day_db)
        cur_in = conn_in.cursor()

        for row in cur_in.execute("SELECT timestamp, path, embedding FROM frames"):
            ts, path, emb = row
            cur_out.execute("INSERT INTO frames (day, timestamp, path, embedding) VALUES (?, ?, ?, ?)",
                            (day_label, ts, path, emb))

        conn_out.commit()
        conn_in.close()
        print(f"Finished {day_db}")

    conn_out.close()
    print(f"All merged into {output_db}")
    
def process_videomme_video(vidname):
    start = time.time()
    frames_dir = f'{dataset_root}/video-mme/video_1fps/{selected_video}/'
    db_path = f"frames_db/videomme/videomme_frames_{vidname}.db"
    if os.path.exists(db_path):
        return
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
    embs_path = "" # path to .npy files of embeddings of each video in videomme (long)
    embs = np.load(f'{embs_path}/{vidname}_{retriever}.npy').astype('float32')
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


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    dataset = 'egolife' # egolife, videomme
    
    if dataset == 'egolife':
        for selected_day in range(1, 8):
            frames_dir = f'{dataset_root}/EgoLife/image_1fps_A1_JAKE/DAY{selected_day}'
            process_egolife_day(selected_day, frames_dir, device, batch_size=256)
        day_dbs = [f"dbs/egolife/egolife_jake_frames_day{d}.db" for d in range(1, 8)]
        merge_day_dbs("dbs/egolife/egolife_jake_frames.db", day_dbs)
        
    elif dataset == 'videomme':
        df_videomme = json.loads(pd.read_parquet(f"{dataset_root}/video-mme/videomme/test-00000-of-00001.parquet").to_json(orient='records'))
        videomme_long = [e for e in df_videomme if e['duration'] == 'long']
        for v in videomme_long:
            selected_video = v['videoID']
            process_videomme_video(selected_video)