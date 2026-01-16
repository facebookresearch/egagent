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


import ast
from collections import OrderedDict
from datetime import datetime, timedelta
import glob
import json
import numpy as np
import os
import pandas as pd
import pysrt
import torch
import base64
import faiss
import re
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import AutoProcessor, AutoModel
from typing import Optional, Any, List, Dict, Literal

model_root = "/source/data/aniketr/models"
retriever = "siglip2-giant-opt-patch16-384"
ckpt = f"{model_root}/{retriever}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

def flatten_list(xss):
    return [x for xs in xss for x in xs]

def get_base64imagelist_from_filepathlist(retrieved_image_paths):
    image_contents = []
    for image_path in retrieved_image_paths:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            })
    return image_contents
    
def get_search_idx_for_selected_video(dataset, frames_dir, selected_video: str):
    import time
    idx_path = f'indexes/{dataset}/{retriever}/{selected_video}_{retriever}.index'
    os.makedirs(f'indexes/{dataset}/{retriever}', exist_ok=True)
    # start = time.time()
    if not os.path.exists(idx_path):
        idx, times, embs = index_frames(dataset, frames_dir, selected_video, device, batch_size=1024)
        faiss.write_index(idx, idx_path)
        # print(f'Index build time: {time.time() - start: .2f}s')
    else:
        idx = faiss.read_index(idx_path)
        times, embs = None, None
        # print(f'Index load from disk time: {time.time() - start: .2f}s')
    return idx, times, embs
    
# Vision-capable GPT‑4.1
def get_vision_llm(llm_name: str):
    vision_llm = ChatOpenAI(
        api_key=APE_API_KEY,
        base_url="https://api.wearables-ape.io/models/v1",
        max_retries=3,
        model=llm_name,
        temperature=0,
        stream_usage=True,
        streaming=False,
        callbacks=None,
    )
    return vision_llm

# e.g. o3, gpt-5
def get_reasoning_llm(llm_name: str):
    vision_llm = ChatOpenAI(
        api_key=APE_API_KEY,
        base_url="https://api.wearables-ape.io/models/v1",
        max_retries=3,
        model=llm_name,        # e.g. "gpt-5" or "gpt-5-chat"
        temperature=0,
        streaming=False,
        callbacks=None,
        reasoning_effort="high"
    )
    return vision_llm

# e.g. gemini-1.5-pro
def get_external_gemini_llm(llm_name: str):
    gemini_llm = ChatGoogleGenerativeAI(
        api_key=GOOGLE_GENAI_API_KEY,
        model=llm_name,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )
    return gemini_llm

def get_vLLM(ip_address: str, llm_name: str):
    qwen_vl_local = ChatOpenAI(
        model=llm_name,
        api_key="EMPTY",
        base_url=f"http://{ip_address}:8000/v1",
        temperature=0,
    )
    return qwen_vl_local

def query_text_only(system_prompt, query, llm_name):
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
            ],
        },
    ]
    if llm_name in ['gpt-5']:
        llm_client = get_reasoning_llm(llm_name)
    else:
        llm_client = get_vision_llm(llm_name)
    out = llm_client.invoke(messages)
    return out

def query_multimodal(system_prompt, query, image_paths, llm_name):
    image_contents = []
    for image_path in image_paths:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            })
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
                *image_contents
            ],
        },
    ]
    if llm_name in ['gpt-5']:
        llm_client = get_reasoning_llm(llm_name)
    else:
        llm_client = get_vision_llm(llm_name)
    out = llm_client.invoke(messages)
    return out

# todo: delete the below and replace with query_multimodal wherever it is used
def query_llm_langchain_multiple_images(system_prompt, query, image_paths, llm_name):
    image_contents = []
    for image_path in image_paths:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"},
            })
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
                *image_contents
            ],
        },
    ]
    llm_client = get_vision_llm(llm_name)
    out = llm_client.invoke(messages)
    return out

def embed_frames_batch(images: list, device, batch_size=256):
    from tqdm import tqdm
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

def embed_texts_batch(texts: list, device, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = processor(text=batch,
                      padding="max_length",
                      truncation=True,
                      max_length=64,
                      return_tensors="pt").to(device)
        # Now inputs includes attention_mask, input_ids, etc.
        with torch.no_grad():
            outputs = model.text_model(
                input_ids=inputs.input_ids,
            )
        # Take CLS token embedding
        embs = outputs.pooler_output.cpu().numpy()
        all_embs.append(embs)
    return np.vstack(all_embs)
    
def get_file_contents(filename):
    """ Given a filename,
        return the contents of that file
    """
    try:
        with open(filename, 'r') as f:
            # It's assumed our file contains a single line,
            # with our API key
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" % filename)

GOOGLE_GENAI_API_KEY = get_file_contents('/home/aniketr/intern_github/aniket/google-genai-key.txt')
OPENAI_API_KEY = get_file_contents('/home/aniketr/intern_github/aniket/openai-api-key.txt')
APE_API_KEY = get_file_contents('/home/aniketr/intern_github/aniket/ape-key-yuliang.txt')

def get_50_frames_from_video(image_dir:str, n_samples=50):
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
    total_images = len(image_files)
    sample_indices = np.linspace(0, total_images - 1, n_samples, dtype=int) # Generate n_samples evenly spaced indices
    return total_images, [f'{image_dir}/{image_files[i]}' for i in sample_indices]

def seconds_to_hhmmss(seconds_str):
    seconds = int(seconds_str)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}{m:02d}{s:02d}"
    
def clean_html_tags(text):
    """Removes HTML tags from the subtitle text."""
    return re.sub(r'<.*?>', '', text).strip()
    
def preprocess_srt_content(subtitles):
    return [
        {
            'start': start,  # Convert to milliseconds
            'end': end,      # Convert to milliseconds
            'subtitle': clean_html_tags(subtitle)
        }
        for start, end, subtitle in subtitles
    ]
    
def load_srt(path):
    subs = pysrt.open(path)
    return [(s.start.ordinal/1000.0, s.end.ordinal/1000.0, s.text.strip()) for s in subs]

def load_srt_hhmmss(path):
    subs = pysrt.open(path)
    results = []
    for s in subs:
        start = str(s.start).split(',')[0]  # e.g. "00:00:00"
        end = str(s.end).split(',')[0]      # e.g. "00:00:02"
        text = clean_html_tags(s.text.strip())
        results.append(f"{start} --> {end} : '{text}'")
    return results

def load_srt_only_text(path):
    subs = pysrt.open(path)
    results = ""
    for s in subs:
        text = clean_html_tags(s.text.strip())
        results += text + "  " 
    return results


import re
from datetime import timedelta
from pathlib import Path

def parse_offset_from_filename(filename: str, include_day: bool = False) -> timedelta:
    """
    Extract offset from filenames like DAY4_11000000.srt.
    Filenames are encoded in centiseconds (HHMMSScc).
    Example: 11000000 -> 11:00:00.00 (cs).
    
    Returns offset as timedelta in milliseconds.
    """
    name = Path(filename).name
    m = re.search(r"DAY(?P<day>\d+)_?(?P<time>\d{8})", name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename {filename} does not match expected format DAY<d>_<HHMMSScc>")

    day = int(m.group("day"))
    digits = m.group("time")

    hh = int(digits[0:2])
    mm = int(digits[2:4])
    ss = int(digits[4:6])
    cs = int(digits[6:8])  # centiseconds

    # Convert to ms
    ms = cs * 10

    base = timedelta(hours=hh, minutes=mm, seconds=ss, milliseconds=ms)
    if include_day:
        base += timedelta(days=(day - 1))
    return base


def shift_srt_file(input_path: str, output_path: str = None, include_day: bool = False) -> str:
    """
    Read an SRT, shift all ' --> ' lines by offset from filename (centisecond-based),
    write output. Returns output path.
    """
    offset = parse_offset_from_filename(input_path, include_day=include_day)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "-->" in line:
            start_str, end_str = [p.strip() for p in line.split("-->")]
            new_start = shift_timestamp(start_str, offset)
            new_end = shift_timestamp(end_str, offset)
            new_lines.append(f"{new_start} --> {new_end}\n")
        else:
            new_lines.append(line)
    return new_lines
    

def search_sql(conn, day, start, end, query_embs, topk, dataset='egolife'):
    """
    conn: SQLite connection
    day: e.g. "day1"
    start, end: ints like 1310, 1320
    query_embs: np.ndarray, shape (N_q, D) or (D,)
    topk: number of top results to keep per query

    Returns: list of shortlisted (path, score) tuples
    """
    
    def blob_to_np(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)
    
    if query_embs.ndim == 1:
        query_embs = query_embs[None, :]
    N_q, D = query_embs.shape

    if dataset == 'videomme':
        start_ts = start
        end_ts = end
    else:
        start_ts = int(f"{start:06d}00")
        end_ts   = int(f"{end:06d}00")
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT rowid, path, embedding FROM frames
        WHERE day = ? AND timestamp BETWEEN ? AND ?
    """, (day, start_ts, end_ts))

    # Collect DB embeddings
    db_rows = cursor.fetchall()
    db_ids, db_paths, db_embs = [], [], []
    for rowid, path, emb_blob in db_rows:
        db_ids.append(rowid)
        db_paths.append(path)
        db_embs.append(blob_to_np(emb_blob))
    if not db_ids:
        return []

    db_embs = np.stack(db_embs)  # (N_db, D)
    N_db = db_embs.shape[0]
    # print(f'query: {query_embs.shape}')
    # print(f'DB: {db_embs.shape}')
    
    # Normalize (so cosine similarity is just dot product)
    query_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    db_norm = db_embs / np.linalg.norm(db_embs, axis=1, keepdims=True)

    sims = query_norm @ db_norm.T  # (N_q, N_db)

    k = min(int(topk), N_db)
    if k <= 0:
        return []
    
    if k == N_db:
        # full sort (descending) -> deterministic ordering
        topk_idx = np.argsort(-sims, axis=1)[:, :k]
    else:
        # faster partial selection when k < N_db
        topk_idx = np.argpartition(-sims, k - 1, axis=1)[:, :k] # (N_q, topk)
    # # Top-k indices for each query
    # topk_idx = np.argpartition(-sims, topk, axis=1)[:, :topk]  

    # Flatten across queries → frequency count
    shortlist = topk_idx.flatten()
    frames, counts = np.unique(shortlist, return_counts=True)
    idx_sorted = np.argsort(counts)[::-1]

    # Final shortlist (frame IDs, sorted by freq)
    final_idx = frames[idx_sorted]

    # Map back to (path, max_score_across_queries)
    final_results = []
    for idx in final_idx:
        path = db_paths[idx]
        score = sims[:, idx].max()  # best similarity across queries
        final_results.append((path, score))

    return final_results


def keep_english_subs(input_path, remove_diarization=True, newlines=None):
    cleaned_lines = []
    skip_chinese = re.compile(r"[\u4e00-\u9fff]")  # regex to detect CJK characters
    diarization_names = ["Jake", "Tasha", "Shure", "Katrina", "Alice", "Lucia", "Nicous", "Choiszt", "Violet", "Jack"]
    
    for line in newlines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip indices (just digits on a line)
        if line.isdigit():
            continue

        # Skip Chinese lines
        if skip_chinese.search(line):
            continue

        if remove_diarization:
            for name in diarization_names:
                if line.startswith(f"{name}:"):
                    line = line[len(name) + 1 :].strip()  # remove "Name:"
                
        # Keep everything else (timestamps, English subtitles with diarization)
        cleaned_lines.append(line)
        
    return cleaned_lines

    
def get_egolife_diarized_transcripts(participants: List[str] = ['A1_JAKE'], remove_diarization=False):
    """
    Return a dictionary of diarized transcripts provided by EgoLife for each day.
    """
    subtitles_dict = {}
    sub_token_count_per_day = []

    for p in participants:
        for day in range(7):
            subs_path = f'/source/data/aniketr/EgoLife/EgoLifeCap/Transcript/{p}/DAY{day+1}'
            srt_files = sorted(glob.glob(os.path.join(subs_path, "*.srt")))
            
            all_texts = []
            for file_path in srt_files:
                timeshifted_subs = shift_srt_file(file_path)
                all_texts.append(str(keep_english_subs(file_path, remove_diarization, newlines=timeshifted_subs)))
            
            # Join with a blank line between files
            subtitles = "\n\n".join(all_texts)
            subtitles_for_day = f"DAY {day+1} SUBTITLES:\n {subtitles}\n\n"
            subtitles_dict[f'DAY{day+1}'] = subtitles_for_day
    
    return subtitles_dict


def get_egolife_transcript_df(participants: List[str] = ['A1_JAKE']):
    transcript_file = f'transcript_csv/diarized_transcripts_all_days_JAKE.csv'
    if os.path.exists(transcript_file):
        return pd.read_csv(transcript_file)
        
    for p in participants:
        dfs = []
        for day in range(7):
            subs_path = f'/source/data/aniketr/EgoLife/EgoLifeCap/Transcript/A1_JAKE/DAY{day+1}'
            srt_files = sorted(glob.glob(os.path.join(subs_path, "*.srt")))
            for file_path in srt_files:        
                df_temp = parse_egolife_srt_to_df(file_path, day=day+1)
                dfs.append(df_temp)

    # Combine all into one DataFrame
    df_transcript_all_days = pd.concat(dfs, ignore_index=True)
    df_transcript_all_days['start_t'] = df_transcript_all_days['start_t'].str.replace(r',\d{1,3}', '', regex=True) # remove milliseconds
    df_transcript_all_days['end_t'] = df_transcript_all_days['end_t'].str.replace(r',\d{1,3}', '', regex=True)
    df_transcript_all_days.to_csv(transcript_file, index=False)
    
    return df_transcript_all_days


def get_videomme_transcript_df(selected_video):
    transcript_file = f'transcript_csv/videomme/transcript-{selected_video}.csv'
    if os.path.exists(transcript_file):
        return pd.read_csv(transcript_file)
        
    path = f'/source/data/video-mme/subtitle/{selected_video}.srt'
    if not os.path.exists(path):
        return pd.DataFrame()
        
    subs = pysrt.open(path)
    results = []
    for s in subs:
        start = str(s.start).split(',')[0]  # e.g. "00:00:00"
        end = str(s.end).split(',')[0]      # e.g. "00:00:02"
        text = clean_html_tags(s.text.strip())
        results.append({
            'day': 0,
            'start_t': start,
            'end_t': end,
            'transcript_english': text
        })
    tdf = pd.DataFrame(results)
    tdf.to_csv(transcript_file, index=False)
    return tdf


def get_videomme_transcripts_for_vid(selected_video):
    df_transcript = get_videomme_transcript_df(selected_video)
    eng_transcripts_to_search = list(df_transcript['transcript_english'].values)
    return eng_transcripts_to_search, df_transcript


timeformatter = lambda s : f"{s[0:2]}:{s[2:4]}:{s[4:6]},{s[6:8]}"

def get_egolife_transcripts_for_qid(query_time):
    df_transcript_all_days = get_egolife_transcript_df()

    current_day = int(query_time['date'][3])
    all_previous_days = [i+1 for i in range(current_day - 1)]
    
    # take all times on previous days
    df_previous_days = df_transcript_all_days.loc[df_transcript_all_days['day'].isin(all_previous_days)]
    
    # take only up to query_time on current day
    cutoff_time = timeformatter(query_time['time'])[:-3]
    cutoff_dt = pd.to_datetime(cutoff_time, format="%H:%M:%S").time()
    df_transcript_all_days['start_t_dt'] = pd.to_datetime(df_transcript_all_days['start_t'], format="%H:%M:%S").dt.time
    df_current_day = df_transcript_all_days[
        (df_transcript_all_days['day'] == current_day) &
        (df_transcript_all_days['start_t_dt'] < cutoff_dt)
    ]
    df_final = pd.concat([df_previous_days, df_current_day], ignore_index=True)
    eng_transcripts_to_search = list(df_final['transcript_english'].values)

    return eng_transcripts_to_search, df_final

    
def parse_egolife_srt_to_df(srt_path, day=1):
    offset = parse_offset_from_filename(srt_path)
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Split entries by double newlines (each subtitle block)
    entries = re.split(r'\n\s*\n', content)

    data = []
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue
        
        # Time line is always the second line
        time_line = lines[1]
        match = re.match(
            r'(\d{2}:\d{2}:\d{2},\d{1,3}) --> (\d{2}:\d{2}:\d{2},\d{1,3})',
            time_line
        )
        if not match:
            continue
        
        start_t, end_t = match.groups()
        
        # Remaining lines contain the transcripts (Chinese + English)
        chinese_line = lines[2] if len(lines) > 2 else ''
        english_line = lines[3] if len(lines) > 3 else ''
        data.append({
            'day': day,
            'start_t': shift_timestamp(start_t, offset),
            'end_t': shift_timestamp(end_t, offset),
            'transcript_chinese': chinese_line.strip(),
            'transcript_english': english_line.strip()
        })
    
    return pd.DataFrame(data)


def shift_timestamp(ts: str, offset: timedelta) -> str:
    """
    Shift SRT timestamp (HH:MM:SS,mmm) by offset.
    Input is in milliseconds, offset is timedelta (ms).
    """
    h, m, s_ms = ts.strip().split(":")
    s, ms = s_ms.split(",")
    t = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
    new_t = t + offset

    total_seconds = int(new_t.total_seconds())
    new_ms = new_t.microseconds // 1000
    new_h = total_seconds // 3600
    new_m = (total_seconds % 3600) // 60
    new_s = total_seconds % 60
    return f"{new_h:02}:{new_m:02}:{new_s:02},{new_ms:03}"
    

def load_content(content_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a string that may be:
    1) Proper JSON object with "response": […]
    2) A list of dicts (Python-style literal)
    Returns a normalized list of dicts, or None on failure.
    """
    # Try JSON first
    try:
        parsed = json.loads(content_str)
    except json.JSONDecodeError:
        # Fallback to Python literal evaluation
        try:
            parsed = ast.literal_eval(content_str)
        except (ValueError, SyntaxError):
            return None

    # Normalize into list of dicts
    if isinstance(parsed, dict) and "response" in parsed and isinstance(parsed["response"], list):
        return parsed["response"]
    elif isinstance(parsed, dict) and "mcq_prediction" in parsed:
        return [parsed]
    elif isinstance(parsed, list):
        return parsed
    else:
        return None
        
def extract_mcq_prediction(content_str: str) -> Optional[str]:
    items = load_content(content_str)
    if not items:
        return None
    first = items[0]
    if isinstance(first, dict):
        return first.get("mcq_prediction")
    return None


# load and clean up egolife_qa time formatting inconsistencies
def load_egolife_qa_jake():
    with open("/source/data/aniketr/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json", "r", encoding="utf-8") as f:
        egolife_qa_jake = json.load(f)

    def split_entry(entry):
        """
        Split a single entry's 'time' string into separate {date, time:list} dicts.
        Each numeric token is assigned to the current day. A 'DAYx' token changes
        the current day for subsequent timestamps. The initial current day is
        entry.get('date').
        """
        s = entry["time"]
        current_day = entry.get("date")   # fallback initial day
        per_day = OrderedDict()
    
        # iterate tokens in order: either "DAY\d+" or a number (timestamp)
        for m in re.finditer(r"(DAY\d+)|(\d+)", s):
            day_token = m.group(1)
            num_token = m.group(2)
    
            if day_token:
                # change current day
                current_day = day_token
                # ensure key exists so ordering is preserved
                per_day.setdefault(current_day, [])
            else:
                ts = num_token
                if current_day is None:
                    raise ValueError(
                        f"No day to attach timestamp {ts} in '{s}' (no fallback 'date' provided)"
                    )
                per_day.setdefault(current_day, []).append(ts)
    
        # return a list of dicts for this single original dict
        return [{"date": day, "time_list": times} for day, times in per_day.items()]

    def convert_singlet_to_list(entry):
        return [{
            "date": entry["date"],
            "time_list": [entry["time"]]
        }]
    
    multiple_target_time = []
    for e in egolife_qa_jake:
        tt = e['target_time']
        if isinstance(tt, dict):
            if'time' not in tt.keys():
                e['target_time'] = [tt]
            else:
                if len(tt['time']) > 8:
                    e['target_time'] = split_entry(tt)
                else:
                    e['target_time'] = convert_singlet_to_list(tt)
    return egolife_qa_jake

# merge results of egagent batches to single json from config
def merge_results(config, agent_backbone):
    merged_json = []
    for batch_json in glob.glob(f'egagent/{config}_start*'):
        with open(batch_json, 'r') as file:
            try:
                data = json.load(file)
                merged_json.extend(data)  # Add the list of dictionaries to merged_data
            except Exception as e:
                print(batch_json)
                print(e)
    if len(merged_json) > 0:
        with open(f'egolife_results/agent_{agent_backbone}/{config}.json', 'w') as output_file:
            json.dump(merged_json, output_file, ensure_ascii=False, indent=4)