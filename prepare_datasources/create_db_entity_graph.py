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
Creates a SQLite table 'entity_graph_table' with columns:
  day (int), start_t (int), end_t (int), transcript (text),
  source_id, source_type, target_id, target_type, rel_type
"""

import os
import glob
import json
import re
import sqlite3
from typing import Optional, Tuple, List, Any

from paths import DB_ROOT, ENTITYGRAPH_DB_ROOT, TIMESTAMP_EPISODES_ROOT

# ----------------------------
# Timestamp helper
# ----------------------------
def ts_to_int_centiseconds(ts: Optional[str]) -> Optional[int]:
    """
    Convert "HH:MM:SS,mmm" -> int HHMMSScc (centiseconds).
    Example: "11:09:47,100" -> 11094710
    Returns None if input is None or empty.
    """
    if not ts:
        return None
    m = re.match(r'\s*(\d{1,2}):(\d{2}):(\d{2}),(\d{1,3})\s*$', ts)
    if not m:
        raise ValueError(f"Bad timestamp format: {ts!r}")
    hh, mm, ss, ms = m.groups()
    # Make ms 3 digits (pad right with zeros), then convert to centiseconds (ms // 10)
    ms_full = int(ms.ljust(3, '0'))
    cc = ms_full // 10  # 0..99
    # Build fixed-width string to avoid arithmetic mistakes: HH MM SS CC each 2 digits except HH still 2
    s = f"{int(hh):02d}{int(mm):02d}{int(ss):02d}{cc:02d}"
    return int(s)

# ----------------------------
# Filename day extractor
# ----------------------------
def extract_day_from_filename(fname: str) -> Optional[int]:
    """
    Try patterns like day{d}_hour{h}.json or day{d}.json and return integer day.
    If none found, return None.
    """
    b = os.path.basename(fname)
    m = re.search(r'day[_\-]?(\d+)', b, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def build_db_videomme(json_folder: str, dataset: str, config: str) -> None:
    """Build a SQLite DB from all JSON files in json_folder."""
    json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in folder: {json_folder}")
    print(f'Building DB for {len(json_files)}')
          
    for jf in json_files:
        video_id = jf.split("/")[-1][:-5] # remove .json
        db_path = ENTITYGRAPH_DB_ROOT / f"{dataset}/{config}/videomme_{video_id}.db"
        
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
    
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entity_graph_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            start_t TIMESTAMP,
            end_t TIMESTAMP,
            transcript TEXT,
            source_id TEXT,
            source_type TEXT,
            target_id TEXT,
            target_type TEXT,
            rel_type TEXT
        )
        """)
    
        insert_sql = """
        INSERT INTO entity_graph_table
          (video_id, start_t, end_t, transcript, source_id, source_type, target_id, target_type, rel_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        with open(jf, 'r', encoding='utf-8') as fh:
            try:
                data = json.load(fh)
            except Exception as e:
                print(f"Skipping {jf}: JSON parse error: {e}")
                continue

        # tolerate either top-level list or dict with key 'relationships'
        if isinstance(data, dict) and isinstance(data.get('relationships'), list):
            rels = data['relationships']
        elif isinstance(data, list):
            rels = data
        else:
            print(f"Skipping {jf}: unexpected JSON structure")
            continue

        total = 0
        for rel in rels:
            # allow relationship to have a 'day' field that overrides filename
            source_id = rel.get('source_id')
            source_type = rel.get('source_type')
            target_id = rel.get('target_id')
            target_type = rel.get('target_type')
            rel_type = rel.get('rel_type')

            intervals = rel.get('intervals') or []
            # skip relationships without intervals (since start/end ints are required)
            if not intervals:
                continue

            for interval in intervals:
                start_s = interval.get('start_t')
                end_s = interval.get('end_t')
                transcript = interval.get('transcript') or interval.get('text') or None

                cur.execute(insert_sql, (
                    video_id, start_s, end_s, transcript,
                    source_id, source_type, target_id, target_type, rel_type
                ))
                total += 1

        conn.commit()
    
        # Indexes for fast agent queries (single-field and combos used often)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON entity_graph_table(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_start ON entity_graph_table(start_t)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_end ON entity_graph_table(end_t)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_video_id_start ON entity_graph_table(video_id, start_t)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON entity_graph_table(source_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_target ON entity_graph_table(target_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_source_type ON entity_graph_table(source_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_target_type ON entity_graph_table(target_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON entity_graph_table(rel_type)")
    
        conn.commit()
        conn.close()
        print(f"Inserted {total} rows into '{db_path}' (table: entity_graph_table)")

# ----------------------------
# DB builder
# ----------------------------
def build_db(json_folder: str, db_path: str) -> None:
    """
    Build SQLite DB from all json files in json_folder.
    The DB table created: entity_graph_table
    """
    json_files = sorted(glob.glob(os.path.join(json_folder, '*.json')))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in folder: {json_folder}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS entity_graph_table (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        day INTEGER,
        start_t INTEGER,
        end_t INTEGER,
        transcript TEXT,
        source_id TEXT,
        source_type TEXT,
        target_id TEXT,
        target_type TEXT,
        rel_type TEXT
    )
    """)

    insert_sql = """
    INSERT INTO entity_graph_table
      (day, start_t, end_t, transcript, source_id, source_type, target_id, target_type, rel_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    total = 0
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as fh:
            try:
                data = json.load(fh)
            except Exception as e:
                print(f"Skipping {jf}: JSON parse error: {e}")
                continue

        day_from_filename = extract_day_from_filename(jf)

        # tolerate either top-level list or dict with key 'relationships'
        if isinstance(data, dict) and isinstance(data.get('relationships'), list):
            rels = data['relationships']
        elif isinstance(data, list):
            rels = data
        else:
            print(f"Skipping {jf}: unexpected JSON structure")
            continue

        for rel in rels:
            # allow relationship to have a 'day' field that overrides filename
            day = rel.get('day', day_from_filename)

            source_id = rel.get('source_id')
            source_type = rel.get('source_type')
            target_id = rel.get('target_id')
            target_type = rel.get('target_type')
            rel_type = rel.get('rel_type')

            intervals = rel.get('intervals') or []
            # skip relationships without intervals (since start/end ints are required)
            if not intervals:
                continue

            for interval in intervals:
                start_s = interval.get('start_t')
                end_s = interval.get('end_t')
                transcript = interval.get('transcript') or interval.get('text') or None

                try:
                    start_i = ts_to_int_centiseconds(start_s) if start_s else None
                    end_i = ts_to_int_centiseconds(end_s) if end_s else None
                except ValueError as e:
                    print(f"Skipping interval in {jf} due to bad timestamp: {e}")
                    continue

                cur.execute(insert_sql, (
                    day, start_i, end_i, transcript,
                    source_id, source_type, target_id, target_type, rel_type
                ))
                total += 1

    conn.commit()

    # Indexes for fast agent queries (single-field and combos used often)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_day ON entity_graph_table(day)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_start ON entity_graph_table(start_t)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_end ON entity_graph_table(end_t)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_day_start ON entity_graph_table(day, start_t)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON entity_graph_table(source_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_target ON entity_graph_table(target_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source_type ON entity_graph_table(source_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_target_type ON entity_graph_table(target_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON entity_graph_table(rel_type)")

    conn.commit()
    conn.close()
    print(f"Inserted {total} rows into '{db_path}' (table: entity_graph_table)")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    dataset = 'egolife' # egolife, videomme
    config = 'diarized_transcripts_only' # diarized_transcripts_only, fused_dt_and_gpt-4.1_captions, fused_dt_and_llava-video-7bcaptions
    
    json_folder = TIMESTAMP_EPISODES_ROOT / f"{config}/{dataset}/"
    db_path = DB_ROOT / f"{dataset}/{dataset}_entity_graph_{config}.db"

    print("Building DB...")

    if dataset == 'egolife_jake':
        if not os.path.exists(db_path):
            build_db(json_folder, db_path) 
        else:
            print(f'DB already built! Skipping re-build')
    else:
        build_db_videomme(json_folder, dataset, config)