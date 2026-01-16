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
import asyncio
from datetime import datetime, timedelta
import glob
import json
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
import os
import pandas as pd
from pydantic import BaseModel, parse_obj_as
import pysrt
import re
import sqlite3
import sys
import time
from typing import Literal, List, Dict, Any, Optional

from utils import get_vision_llm, get_egolife_diarized_transcripts

dataset = 'videomme' # videomme, egolife

def get_llm_worker(system_prompt, human_prompt, structured_llm_class, model):
    """
    Pass in custom llm BaseModel with structured output along with system and human prompts.
    Returns llm for use in graph nodes / edges.
    """
    llm = get_vision_llm(model)
    llm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    structured_llm = llm.with_structured_output(structured_llm_class) # pipe llm with prompt
    return llm_prompt | structured_llm

class Interval(BaseModel):
    start_t: str  # e.g. "11:09:47,100"
    end_t: str

class RelationshipOutput(BaseModel):
    """Given an audio transcript with timestamps and a source-to-target relationship, add timestamps to the relationship along with confidence and explanation."""
    relationship_id: int           # must match the input relationship.id
    intervals: List[Interval]      # one or more supporting utterance spans
    confidence: float
    explanation: Optional[str] = None

class RelationshipsOutput(BaseModel):
    relationships: List[RelationshipOutput]


def get_rel_timestamper_llm(model, config):
    rel_timestamper_system = """
    You are a helpful assistant that adds timestamps to relationships between graph nodes. Output only valid JSON, no prose.
    """
    
    rel_timestamper_user_dtonly = """I will give you:
    1) a list of relationships: each relationship has relationship_id, source_id node, target_id node, and relationship type.
    2) a list of transcripts: [t1, d1, t2, d2, ...] where each t = 'start_t --> end_t' and each d = 'speaker_name: transcript'.
    If no speaker_name is provided, ignore it.
    
    For each relationship, find all utterances that support it.
    
    Relationships : {relationships}
    Transcripts: {transcripts}
    
    Rules:
    - Use only timestamps already present in transcript utterances.
    - If no supporting utterances exist, return intervals: [] and confidence: 0.0.
    - Output an array of RelationshipOutput objects in the same order as the input relationships.
    """

    rel_timestamper_user_caption = """I will give you:
    1) a list of relationships: each relationship has relationship_id, source_id node, target_id node, and relationship type.
    2) a caption containing dialogue: the caption contains a timestamp (start_t --> end_t) and  visual information from the scene as well as information on spoken dialogue.
    3) a list of transcripts [t1, t2, ...], each containing a 'timestamp' (start_t --> end_t) and 'text' containing spoken dialogue.
    
    For every single provided relationship, find all transcripts and captions that support it.
    
    Relationships : {relationships}
    Captions: {captions}
    Transcripts: {transcripts}
    
    Rules:
    - First, try to use only timestamps already present in transcript utterances.
    - If no supporting utterances exist, use the entire interval from the caption as start_t and end_t.
    - Output an array of RelationshipOutput objects in the same order as the input relationships.
    """
    
    if config == 'diarized_transcripts_only':
        return get_llm_worker(rel_timestamper_system, rel_timestamper_user_dtonly, RelationshipsOutput, model)
    else:
        return get_llm_worker(rel_timestamper_system, rel_timestamper_user_caption, RelationshipsOutput, model)

async def generate_graph_for_hour(text:str):
    allowed_nodes = ["Person", "Location", "Object"]
    allowed_relationships = ["TALKS_TO", "INTERACTS_WITH", "MENTIONS", "USES"]
    llm = get_vision_llm('gpt-4.1')
    documents = [Document(page_content=str(text))]
    props_defined = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )
    graph_documents = await props_defined.aconvert_to_graph_documents(documents)
    return graph_documents

def parse_relationships(graph_documents):
    relationships_parsed = []
    for i, rel in enumerate(graph_documents[0].relationships):
        r = {}
        r['rel_id'] = i+1
        r['source_id'] = rel.source.id
        r['source_type'] = rel.source.type
        r['target_id'] = rel.target.id
        r['target_type'] = rel.target.type
        r['rel_type'] = rel.type
        relationships_parsed.append(r)
    return relationships_parsed

def clean_html_tags(text):
        """Removes HTML tags from the subtitle text."""
        return re.sub(r'<.*?>', '', text).strip()

def load_srt_formatted(path):
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


# add egolife raw transcript to graph
def diarized_list_to_dict(lst):
    results = []
    i = 0
    while i < len(lst):
        # Extract time range
        time_range = lst[i]
        start, end = [t.strip() for t in time_range.split('-->')]
        
        # Peek ahead to find how many utterances follow this time range
        # At least one utterance is guaranteed, maybe two
        utterances = []
        j = i + 1
        while j < len(lst) and '-->' not in lst[j]:
            utterances.append(lst[j])
            j += 1
        
        # Keep the last utterance (if two are present)
        if utterances:
            chosen = utterances[-1]
            results.append({
                "start": start,
                "end": end,
                "transcript": chosen
            })
        
        # Move to the next time range
        i = j
    
    return results

def time_to_ms(t):
    # Convert 'HH:MM:SS,mmm' to milliseconds
    h, m, s = t.split(':')
    s, ms = s.split(',')
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)
    
def add_dtranscripts_to_rel_dict(dt_hour, rel_with_timestamps):
    raw_dt_dict = diarized_list_to_dict(ast.literal_eval(dt_hour))
    # Precompute timestamps in ms
    dt_times = [(time_to_ms(d['start']), d['transcript']) for d in raw_dt_dict]
    
    # For each interval, find closest dt entry
    for rel in rel_with_timestamps:
        for interval in rel['intervals']:
            interval_ms = time_to_ms(interval['start_t'])
            # find closest
            closest = min(dt_times, key=lambda x: abs(x[0] - interval_ms))
            interval['transcript'] = closest[1]
    return rel_with_timestamps


# add videomme raw subtitles to graph
def time_to_seconds(t):
    """Convert HH:MM:SS string to total seconds."""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def parse_subtitle_entry(entry):
    """Parse an entry like '00:00:00 --> 00:00:03 : text'."""
    parts = entry.split(" --> ")
    start_t = parts[0].strip()
    end_t, text = parts[1].split(" : ", 1)
    end_t = end_t.strip()
    text = text.strip().strip("'\"")
    return {
        "start_t": start_t,
        "end_t": end_t,
        "text": text
    }

def get_transcripts_for_interval(start_t, end_t, subtitles):
    """Return all subtitles overlapping with [start_t, end_t]."""
    start_s = time_to_seconds(start_t)
    end_s = time_to_seconds(end_t)
    collected = []

    for sub in subtitles:
        sub_start = time_to_seconds(sub["start_t"])
        sub_end = time_to_seconds(sub["end_t"])

        # Check for overlap
        if sub_end >= start_s and sub_start <= end_s:
            collected.append(sub["text"])

    return " ".join(collected) if collected else ""


def attach_transcripts_to_videomme_graph(scene_graph, subtitles_with_timestamps):
    """Attach transcripts to each interval in the scene graph."""
    # Parse subtitles into structured list

    if subtitles_with_timestamps == "":
        for rel in scene_graph:
            for interval in rel["intervals"]:
                interval["transcript"] = ""
        return scene_graph
    
    subtitles = [parse_subtitle_entry(s) for s in subtitles_with_timestamps]

    # Process each relationship
    for rel in scene_graph:
        for interval in rel["intervals"]:
            interval["transcript"] = get_transcripts_for_interval(
                interval["start_t"],
                interval["end_t"],
                subtitles
            )
    return scene_graph

def extract_time_from_path(path_str):
    # Example filename: DAY1_A1_JAKE_11094208.mp4
    match = re.search(r"_(\d{8})\.mp4$", path_str)
    if not match:
        return None
    hh, mm, ss = match.group(1)[:2], match.group(1)[2:4], match.group(1)[4:6]
    return f"{hh}:{mm}:{ss}"

def seconds_to_hhmmss(seconds_str):
    seconds = int(seconds_str)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}{m:02d}{s:02d}"
    
def add_start_and_end_times_to_egolife_captions(context_for_hour):
    for entry in context_for_hour:
        for path, caption in entry.items():
            start_time = extract_time_from_path(path)
            et = datetime.strptime(start_time, '%H:%M:%S') +timedelta(seconds=30)
            end_time = datetime.strftime(et, '%H:%M:%S,%f')[:-3]
            entry[path] = f"{start_time + ',000'} --> {end_time} : {caption}"
    return context_for_hour

def add_start_and_end_times_to_videomme_captions(context_for_hour):
    window_duration = 64 # each window captions 64 seconds
    for entry in context_for_hour:
        for w_num, (window, caption_content) in enumerate(entry.items()):
            start_t = seconds_to_hhmmss(w_num * window_duration)
            end_t = seconds_to_hhmmss((w_num+1) * window_duration)
            entry[window] = f"Caption start time: {start_t}, Caption end time: {end_t}, Caption content: '{caption_content}'"
    return context_for_hour


async def extract_entity_graph_egolife_day(config, day:int = 1, captioner = 'gpt-4.1'):
    diarized_transcripts_dict = get_egolife_diarized_transcripts()
    dt_for_day = diarized_transcripts_dict[f'DAY{day}'].split("\n")
    episodes = [dt_for_day[i] for i in range(len(dt_for_day)) if i%2!=0][:-1] # each hour
    rel_timestamper = get_rel_timestamper_llm(model='gpt-4.1', config=config)
    
    for hour in range(len(episodes)):
        ofilename = f'timestamp_episodes/{config}/egolife/day{day}_hour{hour+1}.json'
        rel_outfile = f'timestamp_episodes/{config}/egolife/relationships/day{day}_hour{hour+1}_relationships.json'
        if os.path.exists(ofilename):
            continue

        if config == f'fused_dt_and_{captioner}captions':
            with open(f'captioning/fused_dt_and_{captioner}captions/{captioner}_day{day}_hour{hour+1}.json', "r") as f:
                context_for_hour = add_start_and_end_times_to_egolife_captions(json.load(f))
        elif config == 'diarized_transcripts_only':
            context_for_hour = episodes[hour]
        else:
            raise ValueError('Invalid input data config')
        
        print(f'GRAPH_EXTRACT: Extracting graph from DAY{day}, HOUR{hour+1}')
        start = time.time()
        if not os.path.exists(rel_outfile):
            graph_documents = await generate_graph_for_hour(context_for_hour)
            print(f'GRAPH_EXTRACT: took {time.time() - start: .2f} sec')
            relationships_parsed = parse_relationships(graph_documents)
            with open(rel_outfile, "w") as f:
                json.dump(relationships_parsed, f, indent=4) # delete these later
        else:
            with open(rel_outfile, "r") as f:
                relationships_parsed = json.load(f)
        rel_lookup = {r["rel_id"]: r for r in relationships_parsed}

        print(f'ADD_TSTAMP: starting')
        while not os.path.exists(ofilename):
            try:
                if config == f'fused_dt_and_{captioner}captions':
                    rel_with_timestamps = rel_timestamper.invoke({"relationships": relationships_parsed, "transcripts": episodes[hour], "captions": context_for_hour})
                elif config == 'diarized_transcripts_only':
                    rel_with_timestamps = rel_timestamper.invoke({"relationships": relationships_parsed, "transcripts": context_for_hour})
                rel_timestamped_dicts = [item.model_dump() for item in rel_with_timestamps.relationships]
                print(f'ADD_TSTAMP: took {time.time() - start: .2f} sec')
                for entry in rel_timestamped_dicts:
                    rel_id = entry["relationship_id"]
                    if rel_id in rel_lookup:
                        rel_info = rel_lookup[rel_id]
                        entry["source_id"] = rel_info["source_id"]
                        entry["source_type"] = rel_info["source_type"]
                        entry["target_id"] = rel_info["target_id"]
                        entry["target_type"] = rel_info["target_type"]
                        entry["rel_type"] = rel_info["rel_type"]
                rel_timestamped_dicts = add_dtranscripts_to_rel_dict(episodes[hour], rel_timestamped_dicts)
                with open(ofilename, "w") as f:
                    json.dump(rel_timestamped_dicts, f, indent=4)
                print(f'DONE: saved {ofilename}\n')
            except Exception as e:
                print(e)
                print('Error parsing, retrying')


async def extract_entity_graph_videomme(config, selected_video, captioner):
    rel_timestamper = get_rel_timestamper_llm(model='gpt-4.1', config=config)
    asr_dir = '/source/data/video-mme/subtitle'

    subtitles_only_text = load_srt_only_text(f'{asr_dir}/{selected_video}.srt') if os.path.exists(f'{asr_dir}/{selected_video}.srt') else ""
    subtitles_with_timestamps = load_srt_formatted(f'{asr_dir}/{selected_video}.srt') if os.path.exists(f'{asr_dir}/{selected_video}.srt') else ""

    ofilename = f'timestamp_episodes/{config}/videomme/{selected_video}.json'
    rel_outfile = f'timestamp_episodes/{config}/videomme/relationships/{selected_video}_relationships.json'
    if config == f'fused_dt_and_{captioner}captions':
        with open(f'captioning/fused_dt_and_{captioner}captions/gpt-4.1_{selected_video}.json', "r") as f:
            context_for_hour = add_start_and_end_times_to_videomme_captions(json.load(f))
    elif config == 'diarized_transcripts_only':
        context_for_hour = subtitles_only_text
    else:
        raise ValueError('Invalid input data config')
    
    if not os.path.exists(rel_outfile):
        print(f'Generating Graph for {selected_video}')
        start = time.time()
        graph_documents = await generate_graph_for_hour(context_for_hour)
        print(f'GRAPH_EXTRACT: took {time.time() - start: .2f} sec')
        relationships_parsed = parse_relationships(graph_documents)
        with open(rel_outfile, "w") as f:
            json.dump(relationships_parsed, f, indent=4) # delete these later
    else:
        with open(rel_outfile, "r") as f:
            relationships_parsed = json.load(f)
        print(f'Loaded {len(relationships_parsed)} relationships')
    rel_lookup = {r["rel_id"]: r for r in relationships_parsed}
    
    start = time.time()
    while not os.path.exists(ofilename):
        print(f'Adding Timestamps for {selected_video}')
        if config == f'fused_dt_and_{captioner}captions':
                rel_with_timestamps = rel_timestamper.invoke({"relationships": relationships_parsed, "transcripts": subtitles_with_timestamps, "captions": context_for_hour})
        elif config == 'diarized_transcripts_only':
            rel_with_timestamps = rel_timestamper.invoke({"relationships": relationships_parsed, "transcripts": subtitles_with_timestamps})
        rel_timestamped_dicts = [item.model_dump() for item in rel_with_timestamps.relationships]
        print(f'ADD_TSTAMP: took {time.time() - start: .2f} sec')
        for entry in rel_timestamped_dicts:
            rel_id = entry["relationship_id"]
            if rel_id in rel_lookup:
                rel_info = rel_lookup[rel_id]
                entry["source_id"] = rel_info["source_id"]
                entry["source_type"] = rel_info["source_type"]
                entry["target_id"] = rel_info["target_id"]
                entry["target_type"] = rel_info["target_type"]
                entry["rel_type"] = rel_info["rel_type"]
        updated_graph = attach_transcripts_to_videomme_graph(rel_timestamped_dicts, subtitles_with_timestamps)
        with open(ofilename, "w") as f:
            json.dump(updated_graph, f, indent=4)
        print(f'DONE: saved {ofilename}\n')
        
    
async def main(day):
    if dataset == 'egolife':
        captioner='gpt-4.1'
        config = f'fused_dt_and_{captioner}captions'
        await extract_entity_graph_egolife_day(config, day, captioner)
    else:
        captioner='llava-video-7b'
        config = f'fused_dt_and_{captioner}captions'
        df_videomme = json.loads(pd.read_parquet("/source/data/video-mme/videomme/test-00000-of-00001.parquet").to_json(orient='records'))
        df_videomme_long_vIDs = np.unique([e['videoID'] for e in df_videomme if e['duration'] == 'long'])
        batch_offset = 50 # 6 batches of 50 each = 300 long videos
        start_idx = int(sys.argv[1])
        end_idx = start_idx + batch_offset
        for selected_video in df_videomme_long_vIDs[start_idx:end_idx]:
            try:
                await extract_entity_graph_videomme(config, selected_video, captioner)
            except Exception as e:
                print(e)

day = sys.argv[1]

if __name__ == "__main__":
    asyncio.run(main(day))