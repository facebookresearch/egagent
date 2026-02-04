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
from collections import defaultdict
from create_entity_graph import get_egolife_diarized_transcripts
from datetime import datetime, timedelta, time as dtime
import json
import numpy as np
import os
import pandas as pd 
from pathlib import Path
import re
import sys
import time
from utils import get_egolife_transcript_df, get_videomme_transcript_df, timeformatter

dataset = 'videomme' # videomme, egolife
mllm = 'gpt-4.1' # multimodal llm used to fuse captions and transcripts

def parse_time(t):
    """Parse a time string into a datetime object."""
    t = re.sub(r',(\d{1,2})(?!\d)', lambda m: ',' + m.group(1).ljust(3, '0'), t)
    # return datetime.strptime(t, "%H:%M:%S")
    return datetime.strptime(t, "%H:%M:%S,%f")

def parse_caption_range(caption_path):
    """Parse the start and end times of a caption from a path string."""
    t_str = re.search(r'(\d{8})\.mp4', caption_path).group(1)
    h, m, s, ms = t_str[:2], t_str[2:4], t_str[4:6], t_str[6:]
    # start = parse_time(f"{h}:{m}:{s}")
    start = parse_time(f"{h}:{m}:{s},{ms}")
    end = start + timedelta(seconds=30)  # adjust if segment duration differs
    return start, end

def merge_captions_with_transcripts(captions, diarized):
    """Merge captions with audio transcripts. For each caption, collect overlapping diarized segments."""
    merged = []
    entries = []

    # Build structured diarized list: [(start, end, raw_time, text)]
    for i in range(0, len(diarized), 2):
        if '-->' in diarized[i]:
            raw_time = diarized[i].strip()
            t1, t2 = [parse_time(x.strip()) for x in raw_time.split('-->')]
            text = diarized[i+1]
            entries.append((t1, t2, raw_time, text))

    # For each caption, collect overlapping diarized segments
    for cap_dict in captions:
        cap_path = list(cap_dict.keys())[0]
        c_start, c_end = parse_caption_range(cap_path)

        relevant = [
            {'timestamp': raw_time, 'text': text}
            for (t1, t2, raw_time, text) in entries
            if not (t2 < c_start or t1 > c_end)
        ]
        merged.append(relevant)

    return merged

def get_chunkwise_caps_for_day(captioner, json_fname, num_minutes = 5):
    """Group captions by 10-minute intervals."""
    
    json_path = Path(json_fname)  # change this to your file path
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # --- Group captions by 10-minute intervals ---
    interval_captions = defaultdict(list)
    
    def extract_time_from_path(path_str):
        """Extract the time from an EgoLife path string in the format DAY1_A1_JAKE_11094208.mp4."""
        # Example filename: DAY1_A1_JAKE_11094208.mp4
        match = re.search(r"_(\d{8})\.mp4$", path_str)
        if not match:
            return None
        hh, mm, ss = match.group(1)[:2], match.group(1)[2:4], match.group(1)[4:6]
        return f"{hh}:{mm}:{ss}"
        
    for entry in data:
        for path, caption in entry.items():
            start_time = extract_time_from_path(path)
            if start_time:
                hh, mm, ss = map(int, start_time.split(":"))
                interval_start = (mm // num_minutes) * num_minutes  # 0, 5, 10, ... 55, 60
                interval_label = f"{hh:02d}:{interval_start:02d}"
                interval_captions[interval_label].append((start_time, caption))
    
    # --- Sort and generate formatted strings ---
    output_by_interval = {}
    
    for interval, entries in sorted(interval_captions.items()):
        entries.sort(key=lambda x: x[0])  # sort by start time
        formatted_entries = []
    
        for i, (start_time, caption) in enumerate(entries):
            # Estimate end time = next entry’s start time if exists, else +30s
            if i + 1 < len(entries):
                end_time = entries[i + 1][0]
            else:
                hh, mm, ss = map(int, start_time.split(":"))
                ss += 30
                if ss >= 60:
                    ss -= 60
                    mm += 1
                if mm >= 60:
                    mm -= 60
                    hh = (hh + 1) % 24
                end_time = f"{hh:02}:{mm:02}:{ss:02}"

            if captioner == 'gpt-4.1':
                caption = caption['content']
            
            formatted_entries.append(
                f'{start_time} to {end_time} - "{caption}"'
            )
    
        output_by_interval[interval] = "\n\n".join(formatted_entries)
    return output_by_interval

from utils import query_text_only
from langgraph_agent import get_llm_worker
from pydantic import BaseModel

class SummarizedCaption(BaseModel):
    """Given a series of text captions about an hour of video, summarize them to a smaller length."""
    summarized_caption: str
    
def get_caption_summary_llm(model='gpt-4.1'):
    capsummary_system = """
        You are an expert multimodal summarization model.
        You will be given a series of text captions about 5 minutes of video along with timestamps for context. 
        The input has around 2500 - 3000 words.
        Create a detailed text summary of this input between 1000 - 2000 words in total length. 
        Follow these rules carefully:
        * Focus on relevant visual content (location, people, actions, and objects).
        * Keep the caption written in neutral, descriptive tone and preserve factual details.
        * Remove references to timestamps, as these are not relevant to what is happening in the scene.
        """
    
    capsummary_human = """
        5 minutes of captions to summarize: {caption_text}
        """
    
    return get_llm_worker(capsummary_system, capsummary_human, SummarizedCaption, model)

class FusedCaption(BaseModel):
    """Given a diarized transcript with timestamps and a text caption, naturally fuse them together."""
    fused_caption: str
    
def get_caption_dt_fuser_llm(model='gpt-4.1'):
    fuser_system = """
        You are an expert multimodal summarization model.
        You will be given two aligned inputs corresponding to the same short video segment (about 30 seconds):
        1. Visual Caption — a detailed description of what is visible in the video.
        2. Diarized Transcript — spoken dialogue transcribed with timestamps and speaker names.
        Your task is to fuse these into a single, coherent, and natural paragraph that integrates both visual and spoken information.
        Follow these rules carefully:
        * Focus on relevant spoken content (who says what and its meaning) and highlight visual content (location, people, actions, and objects).
        * Preserve factual details but avoid repetition or speculation.
        * Keep the fused caption written in neutral, descriptive tone.
        * Output only the fused caption — no explanations or metadata.
        """
    
    fuser_human = """
        Here are the inputs for this segment:
        1. Visual Caption: {caption_text}
        2. Diarized Transcript: {transcript_text}
        Produce one fused caption that naturally combines both.
        """
    
    return get_llm_worker(fuser_system, fuser_human, FusedCaption, model)

def summarize_captions(captions_file):
    """Summarize captions for EgoLife at chunk intervals (default 5 minutes)."""
    chunk_num_minutes = 5 # summarize caption chunks of this duration (min)
    results_json = f'captioning/summarized_captions/day{day}_captioner-{captioner}_summarized-{mllm}_{chunk_num_minutes}min-intervals.json'
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            final_caption_list = json.load(f)
    else:
        final_caption_list = []
    
    caption_by_timechunk = get_chunkwise_caps_for_day(captioner, captions_file, chunk_num_minutes)
    cap_summarizer = get_caption_summary_llm(model = mllm)
    
    for interval, cap_text in caption_by_timechunk.items():
        # print(f"\n===== Interval starting {interval} =====\n")
        # print(cap_text) ; print()
        if interval in [list(e.keys())[0] for e in final_caption_list]:
            print(f'Already done {interval}')
            continue
        print(f'Day {day}, {interval}')
        summarized_cap = cap_summarizer.invoke({"caption_text": cap_text}).summarized_caption
        final_caption_list.append({interval: summarized_cap})
        with open(results_json, 'w') as f:
            json.dump(final_caption_list, f, indent=4)

def fuse_captions_and_dt_egolife(captions_file):
    """Fuse captions and diarized transcripts for EgoLife."""
    chunk_num_minutes = 60 # get every hour of captions
    fuser_llm = get_caption_dt_fuser_llm(model=mllm)
    
    with open(captions_file, "r") as f:
        egolife_captions = json.load(f)

    caption_by_timechunk = get_chunkwise_caps_for_day(captioner, captions_file, chunk_num_minutes)
    diarized_transcripts_dict = get_egolife_diarized_transcripts()
    dt_for_day = diarized_transcripts_dict[f'DAY{day}'].split("\n")
    episodes = [dt_for_day[i] for i in range(len(dt_for_day)) if i%2!=0][:-1] # each hour
    
    results_json = f'captioning/summarized_captions/day{day}_captioner-{captioner}_summarized-{mllm}_{chunk_num_minutes}min-intervals.json'
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            final_caption_list = json.load(f)
    else:
        final_caption_list = []
    
    num_hours = len(caption_by_timechunk.keys())

    for hour_idx in range(1, num_hours+1):
        hour = list(caption_by_timechunk.keys())[hour_idx-1][:2]
        captions_selected = [e for e in egolife_captions if list(e.keys())[0][-12:-10] in [hour]]
        dt_for_hour = ast.literal_eval(episodes[hour_idx-1])
        dt_split_by_30sec = merge_captions_with_transcripts(captions_selected, dt_for_hour)
        fused_outfile = f'captioning/fused_dt_and_captions/{captioner}_day{day}_hour{hour_idx}.json'
        
        if os.path.exists(fused_outfile):
            with open(fused_outfile, "r") as f:
                fused_captions = json.load(f)
            already_done = [list(e.keys())[0] for e in fused_captions]
        else:
            fused_captions, already_done = [], []
    
        print(f'Generating {fused_outfile}')
        start = time.time()
        for i in range(len(captions_selected)):
            [[k, v]] = captions_selected[i].items()
            if k in already_done:
                continue
            if len(dt_split_by_30sec[i]) != 0: # if there is dt in this caption window
                # fuse and overwrite v with v_fused
                caption_start_t = timeformatter(k[-12:-4])
                if i == len(captions_selected) - 1:
                    end_t = datetime.strptime(caption_start_t, '%H:%M:%S,%f') +timedelta(seconds=30) # add 30 sec to last caption start time
                    caption_end_t = end_t.strftime('%H:%M:%S')
                else: 
                    [[k2, v2]] = captions_selected[i+1].items()
                    caption_end_t = timeformatter(k2[-12:-4])
                caption_content = v['content'] if captioner == 'gpt-4.1' else v
                caption_with_tstamp = f"Caption start time: {caption_start_t}, Caption end time: {caption_end_t}, Caption content: '{caption_content}'"
                # print('ORIG: ',  caption_with_tstamp)
                # print('DT: ', dt_split_by_30sec[i])
                v_fused = fuser_llm.invoke({"caption_text": caption_with_tstamp, "transcript_text": dt_split_by_30sec[i]}).fused_caption
                # print('FUSED: ', v_fused); print()
                fused_captions.append({k:v_fused})
            else:
                fused_captions.append({k:v})  # ignore fusion when there is no DT
            with open(fused_outfile, "w") as f:
                json.dump(fused_captions, f, indent=4) # delete these later
        print(f'Finished Generating {fused_outfile} in {time.time() - start: .2f} sec\n')
            # print(k, fused_captions[k]) ; print()

def time_to_seconds(t):
    """Convert HH:MM:SS string to total seconds."""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def get_overlapping_transcript_videomme(df, start_t, end_t):
    """Get concatenated transcript for an interval of VideoMME."""
    if len(df) == 0:
        return []
    overlaps = df[(df["end_sec"] > start_t) & (df["start_sec"] < end_t)]
    return " ".join(overlaps["transcript_english"].tolist())

def fuse_captions_and_dt_videomme(captions_file):
    """Fuse captions and diarized transcripts for VideoMME."""
    df_videomme = json.loads(pd.read_parquet("/source/data/video-mme/videomme/test-00000-of-00001.parquet").to_json(orient='records'))
    df_videomme_long_vIDs = np.unique([e['videoID'] for e in df_videomme if e['duration'] == 'long'])
    
    batch_offset = 50 # 6 batches of 50 each = 300 long videos
    start_idx = int(sys.argv[1])
    end_idx = start_idx + batch_offset
    
    with open(captions_file, "r") as f:
        videomme_captions = json.load(f)
    
    fuser_llm = get_caption_dt_fuser_llm(model='gpt-4.1')
    window_duration = 64 # each window captions 64 seconds of video

    print(f'Generating for videos {start_idx} through {start_idx+batch_offset}')
    for selected_video in df_videomme_long_vIDs[start_idx:start_idx+batch_offset]:
        fused_outfile = f'captioning/fused_dt_and_{captioner.lower()}_captions/gpt-4.1_{selected_video}.json'

        filename = f'/source/data/video-mme/data/{selected_video}.mp4'
        all_windows_for_vID = list([e for e in videomme_captions if list(e.keys())[0] == filename][0].values())[0]
        try:
            df_transcript_all_days = get_videomme_transcript_df(selected_video)
            df_transcript_all_days["start_sec"] = df_transcript_all_days["start_t"].apply(time_to_seconds)
            df_transcript_all_days["end_sec"] = df_transcript_all_days["end_t"].apply(time_to_seconds)
        except:
            print(f'Error accessing transcript for {selected_video}')
            df_transcript_all_days = pd.DataFrame()
        
        if os.path.exists(fused_outfile):
            with open(fused_outfile, "r") as f:
                fused_captions = json.load(f)
            already_done = [list(e.keys())[0] for e in fused_captions]
        else:
            fused_captions, already_done = [], []
        
        print(f'Generating {fused_outfile} for {len(all_windows_for_vID)} windows')
        start = time.time()
        for w_num, (window, caption_content) in enumerate(all_windows_for_vID.items()):
            try:
                start_t = w_num * window_duration
                end_t = (w_num+1) * window_duration
                transcript = get_overlapping_transcript_videomme(df_transcript_all_days, start_t, end_t)
                if window in already_done:
                    continue
                if len(transcript) != 0: # if there is dt in this caption window
                    caption_with_tstamp = f"Caption start time: {start_t}, Caption end time: {end_t}, Caption content: '{caption_content}'"
                    cap_fused = fuser_llm.invoke({"caption_text": caption_with_tstamp, "transcript_text": transcript}).fused_caption
                    fused_captions.append({window:cap_fused})
                else:
                    fused_captions.append({window:caption_content})  # ignore fusion when there is no transcript
                with open(fused_outfile, "w") as f:
                    json.dump(fused_captions, f, indent=4) # delete these later
            except Exception as e:
                print(e)
        print(f'Finished Generating {fused_outfile} in {time.time() - start: .2f} sec\n')

if __name__ == "__main__":
    day = sys.argv[1]
    caption_root = '' # path to egolife and videomme captions (see format for each below)
    
    if dataset == 'egolife':
        captioner = 'gpt-4.1'
        captions_file = f'{caption_root}/{captioner}_captions/egolife-jake/{captioner}_day{day}_1fps-captions.json'
        fuse_captions_and_dt_egolife(captions_file)
    elif dataset == 'videomme':
        captioner = 'LLaVA-Video-7B' # LLaVA-Video-7B
        captions_file = f'{caption_root}/{captioner.lower()}_captions/videomme-long/videomme-long_{captioner}_slidingwindow.json'
        fuse_captions_and_dt_videomme(captions_file)