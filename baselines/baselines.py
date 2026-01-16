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


import argparse
import bisect
import glob
from google import genai
import os
import time
import tiktoken
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

client = genai.Client(api_key = GOOGLE_GENAI_API_KEY)
EGOLIFE_ROOT = '/source/data/aniketr/EgoLife' # path to EgoLife dataset (HF)
egolife_caption_root = '' # path to EgoLife captions

def parse_args():
    parser = argparse.ArgumentParser(description="Configure agent and data settings")

    parser.add_argument(
        "--mllm",
        type=str,
        choices=["gpt-4.1", "gpt-5", "gemini-1.5-pro", "gemini-2.5-pro"],
        default="gpt-4.1",
        help="MLLM backbone to use to use (default: gpt-4.1)"
    )

    parser.add_argument(
        "--use_visual_oracle",
        action="store_true",
        help="Use visual frame oracle (default: False)"
    )

    parser.add_argument(
        "--use_captions",
        action="store_true",
        help="Use visual frame oracle (default: False)"
    )

    parser.add_argument(
        "--captioner",
        type=str,
        choices=["llava-video-7b", "gpt-4.1", "gpt-4.1_summarized", "llava-video-7b_summarized"],
        default="gpt-4.1_summarized",
        help="Backbone used to generate captions."
    )

    parser.add_argument(
        "--use_dt",
        action="store_true",
        help="Use diarized transcript data (default: False)"
    )

    parser.add_argument(
        "--use_dt_oracle",
        action="store_true",
        help="Use diarized transcript oracle to pick which day (default: False)"
    )

    parser.add_argument(
        "--num_prev_days",
        type=int,
        default=4,
        help="Number of previous days of diarized transcript to include (default: 3)"
    )

    parser.add_argument(
        "--remove_diarization",
        action="store_true",
        help="Remove diarization from transcripts (default: False)"
    )

    args = parser.parse_args()
    return args

## given a timelist, return frames (e.g. 50 total) centered around each of these moments.
def get_closest_images(entry_list, base_dir=".", total_files=50):
    """
    entry_list: [{'date': 'DAY1', 'time_list': ['11360904']}, ...]
    base_dir: root directory containing DAY1/, DAY2/, etc.
    total_files: how many filepaths to extract in total (default 50).
    
    Returns: list of filepaths
    """
    # Count total timestamps across all days
    all_timestamps = [(e['date'], ts) for e in entry_list for ts in e['time_list']]
    num_timestamps = len(all_timestamps)
    if num_timestamps == 0:
        return []

    # Divide slots equally across timestamps
    per_timestamp = total_files // num_timestamps
    half_window = per_timestamp // 2  # images before + after

    final_paths = []

    for day, ts in all_timestamps:
        folder = os.path.join(base_dir, day)
        if not os.path.isdir(folder):
            continue

        # Get all jpgs and sort numerically by timestamp
        all_files = sorted(
            f for f in os.listdir(folder) if f.endswith(".jpg")
        )
        all_ids = [int(f.replace(".jpg", "")) for f in all_files]

        target = int(ts)
        idx = bisect.bisect_left(all_ids, target)

        # Choose before and after files
        before_idx = max(0, idx - half_window)
        after_idx = min(len(all_ids), idx + half_window)

        # Gather closest before/after
        chosen = all_files[before_idx:idx] + all_files[idx:after_idx]
        
        # Pad if not enough images
        while len(chosen) < per_timestamp and idx + (after_idx-idx) < len(all_ids):
            chosen.append(all_files[idx + (after_idx-idx)])
            after_idx += 1

        while len(chosen) < per_timestamp and before_idx > 0:
            before_idx -= 1
            chosen.insert(0, all_files[before_idx])

        # Save with full paths
        final_paths.extend([os.path.join(folder, f) for f in chosen])

    return final_paths
    
def get_entity_graph_for_day(day='DAY1'):
    day = day.lower()
    all_entries = []

    # iterate over all hours in a day
    for file in sorted(glob.glob(f"timestamp_episodes/{day}_hour*.json")):
        with open(file, "r") as f:
            data = json.load(f)
            all_entries.extend(data)
    return str(all_entries)

    
def load_egolife_captions_for_day(day, query_day, query_time, captioner = 'gpt-4.1'):
    if captioner in ['gpt-4.1_summarized', 'llava-video-7b_summarized']:
        captions_file = f'{egolife_caption_root}/summarized_captions/day{day}_captioner-{captioner}-gpt-4.1_5min-intervals.json'
    else:
        captions_file = f'{egolife_caption_root}/{captioner}_captions/egolife-jake/{captioner}_day{day}_1fps-captions.json'
    with open(captions_file, "r") as f:
        egolife_captions = json.load(f)

    captions_for_llm = []
    for idx in range(len(egolife_captions) - 1):
        fname = list(egolife_captions[idx].keys())[0]
        if captioner in ['gpt-4.1_summarized', 'llava-video-7b_summarized']:
            tstart = datetime.strptime(fname, "%H:%M").time()
            tend = datetime.strptime(list(egolife_captions[idx+1].keys())[0], "%H:%M").time()
        else:
            tstart = timeformatter(fname.split("/")[-1][-12:-4])[:-3]
            tstart = datetime.strptime(tstart, "%H:%M:%S").time()
            tend = timeformatter(list(egolife_captions[idx+1].keys())[0].split("/")[-1][-12:-4])[:-3]
        
        if tstart >= query_time and day == query_day:
            continue

        caption = egolife_captions[idx][fname]['content'] if captioner == 'gpt-4.1' else egolife_captions[idx][fname]
        captions_for_llm.append(f'Between {tstart} and {tend} on day {day}:\n"{caption}"\n\n')

    fname = list(egolife_captions[idx+1].keys())[0]
    if day != query_day: # add the last caption of the day
        caption = egolife_captions[idx+1][fname]['content'] if captioner == 'gpt-4.1' else egolife_captions[idx+1][fname]
        captions_for_llm.append(f'Between {tend} and {tend} on day {day}:\n"{caption}"\n\n')
    
    return captions_for_llm

def get_text_prompt_subs_and_frames(question, options, subtitles):
    return f"""This video's relevant subtitles are listed below, along with time stamps:
    {subtitles}
    Select the best answer to the following multiple-choice question based only on the subtitles and provided 50 image frames and provide a justification for your answer.
    In your justification, reference specific subtitles or image frames that lead to your prediction.
    Question: {question}
    Options: {options}
    Format your response in json: 
    response :
    [ 
        {{
        'justification': 'foo bar',
        'options': options,
        'mcq_prediction': 'A', # A, B, C, or D
        }}
    ]
    """

def get_text_prompt_onlysubs(question, options, subtitles):
    return f"""This video's relevant subtitles are listed below, along with time stamps:
    {subtitles}
    Select the best answer to the following multiple-choice question based only on the provided subtitles and provide a justification for your answer.
    In your justification, reference specific subtitles that lead to your prediction.
    Question: {question}
    Options: {options}
    Format your response in json: 
    response :
    [ 
        {{
        'justification': 'foo bar',
        'options': options,
        'mcq_prediction': 'A', # A, B, C, or D
        }}
    ]
    """

def get_text_prompt_onlycaptions(question, options, captions):
    return f"""This video's captions are listed below, along with time stamps and days:
    {captions}
    Select the best answer to the following multiple-choice question based only on the provided subtitles and provide a justification for your answer.
    In your justification, reference specific subtitles that lead to your prediction.
    Question: {question}
    Options: {options}
    Format your response in json: 
    response :
    [ 
        {{
        'justification': 'foo bar',
        'options': options,
        'mcq_prediction': 'A', # A, B, C, or D
        }}
    ]
    """

def get_text_prompt_onlyframes(question, options):
    return f"""
    Select the best answer to the following multiple-choice question based only on the provided 50 image frames and provide a justification for your answer.
    In your justification, reference specific image frames that lead to your prediction.
    Question: {question}
    Options: {options}
    Format your response in json: 
    response :
    [ 
        {{
        'justification': 'foo bar',
        'options': options,
        'mcq_prediction': 'A', # A, B, C, or D
        }}
    ]
    """

def get_prev_days(days_dict, current_day, window=3):
    keys = list(days_dict.keys())
    idx = keys.index(current_day)
    start = max(0, idx - window + 1)  # ensures we don’t go negative
    return keys[start:idx+1]

def get_subs_multiple_days(days_dict, keys):
    parts = []
    for k in keys:
        parts.append(f"{days_dict[k]}\n")
    return "\n\n".join(parts)

def load_content(content_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a string that may be:
    1) Proper JSON object with "response": […]
    2) A list of dicts (Python-style literal)
    Returns a normalized list of dicts, or None on failure.
    """
    try:
        parsed = json.loads(content_str)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(content_str)

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
    first = items[0]
    return first.get("mcq_prediction")
    
def main():
    egolife_qa_jake = load_egolife_qa_jake()
    subtitles_dict = get_egolife_diarized_transcripts(remove_diarization=False)
    
    # mllm config to run
    args = parse_args()
    mllm = args.mllm
    captioner = args.captioner
    use_visual_oracle = args.use_visual_oracle
    use_captions = args.use_captions and (not use_visual_oracle)
    use_dt = args.use_dt and (not use_captions)
    use_dt_oracle = args.use_dt_oracle and use_dt # tells us which day to look at diarized transcript (DT)
    num_prev_days = (not use_dt_oracle) * (use_dt) * args.num_prev_days # 3, 4
    remove_diarization = args.remove_diarization and use_dt
    
    mcq_language = 'english' # 'chinese', 'english'
    subs_language = 'english' # 'chinese', 'english', 'chinese+english'
    max_frames= 50 # how many frames can MLLM ingest? e.g. GPT 4.1 and Gemini 2.5 Pro with Azure is max 50
    use_entity_graph = False
    assert (use_visual_oracle or use_dt or use_captions)
    print(f'Running {mllm} on EgoLifeQA with {mcq_language} questions and {subs_language} diarized transcripts.')
    
    if use_captions:
        results_root = f'../egolife_results/captions/mcq_{mcq_language}'
        results_json = f'{results_root}/{mllm}_mcq-{mcq_language}_captioner-{captioner}_useDT-{use_dt}_useDToracle-{use_dt_oracle}_prevDTdays-{num_prev_days}_removediarization-{remove_diarization}_egolife_results.json'

    elif not use_visual_oracle:
        results_root = f'../egolife_results/DT_oracle/mcq_{mcq_language}' if use_dt_oracle else f'egolife_results/prevDTdays-{num_prev_days}/mcq_{mcq_language}'
        results_json = f'{results_root}/{mllm}_DTlang-{subs_language}_removediarization-{remove_diarization}_egolife_results.json'
    else:
        results_root = f'../egolife_results/use{max_frames}frames_oracle'
        results_json = f'{results_root}/{mllm}_mcq-{mcq_language}_useDT-{use_dt}_useDToracle-{use_dt_oracle}_prevDTdays-{num_prev_days}_removediarization-{remove_diarization}_egolife_results.json'
    os.makedirs(results_root, exist_ok=True)
    
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            final_prediction_list = json.load(f)
    else:
        final_prediction_list = []
    
    print(f'Running on {results_json}')

    # iterate over all data in egolife
    for question_data in tqdm(egolife_qa_jake):
        selected_qid = question_data['ID']
        if selected_qid in [e['ID'] for e in final_prediction_list]:
            continue
        query_date = question_data['query_time']['date']
        
        if use_captions: 
            query_day = int(query_date[3])
            query_time_str = question_data['query_time']['time']
            query_time_obj = datetime.strptime(query_time_str[:-2], "%H%M%S").time()  # '16524710' -> 16:52:47
            
            # take caps from all days <= query day
            all_previous_days = [i+1 for i in range(query_day)]
            all_prev_day_caps = flatten_list([load_egolife_captions_for_day(day, query_day, query_time_obj, captioner) for day in all_previous_days])
        
        if use_dt_oracle:
            # assume we know this (oracle or hopefully with agentic planner)
            target_day = question_data['target_time'][0]['date'].upper() 
            selected_subtitles = subtitles_dict[target_day]
            if use_entity_graph:
                graph = get_entity_graph_for_day(target_day)
                selected_subtitles += f'\n Entity Graph of Relationships: {graph}'
        else:
            # use previous N days
            prev_n_days = get_prev_days(subtitles_dict, query_date, window=num_prev_days)
            selected_subtitles = get_subs_multiple_days(subtitles_dict, prev_n_days)
            
        vqa_question = question_data['question']
        options = f"A.{question_data['choice_a']}, B.{question_data['choice_b']}, C.{question_data['choice_c']}, D.{question_data['choice_d']}"
        answer = question_data['answer']    
        target_datetime = question_data['target_time']
        
        system_prompt = 'You are a helpful assistant answering questions about long videos, all taken from the first-person perspective of Jake.'
        try:
            if use_dt and use_visual_oracle:
                image_paths = get_closest_images(target_datetime, base_dir=f"{EGOLIFE_ROOT}/image_1fps_A1_JAKE", total_files=max_frames)
                master_prompt = get_text_prompt_subs_and_frames(vqa_question, options, selected_subtitles)
                final_prediction = query_multimodal(system_prompt, master_prompt, image_paths, llm_name=mllm)
            elif use_captions:
                master_prompt = get_text_prompt_onlycaptions(vqa_question, options, all_prev_day_caps)
                final_prediction = query_text_only(system_prompt, master_prompt, llm_name=mllm)
            elif use_dt:
                master_prompt = get_text_prompt_onlysubs(vqa_question, options, selected_subtitles)
                final_prediction = query_text_only(system_prompt, master_prompt, llm_name=mllm)
            elif use_visual_oracle:
                image_paths = get_closest_images(target_datetime, base_dir=f"{EGOLIFE_ROOT}/image_1fps_A1_JAKE", total_files=max_frames)
                master_prompt = get_text_prompt_onlyframes(vqa_question, options)
                final_prediction = query_multimodal(system_prompt, master_prompt, image_paths, llm_name=mllm)
            
            final_prediction = json.loads(final_prediction.model_dump_json())
            final_prediction['ID'] = selected_qid
            if use_visual_oracle:
                final_prediction['oracle_image_paths'] = image_paths
            final_prediction_list.append(final_prediction)
            with open(results_json, 'w') as f:
                json.dump(final_prediction_list, f, indent=4)
        except Exception as e:
            print(e)
            if (mllm == 'gemini-2.5-pro'):
                total_tokens = client.models.count_tokens(model="gemini-2.5-pro", contents=master_prompt)
                print("QID=", selected_qid, " Input tokens Gemini 2.5 Pro = ", total_tokens.total_tokens)
                time.sleep(60) # rate limits
            else:
                encoding = tiktoken.get_encoding("o200k_base")
                print("QID=", selected_qid, " Input tokens GPT 4.1 = ", len(encoding.encode(string)))

if __name__ == "__main__":
    main()