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


from baselines import *
from google import genai
from google.genai import types
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

root_dir = f'{EGOLIFE_ROOT}/image_1fps_A1_JAKE_384x384'

def filter_images_by_time(folder_path, query_time_str):
    """
    Return all images in folder whose timestamp <= query_time.
    """
    query_time = datetime.strptime(query_time_str, "%H:%M:%S").time()
    filtered_images = []

    for img_path in sorted(glob.glob(f"{folder_path}/*.jpg")):
        ts_str = img_path.split("/")[-1][:6]  # first 6 digits are HHMMSS
        img_time = datetime.strptime(ts_str, "%H%M%S").time()
        if img_time <= query_time:
            filtered_images.append(img_path)

    return filtered_images
    
def uniformly_sample_previous_n_frames(root_dir, query_day, query_time, num_samples):
    n_frame_filepaths = []

    # add all frames of all previous days
    all_previous_days = [i+1 for i in range(query_day - 1)]
    for d in all_previous_days:
        n_frame_filepaths += (sorted(glob.glob(f'{root_dir}/DAY{d}/*.jpg')))
        
    # add query day frames up until query_time
    same_day_files = filter_images_by_time(f'{root_dir}/DAY{query_day}', query_time)
    n_frame_filepaths += ((same_day_files))

    # return n uniformly sampled frames from all filepaths until query_time on query_day
    indices = list(np.linspace(1, len(n_frame_filepaths) - 1, num_samples, dtype=int))

    return [n_frame_filepaths[e] for e in indices]

def get_egolife_mllm_text_prompt_with_cot(question, options):
    return f"""
Select the best answer to the following multiple-choice question based only on the provided interleaved image frames and diarized transcripts (each corresponding to the same timestamp) and provide a justification for your answer.
In your justification, reference specific image frames or transcripts that lead to your prediction.
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

# parse HHMMSS-like token to seconds (assumes first 6 chars are HHMMSS)
def _hhmmss_to_secs(s):
    s = s[:6].zfill(6)
    hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:6])
    return hh*3600 + mm*60 + ss

# parse frame path like .../DAY1/11094308.jpg -> (absolute_seconds, day)
def _frame_path_to_secs(path):
    m = re.search(r'DAY(\d+)[/\\](\d+)', path)
    if not m:
        raise ValueError(f"can't parse day/timestamp from {path}")
    day = int(m.group(1))
    ts = m.group(2)
    # treat day 1 as offset 0 seconds, day N offset (N-1)*86400
    return _hhmmss_to_secs(ts) + (day - 1) * 86400

def map_frames_to_transcripts(frame_paths, df_transcript):
    # expects df_transcript columns: 'day','start_t','end_t','transcript_english'
    # convert transcript times to absolute seconds (day offset included)
    def t_to_abssecs(row, col):
        h, m, s = map(int, row[col].split(':'))
        return h*3600 + m*60 + s + (int(row['day']) - 1) * 86400

    starts = df_transcript.apply(lambda r: t_to_abssecs(r, 'start_t'), axis=1).to_numpy()
    ends   = df_transcript.apply(lambda r: t_to_abssecs(r, 'end_t'),   axis=1).to_numpy()
    texts  = df_transcript['transcript_english'].to_numpy()

    mapped = []
    for p in frame_paths:
        fsecs = _frame_path_to_secs(p)
        inside_idx = np.where((starts <= fsecs) & (fsecs <= ends))[0]
        if inside_idx.size:
            mapped.append(texts[inside_idx[0]])
        else:
            nearest = np.argmin(np.abs(starts - fsecs))
            mapped.append(texts[nearest])
    return mapped

def upload_single_image(client, path):
    try:
        return client.files.upload(file=path)
    except Exception as e:
        print(f"Error uploading {path}: {e}")
        return None

def upload_images_parallel(client, image_paths, max_workers=16):
    start = time.time()
    file_refs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(upload_single_image, client, p): p for p in image_paths}
        for future in as_completed(futures):
            ref = future.result()
            if ref:
                file_refs.append(ref)
    print(f"Uploaded {len(file_refs)} images in {time.time() - start:.2f} seconds.")
    return file_refs

def get_content_for_egolife_qid(client, egolife_qa_jake, qid, n_uniform_samples = 3000):
        vqa_question = egolife_qa_jake[qid]['question']
        options = f"""A.{egolife_qa_jake[qid]['choice_a']}, B.{egolife_qa_jake[qid]['choice_b']}, C.{egolife_qa_jake[qid]['choice_c']}, D.{egolife_qa_jake[qid]['choice_d']}"""
        query_time = egolife_qa_jake[qid]['query_time']
        query_day = int(query_time['date'][3])
        query_time_str = timeformatter(query_time['time'])[:-3]
        prev_n_frames = uniformly_sample_previous_n_frames(root_dir, query_day, query_time_str, n_uniform_samples)
        system_prompt = 'You are a helpful assistant that answers questions about long videos taken from the first-person perspective of Jake.'
        query = get_egolife_mllm_text_prompt_with_cot(vqa_question, options)
        contents = [system_prompt + query]
    
        # upload sampled frames via files API
        file_refs = upload_images_parallel(client, prev_n_frames, max_workers=16)
        
        for i in range(len(file_refs)):
            f = file_refs[i]
            contents.append(f)  # each 'f' is a file reference
        return contents
    

def main():
    start_i = int(sys.argv[1])
    end_i = start_i + 50
    n_uniform_samples = 3000
    results_json = f'../egolife_results/gemini-2.5-pro-uniform-sample-frames+dt-{n_uniform_samples}.json'
    selected_qids = [i for i in range(start_i, end_i)]
    
    # Load and clean formatting inconsistencies in EgoLifeQA (Jake)
    with open(f"{EGOLIFE_ROOT}/EgoLifeQA/EgoLifeQA_A1_JAKE.json", "r", encoding="utf-8") as f:
        egolife_qa_jake = json.load(f)    
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
                    
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            final_prediction_list = json.load(f)
        already_done_qids = [int(e['key'].split("-")[-1]) - 1 for e in final_prediction_list]
    else:
        final_prediction_list = []
        already_done_qids = []
    
    client = genai.Client(api_key=GOOGLE_GENAI_API_KEY)

    file_upload_temp = f"../egolife_results/egolife-batch-fileupload-uris_{selected_qids[0]}-{selected_qids[-1]}.json"
    if os.path.exists(file_upload_temp):
        with open(file_upload_temp, "r") as f:
            uris = json.load(f)
        print(f"Loaded {len(uris)} uris from previous progress.")
    else:
        uris = []
        
    content_for_qid = {}
    for qid in selected_qids:
        if qid in already_done_qids:
            print('Already done with qid', qid)
            continue
        uris_for_qid = {"qid": qid, "uris": []}
        print(uris_for_qid)
        content = get_content_for_egolife_qid(client, egolife_qa_jake, qid, n_uniform_samples)
        content_for_qid[qid] = content
        for i, item in enumerate(content_for_qid[qid]):
            if isinstance(item, genai.types.File):
                uris_for_qid['uris'].append({"uri" : item.uri}) 
        uris.append(uris_for_qid)
        with open(file_upload_temp, "w") as f:
            json.dump(uris, f, indent=4)
    
    df_transcript_all_days = get_egolife_dt_dataframe()
    df_transcript_all_days['start_t'] = df_transcript_all_days['start_t'].str.replace(r',\d{1,3}', '', regex=True) # remove milliseconds
    df_transcript_all_days['end_t'] = df_transcript_all_days['end_t'].str.replace(r',\d{1,3}', '', regex=True)
    
    payload_file = f"../egolife_results/egolife-batch-qid_{selected_qids[0]}-{selected_qids[-1]}.jsonl"
    print(f'Generating {payload_file}')
    
    with open(payload_file, "w") as f:
        requests = []
        for q in selected_qids:
            if q in already_done_qids:
                continue
            query_time = egolife_qa_jake[q]['query_time']
            query_day = int(query_time['date'][3])
            query_time_str = timeformatter(query_time['time'])[:-3]
            prev_n_frames = uniformly_sample_previous_n_frames(root_dir, query_day, query_time_str, num_samples=3000)
            closest_dt_to_uniformly_sampled_frames = map_frames_to_transcripts(prev_n_frames, df_transcript_all_days)
            parts = []
            for i, item in enumerate(content_for_qid[q]):
                if isinstance(item, genai.types.File):  # or type(item).__name__ == "File"
                    # replace File object with file reference
                    parts.append({"fileData": {"fileUri": item.uri}})
                    closest_dt = closest_dt_to_uniformly_sampled_frames[i-1]
                    parts.append({"text": closest_dt})
                else:
                    # keep text or dict as-is
                    parts.append({"text": item})
            request = {"key": f"egolife-qid-{q+1}", "request": {"model": "models/gemini-2.5-pro", "contents": [{"parts": parts}]}}
            requests.append(request)
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    uploaded_file = client.files.upload(
        file=payload_file,
        config=types.UploadFileConfig(display_name=payload_file, mime_type='application/json')
    )
    
    print(f"Uploaded payload file: {uploaded_file.name}")
    
    file_batch_job = client.batches.create(
        model="gemini-2.5-pro",
        src=uploaded_file.name,
        config={
            'display_name': f"egolife-job-qid_{selected_qids[0]}-{selected_qids[-1]}",
        },
    )
    
    print(f"Created batch job: {file_batch_job.name}")
    
    job_name = file_batch_job.name  # (e.g. 'batches/your-batch-id')
    batch_job = client.batches.get(name=job_name)
    
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    ])
    
    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name) # Initial get
    while batch_job.state.name not in completed_states:
      print(f"Current state: {batch_job.state.name}")
      time.sleep(30) # Wait for 30 seconds before polling again
      batch_job = client.batches.get(name=job_name)
    
    print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == 'JOB_STATE_FAILED':
        print(f"Error: {batch_job.error}")
    
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
    
        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            print(f"Results are in file: {result_file_name}")
    
            print("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            # Process file_content (bytes) as needed
            print(file_content.decode('utf-8'))
    
    resps = [json.loads(file_content.decode('utf-8').split("\n")[q]) for q in range(len(requests))]
    with open(results_json, 'w') as f:
        json.dump(final_prediction_list + resps, f, indent=4)

if __name__ == "__main__":
    main()