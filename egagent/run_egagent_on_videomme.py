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


from run_egagent_on_egolife import *

TSCRIPT_SEARCH = 'llm'

def run_agentic_inference(app, vqa_question, options, vidstart, vidend, transcripts, query_time, day_search_dict, selected_video, working_memory_init):
    inputs = {
        "plan": ["empty"],
        "working_memory": working_memory_init,
        "current_task": "",
        "previous_tasks": ["empty"],
        "query_time": query_time,
        "day_search_dict": day_search_dict,
        "question": vqa_question, 
        "candidates": options, 
        "audio_transcripts": transcripts,
        "total_tokens": [],
        "vidstart": vidstart, # needed by retrieve_frames_sql on videomme
        "vidend": vidend, # needed by retrieve_frames_sql on videomme
    }
    config = RunnableConfig(recursion_limit=100)
    
    for output in app.stream(inputs, config):
        for key, value in output.items():
            # Node
            pass
    
    # Final generation
    return value
    

def videomme_inference():
    workflow = StateGraph(GraphState)
    
    # Define the nodes
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("search_entity_graph", search_entity_graph)
    workflow.add_node("retrieve_frames_sql", retrieve_frames_sql)
    workflow.add_node("analyze_retrieved_frames", analyze_retrieved_frames)  
    workflow.add_node("search_and_analyze_transcripts_bm25", search_and_analyze_transcripts_bm25) # bm25 search
    workflow.add_node("retrieve_transcripts", retrieve_transcripts) # llm search+analyze
    workflow.add_node("generate_answer", generate_answer)
    
    
    # Build graph
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "retrieve_frames_sql")
    workflow.add_edge("retrieve_frames_sql", "analyze_retrieved_frames")
    workflow.add_edge("analyze_retrieved_frames", "search_entity_graph")
    if TSCRIPT_SEARCH == 'llm':
        workflow.add_edge("search_entity_graph", "retrieve_transcripts")
        workflow.add_conditional_edges(
            "retrieve_transcripts",
            grade_plan_completion,
            {
                "complete": "generate_answer",
                "incomplete": "planner_node",
            },
        )
    elif TSCRIPT_SEARCH == 'bm25':
        workflow.add_edge("search_entity_graph", "search_and_analyze_transcripts_bm25")
        workflow.add_conditional_edges(
            "search_and_analyze_transcripts_bm25",
            grade_plan_completion,
            {
                "complete": "generate_answer",
                "incomplete": "planner_node",
            },
        )
    workflow.add_edge("generate_answer", END)
    app = workflow.compile()


    # Inference over full Video-MME (long) dataset
    df_videomme = json.loads(pd.read_parquet("lmms-lab/Video-MME/videomme/test-00000-of-00001.parquet").to_json(orient='records'))
    df_videomme_long = [e for e in df_videomme if e['duration'] == 'long']
    
    total_questions = len(df_videomme_long)
    results_json = RESULTS_ROOT / 'egagent_videomme-long_results_all.json'
    print(f'Generating ', results_json)
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            final_prediction_list = json.load(f)
    else:
        final_prediction_list = []
    print(f'Done with {len(final_prediction_list)} / {total_questions}')
    completed_ids = {e['ID'] for e in final_prediction_list}
    for question_data in tqdm(df_videomme_long, desc="Processing"):
        results = {}
        selected_qid = question_data['question_id']

        if selected_qid in completed_ids:
            print(f'Skipping {selected_qid}, already done')
            continue
            
        selected_video = question_data['videoID']
        vqa_question = question_data['question']
        options = f"""{question_data['options'][0]}, {question_data['options'][1]}, {question_data['options'][2]}, {question_data['options'][3]}"""
        answer = question_data['answer']
        ts_path = f'{asr_dir}/{selected_video}.srt'
        transcripts = load_srt_hhmmss(ts_path) if os.path.exists(ts_path) else "NO TRANSCRIPTS AVAILABLE " # with timestamps
        num_video_frames, image_paths = get_50_frames_from_video(f'{frames_dir}/{selected_video}/') # uniformly sample from start to end
        vidstart = seconds_to_hhmmss(image_paths[0].split("/")[-1][:-4])
        vidend = seconds_to_hhmmss(image_paths[-1].split("/")[-1][:-4])
        day_search_dict = {'DAY0': {'start': vidstart , 'end': vidend}} # videomme is always one day, assume 0
        query_time = {'date': 'DAY0', 'time': vidend} # assume query is always asked at the end of the video
        working_memory_init = ""

        # wrap in try-except to handle API errors (e.g. rate limits)
        try:
            value = run_agentic_inference(app, vqa_question, options, vidstart, vidend, transcripts, query_time, day_search_dict, selected_video, working_memory_init)
            
            results['ID'] = selected_qid
            results['question'] = vqa_question
            results['options'] = options
            results['answer'] = answer
            results['plan'] = value["plan"]
            results['working_memory'] = value['working_memory']
            results['mcq_prediction'] = value["answer"].mcq_prediction
            results['justification'] = value["answer"].justification
            results['total_tokens'] = value["total_tokens"]
            final_prediction_list.append(results)
            completed_ids.add(selected_qid)
            with open(results_json, 'w') as f:
                json.dump(final_prediction_list, f, indent=4)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    videomme_inference()