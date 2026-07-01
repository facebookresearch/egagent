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

def run_agentic_inference(app, vqa_question, options, vidstart, vidend, transcripts, query_time, day_search_dict, selected_video, working_memory_init, verbose=False):
    inputs = {
        "plan": ["empty"],
        "plan_steps": [],
        "selected_video": selected_video,
        "working_memory": working_memory_init,
        "current_task": "",
        "previous_tasks": ["empty"],
        "remaining_tools_this_step": [],
        "next_route": "",
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

    trace_keys = (
        "plan", "plan_steps", "current_task", "previous_tasks",
        "remaining_tools_this_step", "next_route", "working_memory",
    )
    for output in app.stream(inputs, config):
        for key, value in output.items():
            if verbose:
                snapshot = {k: value.get(k) for k in trace_keys if k in value}
                print(f"\n{'=' * 60}\nNODE: {key}\n{'=' * 60}")
                print(json.dumps(snapshot, indent=2, default=str))
    
    return value


def prepare_videomme_example(question_data):
    selected_video = question_data['videoID']
    vqa_question = question_data['question']
    options = f"""{question_data['options'][0]}, {question_data['options'][1]}, {question_data['options'][2]}, {question_data['options'][3]}"""
    ts_path = f'{asr_dir}/{selected_video}.srt'
    transcripts = load_srt_hhmmss(ts_path) if os.path.exists(ts_path) else "NO TRANSCRIPTS AVAILABLE "
    num_video_frames, image_paths = get_50_frames_from_video(f'{frames_dir}/{selected_video}/')
    vidstart = seconds_to_hhmmss(image_paths[0].split("/")[-1][:-4])
    vidend = seconds_to_hhmmss(image_paths[-1].split("/")[-1][:-4])
    day_search_dict = {'DAY0': {'start': vidstart, 'end': vidend}}
    query_time = {'date': 'DAY0', 'time': vidend}
    return selected_video, vqa_question, options, vidstart, vidend, transcripts, query_time, day_search_dict
    

def videomme_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tscript-search",
        default="llm",
        choices=["llm", "bm25"],
        help="Transcript search backend used inside the agent graph.",
    )
    parser.add_argument(
        "--example-id",
        type=str,
        default=None,
        help="Run a single VideoMME-long question by question_id (e.g. 601-1) and print the agent trace.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print state snapshots after each graph node (use with --example-id).",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    workflow = StateGraph(VeryLongVideoQA)
    
    # Define the nodes (node names match paper Appendix E where applicable)
    workflow.add_node("planner", planner_node)
    workflow.add_node("route_plan", route_next_tool_node)
    workflow.add_node("mark_step_complete", mark_step_complete_node)
    workflow.add_node("search_eg", search_entity_graph)
    workflow.add_node("retrieve_frames_sql", retrieve_frames_sql)
    workflow.add_node("analyze_retrieved_frames", analyze_retrieved_frames)
    workflow.add_node("generate_answer", generate_answer)

    if args.tscript_search == "llm":
        workflow.add_node("search_transcripts", retrieve_transcripts)
    else:
        workflow.add_node("search_transcripts", search_and_analyze_transcripts_bm25)
    
    # Build graph: planner picks tools per step, then early-exit grading
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "route_plan")
    workflow.add_conditional_edges(
        "route_plan",
        route_from_dispatch,
        {
            "visual": "retrieve_frames_sql",
            "eg": "search_eg",
            "audio": "search_transcripts",
            "step_done": "mark_step_complete",
        },
    )
    workflow.add_edge("retrieve_frames_sql", "analyze_retrieved_frames")
    workflow.add_edge("analyze_retrieved_frames", "route_plan")
    workflow.add_edge("search_eg", "route_plan")
    workflow.add_edge("search_transcripts", "route_plan")
    workflow.add_conditional_edges(
        "mark_step_complete",
        grade_plan_completion,
        {
            "complete": "generate_answer",
            "incomplete": "planner",
        },
    )
    workflow.add_edge("generate_answer", END)
    app = workflow.compile()

    df_videomme = json.loads(pd.read_parquet(f"{VIDEO_MME_ROOT}/videomme/test-00000-of-00001.parquet").to_json(orient='records'))
    df_videomme_long = [e for e in df_videomme if e['duration'] == 'long']

    if args.example_id is not None:
        example = next(
            (q for q in df_videomme_long if str(q["question_id"]) == str(args.example_id)),
            None,
        )
        if example is None:
            raise ValueError(f"No VideoMME-long example with question_id {args.example_id!r}")
        selected_video, vqa_question, options, vidstart, vidend, transcripts, query_time, day_search_dict = (
            prepare_videomme_example(example)
        )
        working_memory_init = ""
        print(f"Running example {args.example_id} (video {selected_video}): {vqa_question}")
        print(f"Ground truth: {example['answer']}")
        value = run_agentic_inference(
            app, vqa_question, options, vidstart, vidend, transcripts, query_time,
            day_search_dict, selected_video, working_memory_init, verbose=args.verbose,
        )
        print("\n" + "=" * 60)
        print("FINAL")
        print("=" * 60)
        print(f"Plan: {value['plan']}")
        print(f"Prediction: {value['answer'].mcq_prediction}")
        print(f"Justification: {value['answer'].justification}")
        print(f"Total tokens: {value['total_tokens']}")
        return

    # Inference over full Video-MME (long) dataset
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
            
        selected_video, vqa_question, options, vidstart, vidend, transcripts, query_time, day_search_dict = (
            prepare_videomme_example(question_data)
        )
        answer = question_data['answer']
        working_memory_init = ""

        # wrap in try-except to handle API errors (e.g. rate limits)
        # try:
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
        # except Exception as e:
        #     print(e)

if __name__ == "__main__":
    videomme_inference()