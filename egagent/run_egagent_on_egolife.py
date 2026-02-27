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
import bm25s
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph_agent import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.getLogger('bm25s').setLevel(logging.WARNING)

def get_egolife_daysearchdict(query_time):
    """
    Create dictionary of start and end times of all days in EgoLife (Jake). This is used by the retriever tool when deciding time filters to query.
    """
    day_search_dict = {}
    current_day = int(query_time['date'][3]) # all egolife dates are 'DAYX' format
    all_days = [f'DAY{i+1}' for i in range(current_day)]
    for d in all_days:
        image_dir = f'{frames_dir}/{d}/'
        image_files = sorted(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
        start_t = int(image_files[0][:6]) # first image of day
        end_t = int(image_files[-1][:6]) # last image of day
        day_search_dict[d] = {"start": start_t, "end": end_t}
    
    # replace current day's end time with query time (i.e. no need to search in future)
    current_day_end = int(query_time['time'][:6])
    day_search_dict[query_time['date']]['end'] = current_day_end
    return day_search_dict

    
def search_entity_graph(state):
    def get_eg_query_results(results):
        res_formatted = ""
        for row in results:
            video_id = row[1]
            start_t = str(row[2])
            end_t = str(row[3])
            source_ids = row[5]
            source_types = row[6]
            target_ids = row[7]
            target_types = row[8]
            rel_types = row[9]
            if dataset == 'videomme':
                res_formatted += (f'{source_ids} ({source_types}) {rel_types} {target_ids} ({target_types}) between timestamp {start_t} and {end_t}. ')
            elif dataset == 'egolife':
                res_formatted += (f'{source_ids} ({source_types}) {rel_types} {target_ids} ({target_types}) on day {video_id} between time {timeformatter(start_t)} and {timeformatter(end_t)}. ')
        return res_formatted
        
    logger.debug("---RETRIEVE ENTITY GRAPH---")
    current_task = str(state["current_task"])
    working_memory = state["working_memory"]
    query_time = state["query_time"]
    
    cb = UsageMetadataCallbackHandler()  
    entity_graph_params = eg_strictrelax_sql_query_writer.invoke({"question": state["question"] + state["candidates"], "query_time": f"{timeformatter(query_time['time'])} on {query_time['date']}", "working_memory": state["working_memory"], "current_task": state["current_task"]}, config={"callbacks": [cb]})
    state['total_tokens'].append({'eg_strictrelax_sql_query_writer' : list(cb.usage_metadata.values())[0]['total_tokens']})
    results = ""
    
    if dataset == 'egolife':
        conn_eg = sqlite3.connect(DB_ROOT / f"{dataset}/egolife_jake_entity_graph_dtonly_concatwith_fused_dt_and_gpt-4.1captions.db")
    elif dataset == 'videomme':
        selected_video = state["selected_video"]
        conn_eg = sqlite3.connect(DB_ROOT / f"{dataset}/dtonly_concatwith_fused_dt_and_llava-video-7bcaptions/videomme_{selected_video}.db")
    cursor = conn_eg.cursor()
    
    logger.debug('CURRENT STEP:', current_task)
    for q in entity_graph_params.sql_queries:
        logger.debug(q)
        try: # just ignore if llm doesn't output valid sql query
            cursor.execute(q)
            results = cursor.fetchall()
            if len(results) > 3:
                break
        except Exception as e:
            pass
    eg_query_results = get_eg_query_results(results) if len(results) > 0 else "NONE"
    working_memory += 'EntityGraph_Search: ' + str(eg_query_results) + '\n'
    logger.debug("RELEVANT CONTEXT: ",  working_memory)
    return {"working_memory": working_memory}


get_dt_attributes = lambda data_row : f"Day{data_row.day} {data_row.start_t} to {data_row.end_t} - '{data_row.transcript_english}'"

def search_and_analyze_transcripts_bm25(state):
    """
    Retrieve relevant transcripts with BM25 lexical search and analyze their relevance to the current step of the plan

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): working memory updated
    """
    logger.debug("---RETRIEVE AUDIO TRANSCRIPTS---")
    current_task = str(state["current_task"])
    working_memory = state["working_memory"]

    cb = UsageMetadataCallbackHandler()  
    text_queries = bm25_queryselector.invoke({"working_memory": working_memory, "current_task": current_task}, config={"callbacks": [cb]}).text_queries
    state['total_tokens'].append({'bm25_queryselector' : list(cb.usage_metadata.values())[0]['total_tokens']})    
    
    if dataset == 'egolife':
        eng_transcripts_to_search, df_final = get_egolife_transcripts_for_qid(state["query_time"])
    elif dataset == 'videomme':
        eng_transcripts_to_search, df_final = get_videomme_transcripts_for_vid(state['selected_video'])
        
    corpus = eng_transcripts_to_search 
    dt_retriever = bm25s.BM25() # pass corpus=corpus arg to return docs instead of doc IDs
    dt_retriever.index(bm25s.tokenize(corpus)) # takes ~400ms to tokenize and index for day 7
    results, scores = dt_retriever.retrieve(bm25s.tokenize(text_queries), k=100)
    result_idxs = np.unique(flatten_list(results))
    t_search_results = [get_dt_attributes(df_final.iloc[idx]) for idx in result_idxs]

    cb = UsageMetadataCallbackHandler()  
    analysis = transcript_analyzer.invoke({"relevant_transcripts": t_search_results, "current_task": current_task}, config={"callbacks": [cb]})
    state['total_tokens'].append({'transcript_analyzer' : list(cb.usage_metadata.values())[0]['total_tokens']})    

    working_memory += 'Transcript_Search: ' + analysis.relevance + '\n'
    state["previous_tasks"].append(state["current_task"])
    logger.debug("RELEVANT CONTEXT: ",  working_memory)
    return {"working_memory": working_memory}

    
def search_and_analyze_frames(state):
    """
    Retrieve relevant visual embeddings with SQL and optional time filters and analyze their relevance to the current step of the plan

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): update plan relevant context
    """
    logger.debug("---SEARCH AND ANALYZE FRAMES---")
    query_time = state["query_time"]
    working_memory = state["working_memory"]
    day_search_dict = str(state["day_search_dict"])
    current_task = state["current_task"]
    qtime = f"Query Time: {timeformatter(query_time['time'])[:-3]} on {query_time['date']}"
    
    cb = UsageMetadataCallbackHandler()
    framesearchquery_params = framesearchquery.invoke({"current_task": state["current_task"], "query_time": qtime, "working_memory": working_memory, "day_search_dict": day_search_dict}, config={"callbacks": [cb]})
    state['total_tokens'].append({'framesearchquery_params' : list(cb.usage_metadata.values())[0]['total_tokens']})
    analysis_result = ""

    def analyze_one_fsqp(fsqp, current_task):
        text_queries = fsqp.text_queries
        day = f'day{fsqp.day}'
        start_t = fsqp.start_t
        end_t = fsqp.end_t
    
        retrieved_image_paths = frame_retriever_sql.invoke({
            "selected_video": day,
            "queries": text_queries,
            "topk": 100,
            "start_t": start_t,
            "end_t": end_t
        })
        image_contents = get_base64imagelist_from_filepathlist(retrieved_image_paths)
        frame_timestamp_data = [extract_day_and_time(e) for e in retrieved_image_paths]    
        messages = get_frame_analyzer_context(current_task, text_queries, image_contents, frame_timestamp_data)
    
        cb = UsageMetadataCallbackHandler()  
        analysis = frame_analyzer.invoke(messages, config={"callbacks": [cb]})
      
        return fsqp, analysis, list(cb.usage_metadata.values())[0]['total_tokens']
    
    def parse_day(value):
        """Normalize day values like 'DAY1', 'day1', or '1' -> int(1)."""
        if value is None or value == "None":
            return None
        s = str(value).strip().lower()
        if s.startswith("day"):
            s = s.replace("day", "")
        try:
            return int(s)
        except ValueError:
            return None

    def parse_time(value):
        """Convert times like '11:34:41' → 113441, '113441' → 113441, None → None."""
        if value is None or str(value).lower() == "none":
            return None
        s = str(value).strip()
        s = s.replace(":", "")
        s = ''.join(filter(str.isdigit, s))
        try:
            return int(s)
        except ValueError:
            return None

    if agent_backbone == 'gemini-2.5-pro':
        fsqp_set = [
            FrameSearchQuery(
                text_queries=framesearchquery_params.text_queries,
                day=parse_day(d.get("day")),
                start_t=parse_time(d.get("start_t")),
                end_t=parse_time(d.get("end_t")),
            )
            for d in map(json.loads, framesearchquery_params.timestamp_dict)
        ]
    else:
        fsqp_set = framesearchquery_params.fs_params
        
    MAX_WORKERS = min(4, len(fsqp_set))  # or adjust manually
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(analyze_one_fsqp, fsqp, current_task) for fsqp in fsqp_set]
        tokens = 0
        for future in as_completed(futures):
            fsqp, analysis, num_tokens = future.result()
            analysis_result += f"Between {timeformatter(str(fsqp.start_t))[:-1]} and {timeformatter(str(fsqp.end_t))[:-1]} on Day {fsqp.day} : {analysis.relevance} "
            tokens += num_tokens
    working_memory += 'Frame_Search: ' + analysis_result + '\n'
    state['total_tokens'].append({'frame_analyzer' : tokens})
    
    return {"working_memory": working_memory}


def run_agentic_inference(app, vqa_question, options, transcripts, query_time, day_search_dict, working_memory_init):
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
    }
    config = RunnableConfig(recursion_limit=100)
    
    for output in app.stream(inputs, config):
        for key, value in output.items():
            # Node
            pass
    
    # Final generation
    return value


def egolife_inference():
    with open(f"{dataset_root}/EgoLifeQA/EgoLifeQA_A1_JAKE.json", "r", encoding="utf-8") as f:
        egolife_qa_jake = json.load(f)
    df_egolife = pd.DataFrame(egolife_qa_jake)

    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", default="338", help="EgoLifeQA question ID (string). Default: 338")
    parser.add_argument(
        "--tscript-search",
        default="llm",
        choices=["llm", "bm25"],
        help="Transcript search backend used inside the agent graph.",
    )
    parser.add_argument(
        "--remove-diarization",
        action="store_true",
        help="If set, remove diarization tags from transcripts.",
    )
    args = parser.parse_args()

    # get transcripts
    tscript_dict = get_egolife_diarized_transcripts(remove_diarization=args.remove_diarization)

    workflow = StateGraph(GraphState)
    
    # Define agent graph nodes
    workflow.add_node("planner_node", planner_node)
    workflow.add_node("search_entity_graph", search_entity_graph)
    workflow.add_node("search_and_analyze_frames", search_and_analyze_frames)
    workflow.add_node("search_and_analyze_transcripts_bm25", search_and_analyze_transcripts_bm25)
    workflow.add_node("retrieve_transcripts", retrieve_transcripts) # LLM search + analyze
    workflow.add_node("generate_answer", generate_answer)

    # Build agent graph
    workflow.add_edge(START, "planner_node")
    workflow.add_edge("planner_node", "search_and_analyze_frames")
    workflow.add_edge("search_and_analyze_frames", "search_entity_graph")
    if args.tscript_search == 'llm':
        workflow.add_edge("search_entity_graph", "retrieve_transcripts")
        workflow.add_conditional_edges(
            "retrieve_transcripts",
            grade_plan_completion,
            {
                "complete": "generate_answer",
                "incomplete": "planner_node",
            },
        )
    elif args.tscript_search == 'bm25':
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

    
    # Inference over the full EgoLifeQA dataset
    total_questions = len(egolife_qa_jake)
    results_json = RESULTS_ROOT / 'egagent_egolifeqa_results_all.json'
    print(f'Generating ', results_json)
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            final_prediction_list = json.load(f)
    else:
        final_prediction_list = []
    print(f'Done with {len(final_prediction_list)} / {total_questions}')
    
    completed_ids = {e['ID'] for e in final_prediction_list}
    for question_data in tqdm(egolife_qa_jake, desc="Processing"):
        results = {}
        selected_qid = question_data['ID']
        if selected_qid in completed_ids:
            print(f'Skipping {selected_qid}, already done')
            continue
            
        vqa_question = question_data['question']
        options = f"""A.{question_data['choice_a']}\nB.{question_data['choice_b']}\nC.{question_data['choice_c']}\nD.{question_data['choice_d']}"""
        answer = question_data['answer']
        query_time = question_data['query_time']
        transcripts = tscript_dict[query_time['date']]
        day_search_dict = get_egolife_daysearchdict(query_time)
        working_memory_init = "The long video is taken from the first-person perspective of Jake. "

        # wrap in try-except to handle API errors (e.g. rate limits)
        try:
            value = run_agentic_inference(app, vqa_question, options, transcripts, query_time, day_search_dict, working_memory_init)
            
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
    egolife_inference()