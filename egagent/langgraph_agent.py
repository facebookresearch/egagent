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


from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph, START

import logging
import os
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval_model import device, embed_texts_batch
from utils import *

dataset = "videomme" # videomme, egolife
agent_backbone = 'gemini-2.5-pro' # gpt-4.1 (default), gemini-2.5-pro, gpt-4o, qwen-2.5-vl-7b

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
dataset_root = EGOLIFE_ROOT if dataset == 'egolife' else VIDEO_MME_ROOT # path to EgoLife and VideoMME datasets (HuggingFace)
frames_dir = f'{dataset_root}/image_1fps_A1_JAKE' if dataset == 'egolife' else f'{dataset_root}/video_1fps'
asr_dir = f'{dataset_root}/EgoLifeCap/Transcript/A1_JAKE/' if dataset == 'egolife' else f'{dataset_root}/subtitle'

MAX_PLANNING_STEPS = 5
MAX_FRAMES_FOR_MLLM = 50 # GPT 4.1
CONTEXT_WINDOW_SIZE = 1e9 # GPT 4.1, Gemini 2.5 Pro

"""
****************   Tools   ****************   
"""

@tool
def frame_retriever_sql(start_t, end_t, selected_video: str, queries: list[str], topk: int = 5) -> list[str]:
    """
    Returns top k semantically similar image frames to text query between time start_t and end_t
    for a selected video from sqlite table
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if dataset == 'egolife':
        db_path = DB_ROOT / f"{dataset}/egolife_jake_frames.db"
        selected_video = selected_video.lower()
    elif dataset == 'videomme':
        db_path = DB_ROOT / f"{dataset}/videomme_frames_{selected_video}.db"
        selected_video = "day1"
    conn = sqlite3.connect(db_path)
    query_emb = embed_texts_batch(queries, device) # (N_q, 1152)
    results = search_sql(conn, selected_video, start_t, end_t, query_emb, topk, dataset)
    retrieved_frame_filepaths = [e[0] for e in results][:MAX_FRAMES_FOR_MLLM]
    return retrieved_frame_filepaths


def get_llm_worker(system_prompt, human_prompt, structured_llm_class, model):
    """
    Pass in custom llm BaseModel with structured output along with system and human prompts.
    Returns llm for use in graph nodes / edges.
    """
    if model in ['gpt-4.1', 'gpt-4o']:
        llm = get_vision_llm(model) # All workers currently use GPT 4.1
    elif model in ['gpt-5', 'o3']:
        llm = get_reasoning_llm(model)
    elif model == 'gemini-2.5-pro':
        llm = get_external_gemini_llm(model)
    elif model == 'qwen-2.5-vl-7b':
        llm = get_vLLM("localhost", "Qwen/Qwen2.5-VL-7B-Instruct")
        
    llm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
    )
    structured_llm = llm.with_structured_output(structured_llm_class) # pipe llm with prompt
    return llm_prompt | structured_llm


def get_llm_msg_with_imglist(system: str, human:str, image_contents: List[str]):
    """
    Pass in custom message to multimodal llm containing system and human prompts alongside base64 images
    """
    messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": human,
                    },
                    *image_contents[:MAX_FRAMES_FOR_MLLM] # e.g. GPT 4.1 refuses more than 50 images at a time
                ],
            },
        ]
    
    assert count_tokens_approximately(messages) < CONTEXT_WINDOW_SIZE # e.g. GPT 4.1 context window = 1M

    return messages


"""
********************************************************
****************   Define LLM workers   ****************
********************************************************
"""
if agent_backbone == 'gemini-2.5-pro':
    llm = get_external_gemini_llm(agent_backbone)
elif agent_backbone == 'qwen-2.5-vl-7b':
    llm = get_vLLM("localhost", "Qwen/Qwen2.5-VL-7B-Instruct")
elif agent_backbone in ['gpt-4.1', 'gpt-4o']:
    llm = get_vision_llm(agent_backbone)
elif agent_backbone in ['gpt-5', 'o3']:
    llm = get_reasoning_llm(agent_backbone)

"""
****************   Part 1: Planner, Router, Retrievers   ****************   
"""
### Planner

class MultiHopPlan(BaseModel):
    """Come up with a multi-step plan of how to answer a question about a long video."""

    plan: List[str] = Field(
        description="List of steps of the plan"
    )

planner_system = f"""You are an expert at answer questions about long videos. These questions may require multi-hop reasoning. Your job is to come 
up with a multi-step plan of all possible information that may be needed to answer the question. Each step of your plan will be routed to three search tools. The first search tool looks at transcripts with timestamps, the second looks at image frames sampled at 1 FPS, and the third looks at an entity scene graph extracted from the long video. All search tools will search for context relevant to the plan step. Keep each step concisely framed, and do not add compiling information as the final step, as this will be done automatically. You may use up to {MAX_PLANNING_STEPS} steps, but use as few as necessary to answer the question. """
if dataset == 'egolife':
    planner_system += 'Note that the question and all video frames are from the first-person (egocentric) perspective of Jake. Any references to "me" or "I" thus refer to Jake.'
planner_human = "Question: {question}, Candidates: {candidates}"

planner = get_llm_worker(planner_system, planner_human, MultiHopPlan, agent_backbone)


### Router
class RouteQuery(BaseModel):
    """Route a sub-task of a multi-step plan about long video question answering to the most relevant data source."""

    datasource: Literal["visual", "audio", "eg"] = Field(
        ...,
        description="Given a user question choose to route it to audio, visual or entity scene graph data source.",
    )

    routedstep: str = Field(
        description="The task of the plan selected for data source routing.",
    )

router_system = """You are an expert at routing a multiple-choice question about a long video to one of three data sources: video frames sampled at 1 FPS (visual), audio transcripts (audio) and an entity scene graph (eg). You are given a multi-step plan about the information needed to answer the question, a history of previously routed sub-tasks, as well as the information obtained from previous sub-tasks. You must route the current sub-task of the plan to the appropriate data source. \nThe entity scene graph contains relationships between entities (person, object, location) with timestamps and locations. Route to the eg source to query these relationships. Querying the entity graph is extremely fast and can be prioritized. Route to the visual source if you believe there may be relevant visual content to search for within the sampled image frames (moderately fast). Route to audio source if you believe there may be relevant raw audio transcripts to search to search for (search is very slow and should only be done if necessary)."""
router_human = "Multi-step plan: {plan}, Previously completed sub-tasks: {previous_tasks}, Information known so far: {working_memory}"

question_router = get_llm_worker(router_system, router_human, RouteQuery, agent_backbone)


### Video Frame Retrieval Hyperparameters
class VideoFrameRetrievalWithTimeStamps(BaseModel):
    """Text queries and time filters for text-image retrieval on a long video's 
    image frames between a specified starting and ending time."""

    text_queries: List[str] = Field(
        description="List of text queries (str), maximum three"
    )
    start_t: str = Field(
        description="Starting time to search in hhmmss format"
    )
    end_t: str = Field(
        description="Ending time to search in hhmmss format"
    )

frameret_with_timestamps_params_system = """You are a question re-writer that that rewrites the input question into concise text queries optimized for text-image retrieval on frames from a very long video sampled at 1 FPS, and specifies when to search. You are also given relevant context from previous retrieval steps. Analyze the input question and try to reason about the underlying semantic intent / meaning. Retrieval is carried out with CLIP embeddings, so keep the rewritten queries short (single word wherever possible), distinct, and unambiguous. Do not use generic common nouns that are not specific to the question and options, as these will likely return irrelevant search results using CLIP embeddings. Do not use specific named entities as text queries (such as names of non-famous people), as CLIP will not have seen these during training. If you only need one text query, only use one text query. Only use additional text queries if they are semantically distinct from one another. You are given the starting and ending time of the long video, and asked to select a starting and ending time to search. If you are unsure when to search, search the entire duration. """
if dataset == 'egolife':
    frameret_with_timestamps_params_system += 'Note that all video frames are taken from the first-person (egocentric) perspective of Jake.'
frameret_with_timestamps_params_human = """Here is the initial question: {question}\nRelevant context from previous retrieval steps: {working_memory}\n Video start time {vidstart} and end time {vidend} as HHMMSS.\nFormulate an improved set of concise text queries e.g. ['q1', 'q2', ...] and select when to search (between start_t and end_t, both formatted as HHMMSS)."""


frame_retrieval_with_timestamps_init = get_llm_worker(frameret_with_timestamps_params_system, frameret_with_timestamps_params_human, VideoFrameRetrievalWithTimeStamps, agent_backbone)


### Audio Transcript Retrieval
class AudioTranscriptRetrieval(BaseModel):
    """Describe how specific audio transcripts from a long video are relevant to a query."""

    rel_tscripts: str = Field(
        description="Selected relevant audio transcripts"
    )
    
    relevance: str = Field(
        description="Describe how the selected audio transcripts are relevant to the question"
    )

tscript_ret_system = """You are a helpful assistant who analyzes how retrieved transcripts are relevant to answer a multiple-choice question about a long video. You are given a single step of a multi-step plan for answering the multiple-choice question and a list of transcripts (which may be diarized, i.e. have speaker names annotated) over the entire long video, where each list element is a dictionary of start time, end time, transcript text."""

tscript_ret_human_egolife = """Your task is to select diarized transcripts relevant to this step of the multi-step plan: {current_task}. You are also provided context from previous retrieval steps, as they may be relevant in your selection process: {working_memory}\n Here are the full diarized transcripts of the long video: \n {audio_transcripts}. \nOnce you select all diarized transcripts that may be relevant to answer the step, describe how the diarized transcripts are relevant to the goal: {current_task}. Note that the question diarized transcripts are from the first-person (egocentric) perspective of Jake. Any references to "me" or "I" thus refer to Jake."""

tscript_ret_human_videomme = """Your task is to select transcripts relevant to this step of the multi-step plan: {current_task}. You are also provided context from previous retrieval steps, as they may be relevant in your selection process: {working_memory}\nHere are the full transcripts of the long video: '{audio_transcripts}'.\nOnce you select all transcripts that may be relevant to answer the step, describe how the audio transcripts are relevant to the goal: {current_task}. Note that the timestamps in the transcript correspond to elapsed time in the long YouTube video, and not actual time of day. It is likely that the date, location, and actual time of day may be continuously changing throughout the video."""

if dataset == 'egolife':
    tscript_retriever_oneshot = get_llm_worker(tscript_ret_system, tscript_ret_human_egolife, AudioTranscriptRetrieval, agent_backbone)
elif dataset == 'videomme':
    tscript_retriever_oneshot = get_llm_worker(tscript_ret_system, tscript_ret_human_videomme, AudioTranscriptRetrieval, agent_backbone)
if agent_backbone in ['gpt-4o', 'qwen-2.5-vl-7b']: # context window 128K is insufficient for one day of DT. Replace with GPT-4.1
    tscript_retriever_oneshot = get_llm_worker(tscript_ret_system, tscript_ret_human_videomme, AudioTranscriptRetrieval, 'gpt-4.1')

    
# Search transcript with BM25
class BM25QuerySelector(BaseModel):
    """Design text queries based on a given task to search a long video text transcript with BM25 lexical search."""

    text_queries: List[str] = Field(
        description="List of text queries to search the transcripts, maximum three."
    )

bm25_queryselector_system = """You are a question re-writer that that rewrites a given sub-task into concise text queries 
optimized for retrieving text transcripts relevant to the sub-task from a very long video. You are also given the 
relevant context from previous sub-tasks. Retrieval is done via BM25 lexical search on the text queries you select. Select a maximum
of three concise and unambiguous text queries, and minimize any words overlapping across queries."""

bm25_queryselector_human = """Your task is to design lexical search queries to search transcripts relevant to this step (sub-task) of the multi-step plan: {current_task} You are also provided context from previous retrieval steps, as they may be relevant in your text query selection process: {working_memory}"""

bm25_queryselector = get_llm_worker(bm25_queryselector_system, bm25_queryselector_human, BM25QuerySelector, agent_backbone)


### Entity Graph Retrieval
class SQLQueryEG(BaseModel):
    """SQL query to search an knowledge graph of entities and relationships between them."""
    reasoning: str
    sql_queries: List[str]
    
entitygraph_ret_system_egolife = """
You are an expert SQL reasoning assistant working over a SQLite database `entity_graph_table` with the following schema:

entity_graph_table(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day INTEGER,     # 1 to 7. Must be less than or equal to query time day
    start_t INTEGER, # e.g. 13260970 for 13:26:09,70. Should be earlier than end_t
    end_t INTEGER,   # e.g. 18401600 for 18:40:16,00
    transcript TEXT, # what was said between start_t and end_t
    source_id TEXT,  # name of the source entity e.g. Jake, Microwave, Yard
    source_type TEXT, # ("Person", "Location", "Object")
    target_id TEXT,   # name of the source entity e.g. Shure, Phone, Knife
    target_type TEXT, # ("Person", "Location", "Object")
    rel_type TEXT # ("TALKS_TO", "INTERACTS_WITH", "MENTIONS", "USES")
)

This schema represents an entity graph extracted from long egocentric video (7 days, ~8 hours a day). Each entry of the table represents a relationship in the entity graph:
source_id (source_type) -> rel_type -> target_id (target_type) which occurs between time start_t and end_t on a particular day (from 1 to 7).
e.g. Jake (Person) -> USES -> mobile phone (Object)

You are given a multiple-choice question about the long video and the time it is asked (e.g. day 6 at 15:23:41,00), as well as a specific goal designed by an expert planner.
Your job is to construct SQL queries to query the above table to answer the specific goal given by the planner.

Rules for query generation:
1. Your goal is to find relevant rows describing relationships between entities.
2. You must construct SQL queries progressively, starting with the strictest filter and relaxing step by step if no results are found.
3. Each stage should keep only the necessary filters. The order of relaxation is:

   (a) Strict: exact day, exact timestamp (start_t >=x and end_t<=y), exact source_id, exact target_id, exact rel_type.  
   (b) Relax time: same day, exact source_id/target_id, same rel_type.  
   (c) Relax day: all days, exact source_id/target_id, same rel_type. Day has to be less than or equal to the query time day. 
   (d) Relax entity match: same rel_type but use substring (`LIKE`) for source/target_id. Try to use single word for both IDs here to maximize probability of substring match.
   (e) Relax rel_type: search by entity only (no rel_type constraint).  

4. If fuzzy search is enabled (flag=True), you may suggest fuzzy candidates for entity names (e.g., “tart crust” → “Tart Shells”) using case-insensitive substring or approximate matching.
5. Always return your reasoning, and the SQL for each step, in a structured format.
6. Do not hallucinate entity names; use fuzzy or LIKE matching only to suggest similar candidates.
7. Always use SELECT * FROM entity_graph_table WHERE ... Do not use SELECT transcript or any other specific element of the schema.
8. Do not search the transcript unless you have exhausted other options.
9. Keep relaxing until the last SQL query has ONLY target_id (and optionally transcript). 
"""

entitygraph_ret_system_videomme = """
You are an expert SQL reasoning assistant working over a SQLite database `entity_graph_table` with the following schema:

entity_graph_table(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT,     # matches the name of the video. can be ignored
    start_t TEXT, # e.g. 00:26:09. Should be earlier than end_t
    end_t TEXT,   # e.g. 00:40:16
    transcript TEXT, # what was said between start_t and end_t
    source_id TEXT,  # name of the source entity e.g. Microwave, Yard
    source_type TEXT, # ("Person", "Location", "Object")
    target_id TEXT,   # name of the source entity e.g. Phone, Knife
    target_type TEXT, # ("Person", "Location", "Object")
    rel_type TEXT # ("TALKS_TO", "INTERACTS_WITH", "MENTIONS", "USES")
)

This schema represents an entity graph extracted from a long video. Each entry of the table represents a relationship in the entity graph:
source_id (source_type) -> rel_type -> target_id (target_type) which occurs between TIMESTAMP start_t and end_t. Note that these timestamps correspond to elapsed time in the long video, and not actual time of day. It is likely that the date, location and actual time of day may be continuously changing throughout the video.
e.g. Simon (Person) -> USES -> mobile phone (Object)

You are given a multiple-choice question about the long video and the time it is asked, which corresponds to the end of the long video (i.e. end_t) as well as a specific goal designed by an expert planner. Your job is to construct SQL queries to query the above table to answer the specific goal given by the planner.

Rules for query generation:
1. Your goal is to find relevant rows describing relationships between entities.
2. You must construct SQL queries progressively, starting with the strictest filter and relaxing step by step if no results are found.
3. Each stage should keep only the necessary filters. The order of relaxation is:

   (a) Strict: exact timestamp (start_t >=x and end_t<=y), exact source_id, exact target_id, exact rel_type.  
   (b) Relax time: exact source_id/target_id, same rel_type.  
   (c) Relax entity match: same rel_type but use substring (`LIKE`) for source/target_id. Try to use single word for both IDs here to maximize probability of substring match.
   (d) Relax rel_type: search by entity only (no rel_type constraint).  

4. If fuzzy search is enabled (flag=True), you may suggest fuzzy candidates for entity names (e.g., “tart crust” → “Tart Shells”) using case-insensitive substring or approximate matching.
5. Always return your reasoning, and a valid SQL query for each step, in a structured format.
6. Do not hallucinate entity names; use fuzzy or LIKE matching only to suggest similar candidates.
7. Always use SELECT * FROM entity_graph_table WHERE ... Do not use SELECT transcript or any other specific element of the schema.
8. Do not search the transcript unless you have exhausted other options.
9. Keep relaxing until the last SQL query has ONLY target_id (and optionally transcript). 
"""

entitygraph_ret_human = """
User question: {question} asked at {query_time}, and relevant context gathered thus far: '{working_memory}'.
Your job is to create a SQL query to answer this specific goal given by an expert planner: '{current_task}'.

Return a JSON object with:
- "reasoning": a short summary of your search plan and why constraints are relaxed.
- "sql_queries": an ordered list of candidate SQL strings to execute, from strictest to most relaxed. 

You do not need to run SQL — only generate the statements.
"""

if dataset == 'egolife':
    eg_strictrelax_sql_query_writer = get_llm_worker(entitygraph_ret_system_egolife, entitygraph_ret_human, SQLQueryEG, agent_backbone)
elif dataset == 'videomme':
    eg_strictrelax_sql_query_writer = get_llm_worker(entitygraph_ret_system_videomme, entitygraph_ret_human, SQLQueryEG, agent_backbone)


## Parallel Frame Search over days previous to query
class FrameSearchQuery(BaseModel):
    """Given question and relevant context, select text queries and time filters to search a long video."""
    text_queries: List[str] = Field(description="List of text queries (str), maximum three")
    day: Literal[0, 1, 2, 3, 4, 5, 6, 7, None] = Field(description="day to search")
    start_t: Optional[int] = Field(description="Starting time to search in hhmmss format")
    end_t: Optional[int] = Field(description="Ending time to search in hhmmss format")

class FrameSearchQueries(BaseModel):
    fs_params: List[FrameSearchQuery] # frame search parameters

class FrameSearchQueries_Gemini(BaseModel):
    """Given question and relevant context, select text queries and time filters to search a long video."""
    text_queries: List[str] = Field(description="List of text queries (str), maximum three")
    timestamp_dict: List[str] = Field(description="collection of {day, start_t, end_t} dicts to search")

framesearchquery_system = """You are a question re-writer that that rewrites the input question into concise text queries 
optimized for text-image retrieval on frames from a very long video sampled at 1 FPS, and specifies when to search. You are also given 
relevant context from previous retrieval steps. Retrieval is carried out with CLIP embeddings, so keep the rewritten queries short (single word wherever possible), 
distinct, and unambiguous. Do not use generic common nouns or times of day (e.g. noon, afternoon) that are not specific to objects or actions present in the question and options, as these will likely return irrelevant search results using CLIP embeddings. Do not use specific named entities as text queries (such as names of non-famous people), as CLIP will not have seen these during training.

You are given the starting and ending time of the long video, and asked to select the day and a start time and end time to search between.
If you are unsure when to search, search the entire duration (i.e. the entire day). You may search any day and time before the query time."""

framesearchquery_human = """Here is the initial question: {current_task} asked at {query_time}\nRelevant context from previous retrieval steps: {working_memory}\nHere is a dictionary containing the start and end times of all days formatted as HHMMSS: {day_search_dict}. You may search any day between the start_t and end_t of that day. Select a set of 1 to 3 concise text queries e.g. ['q1'] or ['q1', 'q2', 'q3'] for each day you would like to search, and optionally when during that day to search (between start_t and end_t). If you only need one text query, only use one text query. Only use additional text queries if they are semantically distinct from one another."""

framesearchquery_human_gemini = """Here is the initial question: {current_task} asked at {query_time}\nRelevant context from previous retrieval steps: {working_memory}\nHere is a dictionary containing the start and end times of all days formatted as HHMMSS: {day_search_dict}. You may search any day between the start_t and end_t of that day. Select a set of 1 to 3 concise text queries e.g. ['q1'] or ['q1', 'q2', 'q3'] for each day you would like to search. Also output a timestamp_dict containing a collection of {{day, start_t, end_t}} dictionaries over all days you would like to search, and when during that day to search (between start_t and end_t). Always output the timestamp dict in json format:

timestamp_dict = 
[
    {{
        "day":1, 
        "start_t":110942, 
        "end_t":220549
    }}, 
    {{
        "day":2, 
        "start_t":104425,
        "end_t":225824
    }}
]

If you only need one text query, only use one text query. Only use additional text queries if they are semantically distinct from one another."""

if agent_backbone in ['gemini-2.5-pro']:
    framesearchquery = get_llm_worker(framesearchquery_system, framesearchquery_human_gemini, FrameSearchQueries_Gemini, agent_backbone)
else:
    framesearchquery = get_llm_worker(framesearchquery_system, framesearchquery_human, FrameSearchQueries, agent_backbone)


"""
****************   Part 2: Retrieval Analyzers   ****************   
"""

### Analysis of Retrieved Video Frames
class AnalyzeVideoFrames(BaseModel):
    """Relevance description of retrieved image frames to a question."""

    relevance: str = Field(
        description="Describe how the image frames are relevant to the question"
    )

def get_frame_analyzer_context(question: str, text_queries: List[str], image_contents: List[str], frame_timestamp_data: List[str]):
    human = f"Question: {question} and associated text queries: {text_queries} \n\nRetrieved video frames are attached below, and they have a one-to-one correspondence with these timestamps: {frame_timestamp_data}\n"
    return get_llm_msg_with_imglist(frame_analyzer_system, human, image_contents)

frame_analyzer_system = """You are an expert at analyzing sequences of video frames. Given a list of retrieved video 
    frames and the text queries used to retrieve them, you must describe how the video frames are relevant to the 
    question specified, and which frames are relevant (by the day and timestamp). Be descriptive about visual elements in the scene that are relevant to the query. If you find 
    nothing relevant, you should say so."""

frame_analyzer = llm.with_structured_output(AnalyzeVideoFrames)


### Analysis of Retrieved  Audio Transcripts
class AnalyzeTranscripts(BaseModel):
    """Relevance description of transcripts to a question."""

    relevance: str = Field(
        description="Describe how the transcripts are relevant to the question"
    )

tscripts_analyzer_system = """You are an expert at analyzing audio transcripts from long videos. Given the audio transcripts, you must describe how they are relevant to a provided goal. Keep your analysis concise and in neutral tone. Refer to specific audio transcripts and timestamps if provided."""
tscripts_analyzer_human = "Transcripts: \n\n {relevant_transcripts} \n\n Goal: {current_task}"

transcript_analyzer = get_llm_worker(tscripts_analyzer_system, tscripts_analyzer_human, AnalyzeTranscripts, agent_backbone)


"""
****************   Part 3: Graders   ****************   
"""

### Grade Multi-step plan completion
class GradePlanCompletion(BaseModel):
    """Binary score to assess if all steps of the plan have been addressed."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

plangrader_system = """You are a grader assessing whether all steps of a plan about video question answering have been completed, given the previously completed steps. \n Give a binary score 'yes' or 'no'. Yes' means that the context covers all aspects of the plan."""
plangrader_human = "Original Steps of Plan: \n {plan} \n Completed Steps: {previous_tasks} \n\n Context History: {working_memory}. If the number of completed steps so far is less than the number of original steps, answer yes only if you are very confident that you don't need any more context to answer the question."

plan_grader = get_llm_worker(plangrader_system, plangrader_human, GradePlanCompletion, agent_backbone)


"""
****************   Part 4: Answer Generation   ****************   
"""

### Predict MCQ answer from 4 candidates
class FinalAnswer(BaseModel):
    """MCQ Prediction and Justification."""

    mcq_prediction: Literal["A", "B", "C", "D"] = Field(
        ...,
        description="Given a question, relevant context, and four options, predict the best option.",
    )

    justification: str = Field(
        description="Justify your predicted option based on the question and relevant context."
    )

# Prompt
mcqprediction_system = """Select the best answer to the following multiple-choice question based on the provided context and provide a justification for your answer. The provided context contains information retrieved from three data sources, which are marked by EntityGraph_Search, Transcript_Search, and Frame_Search. In your justification, reference specific portions of this context that lead to your prediction."""
mcqprediction_human = "Multiple-choice question: '{question}' \nFour Options: {candidates}\nThe question is asked at {query_time}\nRelevant Context: {context}"

generate_finalanswer = get_llm_worker(mcqprediction_system, mcqprediction_human, FinalAnswer, agent_backbone)




"""
**************************************************************
****************   Construct Workflow Graph   ****************
**************************************************************
"""


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: multiple-choice question
        candidates: four options for MCQ
        selected_video: name of selected video
        vidstart: in hhmmss
        vidend: in hhmmss
        start_t: when to begin tool search
        end_t: when to end tool search
        day_search_dict: start and end times of all days
        query_time: the time (and day) the query is asked, if provided
        audio_transcripts: full audio transcripts of long video
        plan: decompose the question into multi-step plan
        working_memory: accumulate cross-modal evidence
        current_task: current planner task being executed
        previous_tasks: planner tasks previously completed
        retriever_queries: queries to search for relevant video frames
        relevant_frame_paths: list of retrieved relevant video frames
        answer: VQA agent predicted answer to MCQ
        total_tokens: track total token usage over entire agent
    """

    question: str
    candidates: List[str]
    selected_video: str
    vidstart: int
    vidend: int
    start_t: int
    end_t: int
    day_search_dict: str
    query_time: str
    audio_transcripts: List[str]
    plan: List[str]
    working_memory: str
    current_task: str
    previous_tasks: List[str]
    retriever_queries: List[str]
    relevant_frame_paths: List[str]
    answer: str
    total_tokens: List[str]

"""
****************   Graph Nodes   ****************   
"""


def planner_node(state):
    """
    Come up with multi-step plan for long video question answering

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, plan, that contains multi-step plan
    """
    plan = state["plan"]
    num_completed_steps = len(state["previous_tasks"]) - 1 
    # Only come up with plan on first invocation
    if num_completed_steps == 0:
        logger.debug("---MULTI STEP PLAN---")
        cb = UsageMetadataCallbackHandler()
        resp = planner.invoke({"question": state["question"], "candidates": state["candidates"]}, config={"callbacks": [cb]})
        state['total_tokens'].append({'planner' : list(cb.usage_metadata.values())[0]['total_tokens']})
        plan = resp.plan
        for i,p in enumerate(plan):
            logger.debug(f'{i+1}. {p}')
    state["current_task"] = str(plan[num_completed_steps])
    return {"plan": plan, "current_task": plan[num_completed_steps]}


def get_retrieval_params_sql(state):
    """
    Transform the questions to text queries and select number of video frames (images) of long video to search.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates text_queries and start and end times for search
    """

    logger.debug("---GET RETRIEVAL PARAMS---")
    # Get text queries and topk for text-image retrieval.  Pass in the current plan step as question
    cb = UsageMetadataCallbackHandler()
    retrieval_params = frame_retrieval_with_timestamps_init.invoke({"question": state["current_task"], "working_memory": state["working_memory"], "vidstart": state["vidstart"], "vidend": state["vidend"]}, config={"callbacks": [cb]})
    state['total_tokens'].append({'get_retrieval_params_sql' : list(cb.usage_metadata.values())[0]['total_tokens']})
    logger.debug(state["current_task"], state["working_memory"], state["vidstart"], state["vidend"])
    return {"retriever_queries": retrieval_params.text_queries, "start_t": retrieval_params.start_t, "end_t": retrieval_params.end_t}


def retrieve_frames_sql(state):
    """
    Retrieve relevant video frames from sql

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, relevant_frame_paths
    """
    logger.debug("---RETRIEVE FRAMES SQL---")
    cb = UsageMetadataCallbackHandler()
    retrieval_params = frame_retrieval_with_timestamps_init.invoke({"question": state["current_task"], "working_memory": state["working_memory"], "vidstart": state["vidstart"], "vidend": state["vidend"]}, config={"callbacks": [cb]})
    state['total_tokens'].append({'frame_retrieval_with_timestamps_init' : list(cb.usage_metadata.values())[0]['total_tokens']})

    text_queries = retrieval_params.text_queries
    start_t = retrieval_params.start_t # hardcode to 0 to remove time filtering
    end_t = retrieval_params.end_t # hardcode to 2359 to remove time filtering
    logger.debug(text_queries, start_t, end_t)

    retrieved_image_paths = frame_retriever_sql.invoke({"selected_video": state["selected_video"], "queries": text_queries, "topk": 50, "start_t": start_t, "end_t": end_t})

    return {"relevant_frame_paths": retrieved_image_paths, "retriever_queries": text_queries}

    
def retrieve_transcripts(state):
    """
    Retrieve relevant audio transcripts

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): working memory updated
    """
    logger.debug("---RETRIEVE AUDIO TRANSCRIPTS---")
    current_task = str(state["current_task"])
    audio_transcripts = str(state["audio_transcripts"])
    working_memory = state["working_memory"]
    
    cb = UsageMetadataCallbackHandler()
    relevant_transcripts = tscript_retriever_oneshot.invoke({"current_task": current_task, "working_memory": state["working_memory"], "audio_transcripts": audio_transcripts}, config={"callbacks": [cb]})
    state['total_tokens'].append({'retrieve_transcripts_llm' : list(cb.usage_metadata.values())[0]['total_tokens']})
    state["previous_tasks"].append(current_task)
    working_memory += 'Transcript_Search: ' + relevant_transcripts.relevance + '\n'
    logger.debug("RELEVANT CONTEXT: ",  working_memory)
    return {"working_memory": working_memory}


# Extract day and time from filepath (default egolife filepath format)
extract_day_and_time = lambda filepath: f'{timeformatter(filepath.split("/")[-1][:-4])[:-3]} on {filepath.split("/")[-2]}'

def analyze_retrieved_frames(state):
    """
    Analyze how a sequence of long video image frames are relevant to the question.
    
    Args:
        state (dict): The current graph state
    
    Returns:
        state (dict): Append to working_memory
    """
    logger.debug("---ANALYZE RETRIEVED FRAMES---")
    working_memory = state["working_memory"]
    image_contents = get_base64imagelist_from_filepathlist(state["relevant_frame_paths"])
    frame_timestamp_data = [extract_day_and_time(e) for e in state["relevant_frame_paths"]]
    messages = get_frame_analyzer_context(state["current_task"], state["retriever_queries"], image_contents, frame_timestamp_data)
    cb = UsageMetadataCallbackHandler()
    analysis = frame_analyzer.invoke(messages, config={"callbacks": [cb]})
    state['total_tokens'].append({'analyze_retrieved_frames' : list(cb.usage_metadata.values())[0]['total_tokens']})
    working_memory += 'Frame_Search: ' + analysis.relevance + '\n'
    return {"working_memory": working_memory}


def generate_answer(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, answer, that contains LLM predicted answer
    """
    logger.debug("---GENERATE ANSWER---")
    documents = f"""Plan:{state["plan"]}\n\n Context relevant to plan: {state["working_memory"]}"""
    query_time = state["query_time"]
    qtime = f"Query Time: {timeformatter(query_time['time'])[:-3]} on {state['query_time']['date']}"
    
    cb = UsageMetadataCallbackHandler()
    rag_answer = generate_finalanswer.invoke({"context": documents, "question": state['question'], "candidates": state['candidates'], "query_time": qtime}, config={"callbacks": [cb]})
    state['total_tokens'].append({'generate_finalanswer' : list(cb.usage_metadata.values())[0]['total_tokens']})

    return {"answer": rag_answer, "plan": state["plan"], "working_memory": state["working_memory"], "total_tokens": state["total_tokens"]}


"""
****************   Graph Edges   ****************   
"""


def route_plan(state):
    """
    Route sub-task to entity graph, visual, or transcript search.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    logger.debug("---ROUTE PLAN---")
    working_memory = state["working_memory"]
    cb = UsageMetadataCallbackHandler()
    source = question_router.invoke({"plan": state["plan"], "previous_tasks": state["previous_tasks"], "working_memory": working_memory}, config={"callbacks": [cb]})
    if source.datasource == "visual":
        logger.debug("\t---ROUTE PLAN STEP TO VISUAL SEARCH---")
        logger.debug("Plan step: ", state["current_task"])
        return "visual"
    elif source.datasource == "audio":
        logger.debug("\t---ROUTE PLAN STEP TO AUDIO SEARCH---")
        logger.debug("Plan step: ", state["current_task"])
        return "audio"
    elif source.datasource == "eg":
        logger.debug("\t---ROUTE PLAN STEP TO ENTITY GRAPH SEARCH---")
        logger.debug("Plan step: ", state["current_task"])
        return "eg"

        
def grade_plan_completion(state):
    """
    Decide if all steps of plan are complete.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    logger.debug("---GRADE PLAN---")
    num_completed_steps = len(state["previous_tasks"]) - 1
    logger.debug(f"{num_completed_steps} STEPS COMPLETED")
    cb = UsageMetadataCallbackHandler()
    source = plan_grader.invoke({"plan": state["plan"], "previous_tasks": state["previous_tasks"], "working_memory": state["working_memory"]}, config={"callbacks": [cb]})
    state['total_tokens'].append({'plan_grader' : list(cb.usage_metadata.values())[0]['total_tokens']})

    completion = source.binary_score
    if num_completed_steps == len(state["plan"]):
        logger.debug("---ALL STEPS OF PLAN ATTEMPTED BUT NOT ADDRESSED---")
        return "complete"
    elif completion == "yes":
        logger.debug("---ALL STEPS OF PLAN ADDRESSED---")
        return "incomplete"
    elif completion == "no":
        logger.debug("---ALL STEPS OF PLAN NOT YET ADDRESSED---")
        return "incomplete"