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


import json 
from paths import RESULTS_ROOT
from utils import load_egolife_qa_jake

## Load and clean up formatting of EgoLifeQA
egolife_qa_jake = load_egolife_qa_jake()

## Gemini 2.5 Pro Uniform Sampling (3000 frames + transcripts) eval
print(f'Evaluating Gemini 2.5 Pro Uniform Sampling (3000 f + t)')

mllm_baseline = RESULTS_ROOT / 'gemini-2.5-pro-uniform-sample-frames+dt-3000.json'
with open(mllm_baseline, 'r') as file:
    results_mllm_baseline = json.load(file)

correct = 0
total = 0
# with Gemini 2.5 Pro, some question IDs have mismatched formats of LLM output, which we handle below
misformatted_qids = [9,36,42,62,70,75,81,95,104,115,144,267,282,324,325,359,363,387,395,404,410,421,432,436]
total_token_count = []

results_gemini25_uniformsample = []

for i in range(len(results_mllm_baseline)):
    results = {}
    qid = int(results_mllm_baseline[i]['key'].split("-")[-1])
    answer = egolife_qa_jake[qid-1]['answer']
    try:
        entry = [e for e in results_mllm_baseline if int(e['key'].split("-")[-1]) == (qid)][0]
        if qid in [e+1 for e in misformatted_qids]:
            mcq_pred = json.loads(entry['response']['candidates'][0]['content']['parts'][0]['text'][7:-3])['response'][0]['mcq_prediction']
            just = json.loads(entry['response']['candidates'][0]['content']['parts'][0]['text'][7:-3])['response'][0]['justification']
        else:
            mcq_pred = json.loads(entry['response']['candidates'][0]['content']['parts'][0]['text'][7:-3])[0]['mcq_prediction']
            just = json.loads(entry['response']['candidates'][0]['content']['parts'][0]['text'][7:-3])[0]['justification']

        results['ID'] = egolife_qa_jake[qid-1]['ID']
        results['answer'] = answer
        results['mcq_prediction'] = mcq_pred
        results['total_token_count'] = entry['response']['usageMetadata']['totalTokenCount']
        results['just'] = just
        results_gemini25_uniformsample.append(results)
        correct += mcq_pred == answer
        total += 1
        total_token_count.append(entry['response']['usageMetadata']['totalTokenCount'])
    except Exception as e:
        continue

print(f'Acc = {(correct)} / {len(results_mllm_baseline)} = {(correct) / len(results_mllm_baseline) * 100: .2f}%')


## EGAgent Eval
print(f'Evaluating EGAgent')

def count_tokens(results):
    """Count total tokens used by the MLLM."""
    tokens = [e['total_tokens'] for e in results if 'total_tokens' in e.keys()]
    return tokens

def print_acc(dataset, agent_backbone, config, results):
    """Print accuracy of the MLLM."""
    correct = [e for e in results if e['answer'] == e['mcq_prediction']]
    print(f'{dataset}: {agent_backbone} with {config}, Acc = {len(correct)} / {len(results)} = {len(correct) / len(results) * 100: .2f}%')

def get_eg_f_t_agent_results(agent_backbone = 'gpt-4.1'):
    """
    Get results of the EGAgent using three search tools:
    1. Visual frame search
    2. Entity graph search on entity graph extracted from fused diarized transcript (DT) and captions
    3. LLM search on raw diarized transcript (DT)
    """
    results_json = RESULTS_ROOT / f'agent_{agent_backbone}/egolife_agentic-{agent_backbone}_visual+entitygraph-dtonly-and-dtcaptionfuse+dt-llmsearch_results.json'
    with open(results_json, 'r') as file:
        agent_results = json.load(file)
    return agent_results

dataset = 'egolife'
agent_backbone = 'gpt-4.1'
results_json = RESULTS_ROOT / f'agent_{agent_backbone}/egolife_agentic-{agent_backbone}_visual+dt-llmsearch_results.json'
with open(results_json, 'r') as file:
    agent_gpt41_ft = json.load(file)
print_acc(dataset, 'gpt-4.1', "F + T", agent_gpt41_ft)

agent_gpt41_egft = get_eg_f_t_agent_results('gpt-4.1')
agent_gpt4o_egft = get_eg_f_t_agent_results('gpt-4o')
agent_gemini25_egft = get_eg_f_t_agent_results('gemini-2.5-pro')
print_acc(dataset, 'gpt-4.1', 'EG + F + T', agent_gpt41_egft)
print_acc(dataset, 'gpt-4o', 'EG + F + T', agent_gpt4o_egft)
print_acc(dataset, 'gemini-2.5-pro', 'EG + F + T', agent_gemini25_egft)