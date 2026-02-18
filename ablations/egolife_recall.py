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


from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_egolife_qa_jake

_HMS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
_EG_TS_RE = re.compile(
    r"on day (\d+)\s+between time\s+(\d{2}:\d{2}:\d{2}),\d+\s+and\s+(\d{2}:\d{2}:\d{2}),\d+",
    re.MULTILINE,
)


def _hms_to_seconds(hms: str) -> int:
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def _gt_to_seconds(gt_str: str) -> int:
    # first 6 digits are HHMMSS
    return int(gt_str[0:2]) * 3600 + int(gt_str[2:4]) * 60 + int(gt_str[4:6])


def _parse_interval(item: Mapping[str, Any]) -> tuple[int, int, int] | None:
    try:
        day = int(item.get("day"))
        start_t, end_t = item.get("start_t"), item.get("end_t")
        if day <= 0 or not (_HMS_RE.match(str(start_t)) and _HMS_RE.match(str(end_t))):
            return None
        start_s, end_s = _hms_to_seconds(str(start_t)), _hms_to_seconds(str(end_t))
        return None if end_s < start_s else (day, start_s, end_s)
    except Exception:
        return None


def compute_recall(
    search_intervals: Sequence[Mapping[str, Any]],
    gt_entries: Sequence[Mapping[str, Any]],
    window: int = 10,
) -> tuple[float, int, int]:
    """
    Parameters
    - search_intervals: list of dicts like {"day": "1", "start_t": "HH:MM:SS", "end_t": "HH:MM:SS"}
    - gt_entries: list like [{"date": "DAY1", "time_list": ["17120006", ...]}, ...]
    - window: seconds before/after each GT moment
    """
    search_by_day: dict[int, list[tuple[int, int]]] = {}
    for it in search_intervals:
        parsed = _parse_interval(it)
        if parsed is None:
            continue
        day, start_s, end_s = parsed
        search_by_day.setdefault(day, []).append((start_s, end_s))

    hits = total_gt = 0
    for entry in gt_entries:
        day = int(str(entry["date"]).lower().replace("day", ""))
        times = entry.get("time_list", [])
        if day not in search_by_day:
            total_gt += len(times)
            continue
        intervals = search_by_day[day]
        for gt_str in times:
            total_gt += 1
            gt_sec = _gt_to_seconds(str(gt_str))
            a, b = gt_sec - window, gt_sec + window
            if any(not (end < a or start > b) for start, end in intervals):
                hits += 1

    return (hits / total_gt if total_gt else 0.0), hits, total_gt


def _load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_document(text: str) -> dict[str, list[str]]:
    """Split working memory into tool-wise sections."""
    if not text:
        return {"Frame_Search": [], "EntityGraph_Search": [], "Transcript_Search": []}

    text = re.sub(
        r"^\s*The long video is taken from the first-person perspective of Jake\.?\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    pattern = r"(Frame_Search:|EntityGraph_Search:|Transcript_Search:)\s*(.*?)(?=(?:Frame_Search:|EntityGraph_Search:|Transcript_Search:|$))"
    out = {"Frame_Search": [], "EntityGraph_Search": [], "Transcript_Search": []}
    for label_with_colon, content in re.findall(pattern, text, flags=re.DOTALL):
        out[label_with_colon[:-1]].append(content.strip())
    return out


def _get_eg_output_dict(toolwise_output: Mapping[str, Sequence[str]]) -> list[dict[str, str]]:
    """Extract (day,start_t,end_t) from EntityGraph_Search text."""
    out: list[dict[str, str]] = []
    for tool_output in toolwise_output.get("EntityGraph_Search", []):
        for day, start_t, end_t in _EG_TS_RE.findall(tool_output):
            out.append({"day": day, "start_t": start_t, "end_t": end_t})
    return out


def _qa_by_id():
    return {e["ID"]: e for e in load_egolife_qa_jake()}


def run_all_configs(
    *,
    windows: Sequence[int] = (5, 15, 30, 60, 300, 1800),
    eg_agent_results_json: str = (
        "egolife_results/agent_gemini-2.5-pro/"
        "egolife_agentic-gemini-2.5-pro_visual+entitygraph-dtonly-and-dtcaptionfuse+dt-oracleday_results.json"
    ),
    ft_agent_results_json: str = (
        "egolife_results/agent_gpt-4.1/egolife_agentic_oracleday-visual_notimefilter_results.json"
    ),
    ft_timestamps_dir: str = "agentic_search_timestamps/f+t",
    egft_transcript_timestamps_dir: str = "agentic_search_timestamps/eg+f+t/agent_gemini2.5-pro_transcriptsearch",
    egft_visual_timestamps_dir: str = "agentic_search_timestamps/eg+f+t/agent_gemini2.5-pro_visualsearch",
    uniform_baseline_json: str = "egolife_results/gemini-2.5-pro-uniform-sample-frames+dt-3000.json",
    uniform_timestamps_dir: str = "agentic_search_timestamps/mllm_uniform_sampling",
) -> None:
    """
    Get recall scores for all configurations.
    """
    qa_by_id = _qa_by_id()

    def _run(label: str, ids: Sequence[str], intervals_for_id) -> None:
        print(f"\n== {label} ==")
        ids = [qid for qid in ids if qid in qa_by_id]
        for w in windows:
            hits = total = 0
            for qid in ids:
                _, h, t = compute_recall(intervals_for_id(qid), qa_by_id[qid]["target_time"], window=w)
                hits += h
                total += t
            print(f"With windows={w}, EG search recall = {hits/total: .4f}")

    # --- Config 1: Gemini 2.5 Pro uniform-sampling baseline ---
    results_mllm_baseline = _load_json(uniform_baseline_json)

    misformatted_qids = [9,36,42,62,70,75,81,95,104,115,144,267,282,324,325,359,363,387,395,404,410,421,432,436]
    misformatted_qnums = {e + 1 for e in misformatted_qids}
    
    qa_list = load_egolife_qa_jake()  # needed for qnum -> ID mapping
    ids_uniform: list[str] = []
    for entry in results_mllm_baseline:
        try:
            qnum = int(entry["key"].split("-")[-1])  # 1-indexed
            txt = entry["response"]["candidates"][0]["content"]["parts"][0]["text"][7:-3]
            obj = json.loads(txt)
            _ = obj["response"][0] if qnum in misformatted_qnums else obj[0]
            ids_uniform.append(qa_list[qnum - 1]["ID"])
        except Exception:
            continue

    odir_uniform = Path(uniform_timestamps_dir)

    def _uniform_intervals(qid: str):
        p = odir_uniform / f"gemini2.5pro_qid{qid}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing timestamp file: {p}")
        return _load_json(p)


    # --- Config 2: EGAgent searching only frames and audio transcripts (F + T) ---
    agent_gpt41_ft = _load_json(ft_agent_results_json)
    ids_ft = [e["ID"] for e in agent_gpt41_ft]
    odir_ft = Path(ft_timestamps_dir)

    def _ft_intervals(qid: str):
        p = odir_ft / f"qid{qid}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing timestamp file: {p}")
        return _load_json(p)


    # --- Config 3: EGAgent searching entity graph, frames and audio transcripts (EG + F + T) ---
    agent_gemini25_egft = _load_json(eg_agent_results_json)
    ids_eg = [e["ID"] for e in agent_gemini25_egft]

    def _eg_intervals(qid: str):
        final_m = next(e["plan_relevant_context"] for e in agent_gemini25_egft if e["ID"] == qid)
        return _get_eg_output_dict(_parse_document(final_m))

    # per-tool recall for the same EG+F+T run (visual vs transcript vs EG)
    def _ids_from_qid_dir(d: Path) -> list[str]:
        ids: list[str] = []
        for p in d.glob("qid*.json"):
            stem = p.stem  # qidXYZ
            ids.append(stem[3:])
        return ids

    odir_tx = Path(egft_transcript_timestamps_dir)
    odir_vis = Path(egft_visual_timestamps_dir)

    def _dir_intervals(odir: Path, qid: str):
        p = odir / f"qid{qid}.json"
        if not p.exists():
            return []
        return _load_json(p)

    _run(
        "Gemini 2.5 Pro Uniform Sampling",
        ids_uniform,
        _uniform_intervals,
    )

    _run(
        "EGAgent (F + T) Overall",
        ids_ft, 
        _ft_intervals,
    )

    _run(
        "EGAgent (EG + F + T) M_EG",
        ids_eg,
        _eg_intervals,
    )
    _run(
        "EGAgent (EG + F + T) M_VIS",
        _ids_from_qid_dir(odir_vis),
        lambda qid: _dir_intervals(odir_vis, qid),
    )
    _run(
        "EGAgent (EG + F + T) M_AUD",
        _ids_from_qid_dir(odir_tx),
        lambda qid: _dir_intervals(odir_tx, qid),
    )
    _run(
        "EGAgent (EG + F + T) Overall",
        ids_eg,
        lambda qid: (_eg_intervals(qid) + _dir_intervals(odir_vis, qid) + _dir_intervals(odir_tx, qid)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EgoLife recall computation utilities.")
    p.add_argument(
        "--run-all",
        action="store_true",
        help="Run all recall configurations and print scores.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.run_all:
        run_all_configs()
        return 0

    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())