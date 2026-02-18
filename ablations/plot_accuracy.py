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
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import EGOLIFE_ROOT, RESULTS_ROOT


EGOLIFE_CATEGORIES = ["EntityLog", "EventRecall", "HabitInsight", "RelationMap", "TaskMaster"]
EGOBUTLER_GEMINI15PRO_ACCURACY = {
    "EntityLog": 36.0,
    "EventRecall": 37.3,
    "HabitInsight": 45.9,
    "RelationMap": 30.4,
    "TaskMaster": 34.9,
} # as reported in https://arxiv.org/abs/2503.03803


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_mcq_prediction(content: Any) -> Optional[str]:
    """Extract Multiple-Choice Question (MCQ) answer predicted by the agent."""
    if content is None:
        return None
    if isinstance(content, dict) and "mcq_prediction" in content:
        return content.get("mcq_prediction")
    if not isinstance(content, str):
        return None

    s = _strip_code_fences(content).strip()
    s = s.replace("response :\n", "").strip()

    # JSON (list/dict) parse first.
    try:
        obj = json.loads(s)
        if isinstance(obj, list) and obj:
            obj0 = obj[0]
            if isinstance(obj0, dict):
                return obj0.get("mcq_prediction")
        if isinstance(obj, dict):
            if "mcq_prediction" in obj:
                return obj.get("mcq_prediction")
            if isinstance(obj.get("response"), list) and obj["response"]:
                obj0 = obj["response"][0]
                if isinstance(obj0, dict):
                    return obj0.get("mcq_prediction")
    except Exception:
        pass

    # Python-literal fallback (some logs store single-quoted dicts/lists).
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and obj:
            obj0 = obj[0]
            if isinstance(obj0, dict):
                return obj0.get("mcq_prediction")
        if isinstance(obj, dict):
            return obj.get("mcq_prediction")
    except Exception:
        pass

    # Regex fallback.
    m = re.search(r"""["']mcq_prediction["']\s*:\s*["']([^"']+)["']""", s)
    return m.group(1) if m else None


def load_egolife_qa(egolife_root: Path) -> List[Dict[str, Any]]:
    qa_path = egolife_root / "EgoLifeQA" / "EgoLifeQA_A1_JAKE.json"
    with qa_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_qa_by_id(qa: Iterable[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    # Normalize IDs to string to match result files
    return {str(q.get("ID")): q for q in qa if "ID" in q}


def ensure_answer_type_pred(
    records: List[Dict[str, Any]], qa_by_id: Optional[Dict[Any, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        r = dict(r)
        qid_raw = r.get("ID")
        qid = str(qid_raw) if qid_raw is not None else None
        q = qa_by_id.get(qid) if (qa_by_id and qid is not None) else None
        # If we can map to EgoLifeQA, always override with dataset answer/type.
        if q:
            r["answer"] = q.get("answer")
            r["type"] = q.get("type")
        else:
            r.setdefault("answer", None)
            r.setdefault("type", None)

        if "mcq_prediction" not in r or r.get("mcq_prediction") is None:
            # try notebook-style stored raw response
            r["mcq_prediction"] = extract_mcq_prediction(r.get("content"))

        out.append(r)
    return out


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_gemini_uniform_sampling(
    raw: List[Dict[str, Any]], qa: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Parse `egolife_results/gemini-2.5-pro-uniform-sample-frames+dt-3000.json`
    into [{ID, answer, mcq_prediction, ...}].
    """
    qa_by_idx = {i + 1: qa[i] for i in range(len(qa))}

    out: List[Dict[str, Any]] = []
    for entry in raw:
        try:
            qid = int(str(entry["key"]).split("-")[-1])
            text = entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
            payload = json.loads(_strip_code_fences(text))
            if isinstance(payload, list) and payload:
                obj = payload[0]
            elif isinstance(payload, dict):
                if isinstance(payload.get("response"), list) and payload["response"]:
                    obj = payload["response"][0]
                else:
                    obj = payload
            else:
                continue
            qa_item = qa_by_idx.get(qid)
            if not qa_item:
                continue
            out.append(
                {
                    "ID": qa_item["ID"],
                    "answer": qa_item.get("answer"),
                    "mcq_prediction": obj.get("mcq_prediction") if isinstance(obj, dict) else None,
                    "type": qa_item.get("type"),
                    "total_token_count": entry.get("response", {}).get("usageMetadata", {}).get(
                        "totalTokenCount"
                    ),
                }
            )
        except Exception:
            continue
    return out


def per_type_accuracy_df(records: List[Dict[str, Any]], model_label: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["type", "accuracy", "Model"])
    df = pd.DataFrame(records)
    for col in ("type", "answer", "mcq_prediction"):
        if col not in df.columns:
            df[col] = None
    df = df.dropna(subset=["type", "answer", "mcq_prediction"])
    if df.empty:
        return pd.DataFrame(columns=["type", "accuracy", "Model"])
    df["accuracy"] = (df["answer"] == df["mcq_prediction"]) * 100.0  # same core logic as eval.py
    out = df.groupby("type", as_index=False)["accuracy"].mean()
    out["Model"] = model_label
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot EgoLife category-wise accuracy bars.")
    p.add_argument(
        "--results-json",
        type=Path,
        default=RESULTS_ROOT
        / "agent_gemini-2.5-pro/egolife_agentic-gemini-2.5-pro_visual+entitygraph-dtonly-and-dtcaptionfuse+dt-llmsearch_results.json",
        help="Primary results JSON (list of dicts).",
    )
    p.add_argument(
        "--uniform-json",
        type=Path,
        default=RESULTS_ROOT / "gemini-2.5-pro-uniform-sample-frames+dt-3000.json",
        help="Gemini uniform-sampling raw JSON.",
    )
    p.add_argument(
        "--egolife-root",
        type=Path,
        default=Path(EGOLIFE_ROOT),
        help="EgoLife dataset root (must contain EgoLifeQA/EgoLifeQA_A1_JAKE.json).",
    )
    p.add_argument("--out", type=Path, default=Path("figs/category_wise_barplot.png"))
    p.add_argument("--show", action="store_true", help="Show the plot window.")
    args = p.parse_args()

    qa = load_egolife_qa(args.egolife_root)
    qa_by_id = index_qa_by_id(qa)

    results_raw = load_json(args.results_json)

    if not isinstance(results_raw, list):
        raise SystemExit(f"--results-json must be a list, got {type(results_raw)}")
    results = ensure_answer_type_pred(results_raw, qa_by_id)

    uniform_raw = load_json(args.uniform_json)
    if isinstance(uniform_raw, list) and uniform_raw and isinstance(uniform_raw[0], dict) and "key" in uniform_raw[0]:
        uniform = parse_gemini_uniform_sampling(uniform_raw, qa)
    else:
        uniform = ensure_answer_type_pred(uniform_raw if isinstance(uniform_raw, list) else [], qa_by_id)

    df_a = per_type_accuracy_df(results, "Gemini 2.5 Pro EGAgent")
    df_b = per_type_accuracy_df(uniform, "Gemini 2.5 Pro (Uniform Sample)")
    df_ego = pd.DataFrame(
        [{"type": k, "accuracy": v, "Model": "EgoButler (Gemini 1.5 Pro)"} for k, v in EGOBUTLER_GEMINI15PRO_ACCURACY.items()]
    )
    df_plot = pd.concat([df_ego, df_b, df_a], ignore_index=True)
    df_plot["type"] = pd.Categorical(df_plot["type"], categories=EGOLIFE_CATEGORIES, ordered=True)
    df_plot = df_plot.sort_values(["type", "Model"], ascending=False)

    import matplotlib.pyplot as plt
    import seaborn as sns

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=df_plot, x="type", y="accuracy", hue="Model", palette="Set2")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", label_type="edge", padding=3, fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=18)
    plt.xlabel("", fontsize=18)
    plt.ylim(0, 100)
    plt.legend(title="", loc="upper left", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()