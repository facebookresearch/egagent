from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


WINDOW_LABELS: List[str] = ["10 sec", "30 sec", "1 min", "2 min", "10 min", "1 hour"]
WINDOW_SECONDS: List[float] = [10.0, 30.0, 60.0, 120.0, 600.0, 3600.0]


def build_df() -> pd.DataFrame:
    """
    Manually encode the LaTeX table from `table.txt` into a tidy DataFrame.
    Values are recall@W for different search configurations.
    """
    records = []

    def add_row(label: str, recalls: list[float]) -> None:
        for w_label, w_sec, r in zip(WINDOW_LABELS, WINDOW_SECONDS, recalls):
            records.append(
                {
                    "Window": w_label,
                    "WindowSeconds": w_sec,
                    "recall": r,
                    "Method": label,
                }
            )

    # Row: Gemini 2.5 Pro (uniform sampling baseline)
    add_row(
        "Gemini 2.5 Pro (Uniform Sampling)",
        [0.101, 0.160, 0.192, 0.238, 0.325, 0.410],
    )

    # Row: EGAgent (F + T) overall
    add_row(
        "EGAgent (F + T) Overall",
        [0.232, 0.241, 0.255, 0.268, 0.322, 0.418],
    )

    # Row: EGAgent (EG + F + T) M_EG
    add_row(
        "EGAgent (EG + F + T) : only EG",
        [0.127, 0.166, 0.199, 0.233, 0.413, 0.658],
    )

    # Row: EGAgent (EG + F + T) M_VIS
    add_row(
        "EGAgent (EG + F + T) : only F",
        [0.857, 0.868, 0.873, 0.875, 0.900, 0.930],
    )

    # Row: EGAgent (EG + F + T) M_AUD
    add_row(
        "EGAgent (EG + F + T) : only T",
        [0.218, 0.247, 0.261, 0.288, 0.347, 0.417],
    )

    # Row: EGAgent (EG + F + T) Overall
    add_row(
        "EGAgent (EG + F + T) Overall",
        [0.884, 0.895, 0.898, 0.902, 0.932, 0.962],
    )

    df = pd.DataFrame.from_records(records)
    df["Window"] = pd.Categorical(df["Window"], categories=WINDOW_LABELS, ordered=True)
    return df.sort_values(["WindowSeconds", "Method"])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot EgoLife recall@W curves from LaTeX table in table.txt."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("figs/egolife_recall_curves.png"),
        help="Output path for the figure.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window.",
    )
    p.add_argument(
        "--logy",
        action="store_true",
        help="Use a logarithmic scale for the y-axis.",
    )
    args = p.parse_args()

    df = build_df()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.set_style("whitegrid")

    ax = sns.lineplot(
        data=df,
        x="WindowSeconds",
        y="recall",
        hue="Method",
        marker="o",
    )

    # x-axis: log scale in seconds
    ax.set_xscale("log")
    ax.set_xlim(min(WINDOW_SECONDS) * 0.9, max(WINDOW_SECONDS) * 1.1)
    ax.set_xticks(WINDOW_SECONDS)
    ax.set_xticklabels([str(int(s)) for s in WINDOW_SECONDS])

    if args.logy:
        ax.set_yscale("log")

    ax.set_ylabel("recall@W", fontsize=16)
    ax.set_xlabel("Window Size (W) in seconds", fontsize=16)
    ax.set_ylim(0.05, 1.05)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

