"""Reproducible slice of Playground S6E8 — Predicting Smartphone Addiction.

The full stack (mask_student, OOF 0.970158) lives in the experiment repo.
This script recomputes the label balance and single-feature AUCs from
train.csv, and replots the recorded OOF / public-leaderboard comparison
with a full 0.5–1.0 scale beside the zoomed view.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
TRAIN_PATH = DATA_DIR / "train.csv"
FIG_DIR = ROOT / "figures"
TARGET = "addicted_label"

# Recorded scores from the completed experiment (Mask_Student.ipynb).
# Public leaderboard values are shown at Kaggle's displayed precision.
RECORDED = {
    "mask_student": {"oof": 0.970158, "public": 0.97093},
    "mask_r3": {"oof": 0.970023, "public": 0.97099},
}

NUMERIC = [
    "age",
    "daily_screen_time_hours",
    "social_media_hours",
    "gaming_hours",
    "work_study_hours",
    "sleep_hours",
    "notifications_per_day",
    "app_opens_per_day",
    "weekend_screen_time",
]


def load_train() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise SystemExit(
            "train.csv was not found at "
            f"{TRAIN_PATH}.\n"
            "Kaggle Playground Series S6E8 data is distributed under the "
            "competition rules and is not included in this repository.\n"
            "Download train.csv from the competition data page and point "
            "DATA_DIR at its folder:\n"
            "  export DATA_DIR=/path/to/folder\n"
            "  python analysis.py"
        )
    df = pd.read_csv(TRAIN_PATH)
    missing = [c for c in NUMERIC + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"train.csv is missing columns: {missing}")
    return df


def label_balance(df: pd.DataFrame) -> pd.Series:
    counts = df[TARGET].value_counts().sort_index()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    labels = [f"{int(k)} ({'not addicted' if int(k) == 0 else 'addicted'})" for k in counts.index]
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#F58518"])
    total = int(counts.sum())
    for bar, n in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{n:,}\n({n / total:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Rows")
    ax.set_title(f"Label balance in train.csv (n = {total:,})")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_label_balance.png", dpi=140)
    plt.close(fig)
    return counts


def single_feature_auc(df: pd.DataFrame) -> pd.Series:
    scores = {}
    for col in NUMERIC:
        pair = df[[col, TARGET]].dropna()
        y = pair[TARGET].to_numpy()
        x = pair[col].to_numpy(dtype=float)
        # Orient the score so AUC is at least 0.5. Direction is reported separately.
        auc = roc_auc_score(y, x)
        if auc < 0.5:
            auc = roc_auc_score(y, -x)
            direction = "lower value, higher risk"
        else:
            direction = "higher value, higher risk"
        scores[col] = (auc, direction)
    table = pd.Series({k: v[0] for k, v in scores.items()}).sort_values(ascending=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(table.index[::-1], table.values[::-1], color="#4C78A8")
    ax.axvline(0.5, color="#666666", lw=0.8, ls="--")
    ax.axvline(0.970158, color="#E45756", lw=1.2, label="mask_student OOF 0.970")
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC (feature oriented toward the addicted class)")
    ax.set_title("Each raw feature alone, versus the stacked model")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_single_feature_auc.png", dpi=140)
    plt.close(fig)

    print("Single-feature ROC-AUC")
    for col, auc in table.items():
        print(f"  {auc:.4f}  {col}  ({scores[col][1]})")
    return table


def rank_flip_chart() -> None:
    """Replot recorded scores. These are not recomputed from train.csv."""
    names = ["mask_student", "mask_r3"]
    oof = [RECORDED[n]["oof"] for n in names]
    public = [RECORDED[n]["public"] for n in names]
    colors = {"mask_student": "#E45756", "mask_r3": "#4C78A8"}

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))

    def draw(ax, ymin, ymax, title):
        xs = [0, 1]
        for name, y0, y1 in zip(names, oof, public):
            ax.plot(xs, [y0, y1], color=colors[name], lw=1.6, ls="--", alpha=0.85)
            ax.scatter(xs, [y0, y1], color=colors[name], s=42, zorder=3, label=name)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [
                "5-fold OOF\n691,369 labeled rows",
                "Public leaderboard\n~20% of test",
            ]
        )
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("ROC-AUC")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")

    draw(axes[0], 0.96990, 0.97115, "Zoomed view (rank flip is visible)")
    draw(axes[1], 0.50, 1.00, "Full scale (the gap is tiny)")
    fig.suptitle(
        "Same two submissions, two evaluations — not a change over time",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_oof_vs_public_rank.png", dpi=140)
    plt.close(fig)


def main() -> None:
    df = load_train()
    counts = label_balance(df)
    print("Label counts")
    print(counts.to_string())
    print(f"positive rate {df[TARGET].mean():.4f}")
    single_feature_auc(df)
    rank_flip_chart()
    print(f"wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
