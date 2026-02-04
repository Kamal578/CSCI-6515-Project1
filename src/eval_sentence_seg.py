# src/eval_sentence_seg.py
from __future__ import annotations

from pathlib import Path
from typing import List

from .sentence_segment import sentence_segment


def load_gold(path: str) -> List[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(predicted: List[str], gold: List[str]):
    pred_set = set(predicted)
    gold_set = set(gold)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    return precision, recall, f1


def main():
    text = Path("data/raw/corpus.csv").read_text(encoding="utf-8", errors="ignore")
    predicted = sentence_segment(text)
    gold = load_gold("data/processed/sent_gold.txt")

    p, r, f1 = evaluate(predicted, gold)
    print(f"[Task 4] Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
