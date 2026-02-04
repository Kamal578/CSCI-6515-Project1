# src/eval_sentence_seg.py
from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import pandas as pd

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


def _load_text_from_corpus(path: str, limit: int | None) -> str:
    df = pd.read_csv(path)
    texts = df["text"].dropna().astype(str)
    if limit:
        texts = texts.head(limit)
    return " ".join(texts.tolist())


def main():
    ap = argparse.ArgumentParser(description="Evaluate sentence segmentation against a gold file.")
    ap.add_argument("--gold", type=str, default="data/processed/sent_gold.txt", help="Gold sentence-per-line file.")
    ap.add_argument("--pred", type=str, help="Optional predicted sentence-per-line file. If omitted, segment a corpus.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv", help="Corpus CSV with 'text' column.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of rows from corpus when auto-segmenting.")
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}. Run generate_gold_standard.py and hand-edit it.")

    gold = load_gold(args.gold)

    if args.pred:
        predicted = load_gold(args.pred)
    else:
        text = _load_text_from_corpus(args.corpus_path, limit=args.limit)
        predicted = sentence_segment(text)

    p, r, f1 = evaluate(predicted, gold)
    print(f"[Task 4] Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
