# src/sentence_segment.py (Updated version with better logging)
from __future__ import annotations

import regex as re
from typing import List
import argparse
from pathlib import Path
import pandas as pd

# Common abbreviations (extend if needed)
ABBREVIATIONS = {
    "dr", "prof", "mr", "mrs", "ms", "t.k", "beyləqan", "azərbaycan", "ünvan", "cən", "m", "s", "ş", "b.k.",
    "q.k", "akademiya", "şirkət", "futbolçu", "nömrə",  # Add abbreviations as you spot them
}

# Regex patterns
SENT_END_RE = re.compile(r"[.!?]+")
DECIMAL_RE = re.compile(r"\d+[.,]\d+")
INITIALS_RE = re.compile(r"(?:\b\p{L}\.){2,}$", re.UNICODE)
CATEGORY_GARBAGE_RE = re.compile(
    r"""
    (?im)                             # case-insensitive, multiline
    ^\s*(kateqoriya|istinadlar|qeydlər|əlavə ədəbiyyat)\b.*$  # lines that start with Kateqoriya...
    """,
    re.VERBOSE,
)

def is_abbreviation(token: str) -> bool:
    token = token.lower().strip(".")
    return token in ABBREVIATIONS

def strip_wiki_garbage(text: str) -> str:
    # Remove category/navigation-like lines
    text = CATEGORY_GARBAGE_RE.sub(" ", text)
    # Also remove standalone occurrences like "Kateqoriya:" if embedded
    text = re.sub(r"(?i)\bkateqoriya\b\s*:\s*", " ", text)
    return text

def normalize_text(s: str) -> str:
    # Clean up excessive punctuation
    s = re.sub(r"\s*[;:]+\s*", " ", s)  # Replace excessive semicolons and colons
    s = re.sub(r"\(\s*;\s*\)", " ", s)  # Clean up ( ; )
    s = re.sub(r"\s+", " ", s).strip()  # Normalize multiple spaces
    return s

def sentence_segment(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    i = 0
    length = len(text)

    # Log text length for debugging
    print(f"Processing text of length {length}")

    # Prevent empty strings from causing index errors
    if not text.strip():  # If the text is just spaces or empty
        return sentences

    text = strip_wiki_garbage(text)
    text = normalize_text(text)

    # Add extra check to ensure text length is valid
    length = len(text)
    if length == 0:
        print("Warning: Text after cleaning is empty.")
        return sentences

    while i < length:
        ch = text[i]
        if ch in ".!?":
            chunk = text[start:i+1]

            # Look at the token before punctuation
            prev = chunk.rstrip().split()[-1]

            # Rules to avoid splitting
            if is_abbreviation(prev):
                i += 1
                continue
            if DECIMAL_RE.search(prev):
                i += 1
                continue
            if INITIALS_RE.search(prev):
                i += 1
                continue

            # Accept boundary
            sent = chunk.strip()
            if sent:
                sentences.append(sent)
            start = i + 1
        i += 1

    # Remainder
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)

    # Log the result to ensure it works
    print(f"Generated {len(sentences)} sentences.")
    return sentences


def _load_text_from_csv(corpus_path: str, limit: int | None = None) -> str:
    df = pd.read_csv(corpus_path)
    texts = df["text"].dropna().astype(str)
    if limit:
        texts = texts.head(limit)
    return " ".join(texts.tolist())


def main():
    ap = argparse.ArgumentParser(description="Sentence segmentation for Azerbaijani corpus.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text_path", type=str, help="Plain text file to segment.")
    src.add_argument("--corpus_path", type=str, help="CSV with a 'text' column (e.g., data/raw/corpus.csv).")
    ap.add_argument("--limit", type=int, default=None, help="Optional: limit number of rows from corpus.")
    ap.add_argument("--out", type=str, default="outputs/sentences.txt", help="Where to write segmented sentences.")
    args = ap.parse_args()

    if args.text_path:
        text = Path(args.text_path).read_text(encoding="utf-8")
    else:
        text = _load_text_from_csv(args.corpus_path, limit=args.limit)

    sentences = sentence_segment(text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sentences), encoding="utf-8")
    print(f"Saved {len(sentences)} sentences to {out_path}")


if __name__ == "__main__":
    main()
