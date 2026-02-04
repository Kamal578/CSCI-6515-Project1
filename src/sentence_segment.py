# src/sentence_segment.py (Updated version with better logging)
from __future__ import annotations

import regex as re
from typing import List
import argparse
from pathlib import Path
import pandas as pd

# ----------------------------
# Quote characters (robust)
# ----------------------------
QUOTE_OPEN = set(['"', "'", "“", "«", "‹", "„", "‘", "‚", "「", "『", "（", "(", "[", "{"])
QUOTE_CLOSE = set(['"', "'", "”", "»", "›", "‟", "’", "‛", "」", "』", "）", ")", "]", "}"])

# ----------------------------
# Abbreviations (normalize)
# Merge domain-specific abbreviations from both implementations
# ----------------------------
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "etc", "e.g", "i.e",
    "a.m", "s.a", "b.c", "m.a", "ph.d", "u.s",
    "t.k", "beyləqan", "azərbaycan", "ünvan", "cən", "m", "s", "ş", "b.k.",
    "q.k", "akademiya", "şirkət", "futbolçu", "nömrə"
}

SENT_END = {".", "!", "?"}
DECIMAL_RE = re.compile(r"\d+[.,]\d+")
INITIALS_RE = re.compile(r"(?:\b\p{L}\.){2,}$", re.UNICODE)
CATEGORY_GARBAGE_RE = re.compile(
    r"""
    (?im)
    ^\s*(kateqoriya|istinadlar|qeydlər|əlavə ədəbiyyat)\b.*$
    """,
    re.VERBOSE,
)

def is_abbreviation(token: str) -> bool:
    tok = token.strip()
    tok = tok.strip("".join(QUOTE_OPEN | QUOTE_CLOSE))
    tok = tok.rstrip(".,!?;:")
    return tok.lower() in ABBREVIATIONS

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

    text = normalize_text(strip_wiki_garbage(text))
    n = len(text)
    if n == 0:
        return sentences

    while i < n:
        ch = text[i]

        # Closing quote + space + Uppercase => boundary
        if ch in QUOTE_CLOSE and quote_followed_by_space_upper(text, i):
            sent = text[start:i + 1].strip()
            if sent:
                sentences.append(sent)
            start = i + 1
            i += 1
            continue

        if ch in SENT_END:
            if ch == ".":
                if is_decimal_dot_or_comma(text, i):
                    i += 1
                    continue
                if is_surrounded_by_non_space(text, i):
                    i += 1
                    continue
                if is_initial_period(text, i):
                    i += 1
                    continue

            chunk = text[start:i + 1]
            prev_token = chunk.rstrip().split()[-1] if chunk.rstrip().split() else ""

            if is_abbreviation(prev_token):
                i += 1
                continue

            if ch == "." and is_compact_initials(prev_token):
                i += 1
                continue

            if i + 1 < n and text[i + 1] in QUOTE_CLOSE:
                i += 1
                continue

            sent = chunk.strip()
            if sent:
                sentences.append(sent)
            start = i + 1

        elif ch == ":":
            pass

        i += 1

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)

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
