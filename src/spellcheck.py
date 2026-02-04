from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .tokenize import iter_tokens


def load_vocab(corpus_path: str, lowercase: bool = True, min_freq: int = 1) -> Counter:
    df = pd.read_csv(corpus_path)
    freqs = Counter(iter_tokens(df["text"].fillna("").astype(str).tolist(), lowercase=lowercase))
    if min_freq > 1:
        freqs = Counter({w: c for w, c in freqs.items() if c >= min_freq})
    return freqs


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            ins = current[j - 1] + 1
            del_ = previous[j] + 1
            sub = previous[j - 1] + (ca != cb)
            current.append(min(ins, del_, sub))
        previous = current
    return previous[-1]


def suggest(word: str, vocab: Dict[str, int], max_dist: int = 2, top_k: int = 5) -> List[Tuple[str, int]]:
    candidates: List[Tuple[str, int, int]] = []  # (dist, -freq, token)
    for tok, freq in vocab.items():
        dist = levenshtein(word, tok)
        if dist <= max_dist:
            candidates.append((dist, -freq, tok))
    candidates.sort()
    return [(tok, -freq) for dist, freq, tok in candidates[:top_k]]


def load_words_from_file(path: str) -> List[str]:
    return [w.strip() for w in Path(path).read_text(encoding="utf-8").splitlines() if w.strip()]


def main():
    ap = argparse.ArgumentParser(description="Simple spell checker using Levenshtein distance over corpus vocabulary.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv", help="CSV with 'text' column.")
    ap.add_argument("--lowercase", action="store_true", default=True, help="Lowercase tokens for vocab/building.")
    ap.add_argument("--min_freq", type=int, default=1, help="Drop vocab items under this count.")
    ap.add_argument("--max_dist", type=int, default=2, help="Maximum edit distance for candidates.")
    ap.add_argument("--top_k", type=int, default=5, help="Return up to this many suggestions.")

    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--word", type=str, help="Single word to correct.")
    target.add_argument("--wordlist", type=str, help="File with one word per line to correct.")
    target.add_argument("--text_path", type=str, help="Plain text file; all OOV tokens will be checked.")
    ap.add_argument("--out", type=str, default="outputs/spellcheck/suggestions.txt", help="Where to write suggestions.")
    args = ap.parse_args()

    vocab = load_vocab(args.corpus_path, lowercase=args.lowercase, min_freq=args.min_freq)
    known = set(vocab.keys())

    if args.word:
        words = [args.word]
    elif args.wordlist:
        words = load_words_from_file(args.wordlist)
    else:
        text = Path(args.text_path).read_text(encoding="utf-8")
        words = [w for w in iter_tokens([text], lowercase=args.lowercase)]

    suggestions = []
    for w in words:
        if w in known:
            continue
        cands = suggest(w, vocab, max_dist=args.max_dist, top_k=args.top_k)
        if cands:
            suggestion_str = ", ".join([f"{tok} (freq={freq})" for tok, freq in cands])
        else:
            suggestion_str = "(no candidates)"
        suggestions.append(f"{w} -> {suggestion_str}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(suggestions), encoding="utf-8")
    print(f"Wrote {len(suggestions)} suggestions to {out_path}")


if __name__ == "__main__":
    main()
