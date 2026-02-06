from __future__ import annotations

import regex as re
from collections import Counter
from typing import Iterable
import pandas as pd

from .tokenize import iter_tokens

# Azerbaijani + basic Latin alphabet for sampling synthetic typos
AZ_ALPHABET = list("abcçdeəfgğhxıijkqlmnoöpprsşt tuüvyz".replace(" ", "")) + list("abcdefghijklmnopqrstuvwxyz")

# Azerbaijani casual-typing variants (English keyboard).
AZ_VARIANTS = {
    # digraphs
    "ch": ["ç"],
    "sh": ["ş"],
    "gh": ["ğ"],
    # single letters
    "c": ["ç"],
    "s": ["ş"],
    "e": ["ə"],
    "o": ["ö"],
    "u": ["ü"],
    "g": ["ğ"],
    "i": ["ı"],
    "w": ["v"],
}


def load_freqs(corpus_path: str, lowercase: bool = True) -> Counter:
    df = pd.read_csv(corpus_path)
    freqs = Counter(iter_tokens(df["text"].fillna("").astype(str).tolist(), lowercase=lowercase))
    return freqs


WORD_CHARS_RE = re.compile(r"^[\p{L}]+$", re.UNICODE)


def filter_vocab(
    freqs: Counter,
    min_freq: int = 2,
    min_len: int = 3,
    max_upper_ratio: float = 0.6,
) -> Counter:
    """
    Drop very rare, too short, non-letter, or acronym-like tokens.
    """
    clean = Counter()
    for w, c in freqs.items():
        if len(w) < min_len:
            continue
        if c < min_freq:
            continue
        if not WORD_CHARS_RE.match(w):
            continue
        upper_ratio = sum(1 for ch in w if ch.isupper()) / len(w)
        if upper_ratio > max_upper_ratio:
            continue
        clean[w] = c
    return clean


def _tokenize_az_variants(word: str) -> list[str]:
    """
    Tokenize left-to-right, matching digraphs first.
    """
    units: list[str] = []
    i = 0
    while i < len(word):
        two = word[i:i + 2]
        if two in ("ch", "sh", "gh"):
            units.append(two)
            i += 2
        else:
            units.append(word[i])
            i += 1
    return units


def generate_az_variants_with_edits(
    word: str,
    max_edits: int = 2,
    max_candidates: int = 40,
) -> list[tuple[str, int]]:
    """
    Generate plausible Azerbaijani variants using a bounded beam search.
    Returns (variant, edits) pairs, sorted by edits then token.
    """
    if not word:
        return [("", 0)]

    units = _tokenize_az_variants(word)
    beam: list[tuple[str, int]] = [("", 0)]

    for unit in units:
        options = [unit]
        for repl in AZ_VARIANTS.get(unit, []):
            if repl != unit:
                options.append(repl)

        expanded: list[tuple[str, int]] = []
        for prefix, edits in beam:
            for opt in options:
                new_edits = edits + (0 if opt == unit else 1)
                if new_edits > max_edits:
                    continue
                expanded.append((prefix + opt, new_edits))

        expanded.sort(key=lambda x: (x[1], x[0]))
        if len(expanded) > max_candidates:
            expanded = expanded[:max_candidates]
        beam = expanded
        if not beam:
            break

    best: dict[str, int] = {}
    for cand, edits in beam:
        if cand not in best or edits < best[cand]:
            best[cand] = edits

    return sorted(best.items(), key=lambda x: (x[1], x[0]))


def generate_az_variants(
    word: str,
    max_edits: int = 2,
    max_candidates: int = 40,
) -> list[str]:
    """
    Convenience wrapper returning only variant strings.
    """
    return [w for w, _ in generate_az_variants_with_edits(
        word,
        max_edits=max_edits,
        max_candidates=max_candidates,
    )]
