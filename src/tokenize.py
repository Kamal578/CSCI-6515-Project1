# src/tokenize.py
from __future__ import annotations

import regex as re
from typing import Iterable, List

# Word pattern:
# - letters (including Azerbaijani) with optional internal hyphen/apostrophe parts
# - numbers (including decimals 3.14, 2,5)
WORD_RE = re.compile(
    r"""
    (?:\p{L}+(?:[’'\-]\p{L}+)*)
    |
    (?:\p{N}+(?:[.,]\p{N}+)*)
    """,
    re.VERBOSE | re.UNICODE,
)

# src/tokenize.py (modify the existing strip_wiki_garbage function)
CATEGORY_GARBAGE_RE = re.compile(
    r"""
    (?im)                             # case-insensitive, multiline
    ^\s*(kateqoriya|istinadlar|qeydlər|əlavə ədəbiyyat)\b.*$  # lines that start with Kateqoriya...
    """,
    re.VERBOSE,
)

def strip_wiki_garbage(text: str) -> str:
    # Remove category/navigation-like lines
    text = CATEGORY_GARBAGE_RE.sub(" ", text)
    # Also remove standalone occurrences like "Kateqoriya:" if embedded
    text = re.sub(r"(?i)\bkateqoriya\b\s*:\s*", " ", text)
    text = re.sub(r"(?i)\bistinadlar\b\s*:\s*", " ", text)
    text = re.sub(r"(?i)\bqeydlər\b\s*:\s*", " ", text)
    text = re.sub(r"(?i)\bəlavə ədəbiyyat\b\s*:\s*", " ", text)
    return text


def normalize_text(s: str) -> str:
    # Normalize some common Wikipedia-ish punctuation variants
    s = s.replace("\u00A0", " ")      # nbsp
    s = s.replace("\u2019", "'")      # right single quote
    s = s.replace("\u2018", "'")      # left single quote
    s = s.replace("\u2013", "-")      # en dash
    s = s.replace("\u2014", "-")      # em dash
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(text: str, lowercase: bool = True) -> List[str]:
    text = normalize_text(text)
    text = strip_wiki_garbage(text)  
    if lowercase:
        text = text.lower()
    return [m.group(0) for m in WORD_RE.finditer(text)]

def iter_tokens(texts: Iterable[str], lowercase: bool = True) -> Iterable[str]:
    for t in texts:
        for tok in tokenize(t, lowercase=lowercase):
            yield tok
