# src/sentence_segment.py (Updated version with better logging)
from __future__ import annotations

import regex as re
from typing import List

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
