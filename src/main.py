from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .pull_wikipedia import random_titles, category_titles, fetch_pages_wikitext, clean_wikitext_to_text, save_corpus_csv
from .task1_stats import task1_stats
from .heaps import run_heaps
from .task3_bpe import task3_bpe
from .sentence_segment import sentence_segment
from .spellcheck import load_vocab, suggest


def collect_corpus(args):
    if args.random is None and args.category is None:
        args.random = 500  # sensible default

    if args.random is not None:
        titles = random_titles(args.random, session=args.session)
    else:
        titles = category_titles(args.category, n=args.limit, session=args.session)

    raw_pages = fetch_pages_wikitext(titles, session=args.session)
    pages = []
    for page_id, title, rev_id, ts, wikitext in raw_pages:
        clean = clean_wikitext_to_text(wikitext)
        if len(clean) < args.min_chars:
            continue
        url_title = title.replace(" ", "_")
        url = f"https://az.wikipedia.org/wiki/{url_title}"
        pages.append((page_id, title, rev_id, ts, url, clean))

    # Convert to Page dataclass for save helper
    from dataclasses import dataclass
    @dataclass
    class Page:
        page_id: int
        title: str
        revision_id: int
        timestamp: str
        url: str
        text: str
    save_corpus_csv([Page(*p) for p in pages], args.corpus_path)
    print(f"Collected {len(pages)} docs -> {args.corpus_path}")


def run_sentence_segmentation(corpus_path: str, out_path: str, limit: int | None):
    df = pd.read_csv(corpus_path)
    texts = df["text"].dropna().astype(str)
    if limit:
        texts = texts.head(limit)
    text = " ".join(texts.tolist())
    sentences = sentence_segment(text)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(sentences), encoding="utf-8")
    print(f"Sentence segmentation: wrote {len(sentences)} sentences to {out_file}")


def run_spellcheck_stage(corpus_path: str, wordlist: str | None, out_path: str, max_dist: int, top_k: int, include_known: bool):
    vocab = load_vocab(corpus_path, lowercase=True, min_freq=1)
    known = set(vocab.keys())

    if wordlist:
        words = [w.strip() for w in Path(wordlist).read_text(encoding="utf-8").splitlines() if w.strip()]
    else:
        # default: find low-frequency tokens to spot likely errors
        words = [w for w, c in vocab.items() if c == 1][:200]

    suggestions = []
    for w in words:
        if (not include_known) and w in known:
            continue
        cands = suggest(w, vocab, max_dist=max_dist, top_k=top_k)
        if not cands:
            continue
        suggestion_str = ", ".join([f"{tok} (freq={freq})" for tok, freq in cands])
        suggestions.append(f"{w} -> {suggestion_str}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(suggestions), encoding="utf-8")
    print(f"Spellcheck suggestions written to {out_file} ({len(suggestions)} entries)")


def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline driver for Azerbaijani Wikipedia NLP tasks.")
    ap.add_argument("--corpus_path", type=str, default="data/raw/corpus.csv", help="Corpus CSV path.")

    stages = ap.add_argument_group("Stages")
    stages.add_argument("--collect", action="store_true", help="Download + clean corpus from Wikipedia.")
    stages.add_argument("--stats", action="store_true", help="Run token frequency stats / Zipf.")
    stages.add_argument("--heaps", action="store_true", help="Fit Heaps' law.")
    stages.add_argument("--bpe", action="store_true", help="Train BPE on corpus.")
    stages.add_argument("--sentseg", action="store_true", help="Run sentence segmentation.")
    stages.add_argument("--spell", action="store_true", help="Run spellcheck suggestions.")

    collect = ap.add_argument_group("Collect options")
    collect.add_argument("--random", type=int, help="Number of random articles.")
    collect.add_argument("--category", type=str, help="Category title (non-recursive).")
    collect.add_argument("--limit", type=int, default=500, help="Max pages when using --category.")
    collect.add_argument("--min_chars", type=int, default=400, help="Drop docs shorter than this after cleaning.")

    sentseg = ap.add_argument_group("Sentence segmentation")
    sentseg.add_argument("--sentseg_out", type=str, default="outputs/sentences.txt", help="Output file for sentences.")
    sentseg.add_argument("--sentseg_limit", type=int, default=None, help="Limit rows for segmentation.")

    spell = ap.add_argument_group("Spellcheck")
    spell.add_argument("--wordlist", type=str, help="Optional wordlist (one per line) to check.")
    spell.add_argument("--spell_out", type=str, default="outputs/spellcheck/suggestions.txt")
    spell.add_argument("--max_dist", type=int, default=2)
    spell.add_argument("--top_k", type=int, default=5)

    args = ap.parse_args()
    # If no stage flags set, default to stats+heaps+bpe
    if not any([args.collect, args.stats, args.heaps, args.bpe, args.sentseg, args.spell]):
        args.stats = args.heaps = args.bpe = True

    # Set up requests session lazily for collect
    import requests
    args.session = requests.Session()
    args.session.headers.update({"User-Agent": "NLP-AZWIKI-Corpus-Collector/1.0 (educational project)"})
    return args


def main():
    args = parse_args()

    if args.collect:
        collect_corpus(args)

    if args.stats:
        task1_stats(corpus_path=args.corpus_path)

    if args.heaps:
        run_heaps(corpus_path=args.corpus_path)

    if args.bpe:
        task3_bpe(corpus_path=args.corpus_path)

    if args.sentseg:
        run_sentence_segmentation(args.corpus_path, args.sentseg_out, args.sentseg_limit)

    if args.spell:
        run_spellcheck_stage(args.corpus_path, args.wordlist, args.spell_out, args.max_dist, args.top_k, include_known=not bool(args.wordlist))


if __name__ == "__main__":
    main()
