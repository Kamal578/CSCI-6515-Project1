# src/generate_gold_standard.py
import random
from pathlib import Path
import pandas as pd
from src.sentence_segment import sentence_segment

def generate_gold_standard(corpus_path: str = "data/raw/corpus.csv", output_path: str = "data/processed/sent_gold.txt", num_samples: int = 100):
    # Load the dataset
    df = pd.read_csv(corpus_path)
    
    # Ensure text column exists
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {corpus_path}")
    
    # Sample num_samples rows randomly
    sampled_texts = random.sample(df["text"].tolist(), num_samples)
    
    # Open output file
    with open(output_path, "w", encoding="utf-8") as f:
        # Iterate over sampled texts, segment sentences, and write to file
        for text in sampled_texts:
            sentences = sentence_segment(text)
            for sentence in sentences:
                f.write(sentence + "\n")

    print(f"Gold standard sentences written to {output_path}.")

if __name__ == "__main__":
    generate_gold_standard()
