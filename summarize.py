#!/usr/bin/env python3
"""
summarize.py
────────────
Uses BART (facebook/bart-large-cnn) to compile ≤100-word abstracts of
each video's parsed transcription. Integrates `utils.chunk_transcript` to
securely enforce max token length splits prior to model inference.

Naming convention:
    transcripts/lecture1.txt  →  summaries/lecture1_summary.txt

Usage:
    python summarize.py
"""

import sys
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from utils import chunk_transcript

PROJECT_ROOT = Path(__file__).resolve().parent
IN_DIR = PROJECT_ROOT / "transcripts"
OUT_DIR = PROJECT_ROOT / "summaries"

def batch_summarize(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        print(f"✘ Input directory missing: {input_dir}")
        sys.exit(1)

    txt_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix == ".txt"])
    if not txt_files:
        print(f"⚠ No .txt transcripts found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading facebook/bart-large-cnn on {device}...")
    
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Failed to load summarization model: {e}")
        sys.exit(1)

    for idx, txt_path in enumerate(txt_files, start=1):
        out_path = output_dir / f"{txt_path.stem}_summary.txt"
        print(f"\n[{idx}/{len(txt_files)}] Processing {txt_path.name}...")
        
        raw_text = txt_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            print("  ↳ Skipped (empty file)")
            continue

        chunks = chunk_transcript(raw_text, max_words=200)
        print(f"  ↳ Split into {len(chunks)} chunk(s). Summarizing...")
        
        intermediate_summaries = []
        for i, c in enumerate(chunks, 1):
            try:
                inputs = tokenizer(c, max_length=1024, return_tensors="pt", truncation=True).to(device)
                summary_ids = model.generate(inputs["input_ids"], max_length=60, min_length=15, num_beams=4, early_stopping=True)
                summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                intermediate_summaries.append(summary_text)
            except Exception as e:
                print(f"  ↳ Error on chunk {i}: {e}")
                
        if not intermediate_summaries:
            continue
            
        combined_text = " ".join(intermediate_summaries)
        final_word_count = len(combined_text.split())
        # Keep a copy of the full combined text for sentence expansion passes
        full_combined = combined_text

        # ── Pass 1: collapse if over 100 words ────────────────────────────
        if final_word_count > 100:
            print(f"  ↳ Pass 1 generated {final_word_count} words. Enforcing ≤100 constraint...")
            try:
                inputs = tokenizer(full_combined, max_length=1024, return_tensors="pt", truncation=True).to(device)
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=160, min_length=80,
                    num_beams=4, early_stopping=True
                )
                combined_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            except Exception as e:
                print(f"  ↳ Error during constraint pass: {e}")
                combined_text = " ".join(combined_text.split()[:100])

        # ── Pass 2: sentence count enforcement (3–5 sentences) ────────────
        n_sents = len(re.findall(r'[.!?]+', combined_text))
        print(f"  ↳ Sentence count after pass 1: {n_sents}")

        if n_sents < 3:
            # Feed the FULL combined text so BART has content to work with
            for min_len in [80, 100, 120]:
                try:
                    inputs = tokenizer(full_combined, max_length=1024, return_tensors="pt", truncation=True).to(device)
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=160, min_length=min_len,
                        num_beams=4, early_stopping=True
                    )
                    candidate = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    c_sents = len(re.findall(r'[.!?]+', candidate))
                    print(f"    ↳ min_len={min_len}: {c_sents} sentences")
                    if c_sents >= 3:
                        combined_text = candidate
                        n_sents = c_sents
                        break
                except Exception as e:
                    print(f"  ↳ Error on sentence-expansion attempt (min_len={min_len}): {e}")

        if n_sents > 5:
            # Truncate to the first 5 sentences
            combined_text = " ".join(re.split(r'(?<=[.!?])\s+', combined_text)[:5])

        # ── Final word-count hard cap ≤ 100 ──────────────────────────────
        if len(combined_text.split()) > 100:
            combined_text = " ".join(combined_text.split()[:100])

        out_path.write_text(combined_text.strip(), encoding="utf-8")
        final_sents = len(re.findall(r'[.!?]+', combined_text))
        print(f"  ✔ Saved — {len(combined_text.split())} words, {final_sents} sentence(s)")

if __name__ == "__main__":
    batch_summarize(IN_DIR, OUT_DIR)
