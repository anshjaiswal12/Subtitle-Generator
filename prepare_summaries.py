#!/usr/bin/env python3
"""
prepare_summaries.py
────────────────────
Reads raw .txt transcript files from the transcripts/ folder, chunks the text
intelligently into approximately 200-word blocks using utils.chunk_transcript(),
and returns (or prints) the chunks. This is a preparation step for the
summarization model.

Naming convention:
    transcripts/lecture1.txt  →  Chunked into memory

Usage:
    python prepare_summaries.py
"""

import argparse
import sys
from pathlib import Path

from utils import chunk_transcript

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
WHITE  = "\033[97m"
RESET  = "\033[0m"

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "transcripts"


def process_transcripts(input_dir: Path) -> dict[str, list[str]]:
    """
    Read all .txt files from *input_dir*, chunk each using chunk_transcript(),
    and return a dictionary mapping filenames to lists of text chunks.

    Parameters
    ----------
    input_dir : Path
        Folder containing raw ``.txt`` transcripts.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping the filename (e.g. 'lecture1.txt') to a list
        of string chunks.
    """
    print(f"\n{BOLD}Transcript Chunking Preparation{RESET}")
    print(f"  Input :  {input_dir}\n")

    if not input_dir.exists():
        print(f"  {RED}✘{RESET} Input directory does not exist: {input_dir}")
        sys.exit(1)

    txt_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    ])

    if not txt_files:
        print(f"  {YELLOW}⚠{RESET} No .txt files found in {input_dir}")
        print(f"    Expected raw text transcripts from Whisper.")
        return {}

    print(f"  Found {len(txt_files)} transcript(s). Chunking…\n")

    results: dict[str, list[str]] = {}

    for idx, txt_path in enumerate(txt_files, start=1):
        print(f"  [{idx}/{len(txt_files)}] {txt_path.name} ", end="")
        
        try:
            content = txt_path.read_text(encoding="utf-8").strip()
            if not content:
                print(f"→ {YELLOW}SKIPPED (empty file){RESET}")
                continue

            chunks = chunk_transcript(content, max_words=200)
            results[txt_path.name] = chunks
            
            print(f"→ {GREEN}{len(chunks)} chunks{RESET} created.")
            
            # Print a quick preview of the chunks
            for i, chunk in enumerate(chunks[:2], start=1):
                word_count = len(chunk.split())
                preview = (chunk[:75] + "...") if len(chunk) > 75 else chunk
                print(f"      {WHITE}↳ Chunk {i} ({word_count} words):{RESET} {preview}")
            if len(chunks) > 2:
                print(f"      {WHITE}↳ ...and {len(chunks)-2} more chunk(s){RESET}")

        except Exception as e:
            print(f"→ {RED}FAILED{RESET} ({e})")

    print(f"\n{'─' * 55}")
    print(f"  {GREEN}✔{RESET} Successfully chunked {len(results)} file(s). Ready for summarization.\n")
    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk raw transcripts into blocks of ~200 words."
    )
    parser.add_argument(
        "--input_dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory containing .txt files (default: {DEFAULT_INPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_transcripts(args.input_dir)
