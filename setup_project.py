#!/usr/bin/env python3
"""
setup_project.py
────────────────
Creates the complete folder structure required by the Subtitle Generator
& Summarizer pipeline.  Safe to run multiple times — existing directories
are silently preserved.

Folder layout created
─────────────────────
project_root/
├── videos/          ← input .mp4 lecture files
├── audio/           ← extracted mono 16 kHz .wav files
├── transcripts/     ← raw .txt transcripts + .json segment files
├── subtitles/       ← generated .srt subtitle files
├── summaries/       ← ≤100-word .txt summary files
├── notebooks/       ← Jupyter notebooks (end-to-end pipeline)
└── reports/         ← WER & ROUGE evaluation results

Usage:
    python setup_project.py
"""

import os
import sys
from pathlib import Path

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ── project directories ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DIRECTORIES = {
    "videos":      "Input .mp4 lecture files",
    "audio":       "Extracted mono 16 kHz .wav files",
    "transcripts": "Raw .txt transcripts + .json segment data",
    "subtitles":   "Generated .srt subtitle files",
    "summaries":   "≤100-word .txt summary files",
    "notebooks":   "Jupyter notebooks (end-to-end pipeline)",
    "reports":     "WER & ROUGE evaluation results",
}


def setup_directories(root: Path = PROJECT_ROOT) -> None:
    """
    Create every required project sub-directory under *root*.

    Parameters
    ----------
    root : Path
        The project root directory.  Defaults to the directory
        containing this script.

    Returns
    -------
    None
    """
    print(f"\n{BOLD}Subtitle Generator & Summarizer — Project Setup{RESET}")
    print(f"Project root: {root}\n")

    created = 0
    existed = 0

    for folder, description in DIRECTORIES.items():
        dir_path = root / folder
        if dir_path.exists():
            print(f"  {YELLOW}○{RESET} {folder + '/':18s} already exists  ({description})")
            existed += 1
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  {GREEN}✔{RESET} {folder + '/':18s} created         ({description})")
            created += 1

    # ── summary ────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    if created:
        print(f"  {GREEN}✔{RESET} {created} director{'y' if created == 1 else 'ies'} created, "
              f"{existed} already existed.")
    else:
        print(f"  {GREEN}✔{RESET} All {existed} directories already in place — nothing to do.")

    print(f"\n  {BOLD}Project structure is ready!{RESET}")
    print(f"  Next step: place your .mp4 lecture files in {root / 'videos'}/\n")


if __name__ == "__main__":
    setup_directories()
