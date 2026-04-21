#!/usr/bin/env python3
"""
package_submission.py
─────────────────────
Creates a portable ZIP archive of the entire Subtitle Generator & Summarizer
project, strictly formatted for final submission.

INCLUDES:
  - /subtitles/
  - /summaries/
  - /transcripts/
  - /reports/
  - /notebooks/
  - /audio/
  - requirements.txt
  - *.py scripts

EXCLUDES:
  - /videos/ (too large)
  - /venv/
  - __pycache__/
  - .ipynb_checkpoints/
  - Any random hidden directories or massive untracked folders

Usage:
    python package_submission.py
"""

import os
import sys
import zipfile
from pathlib import Path

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ZIP   = PROJECT_ROOT / "subtitle_summarizer_submission.zip"

INCLUDED_FOLDERS = {
    "subtitles",
    "summaries",
    "transcripts",
    "reports",
    "notebooks",
    "audio"
}

EXCLUDED_FOLDERS = {
    "videos",
    "venv",
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache"
}


def create_submission_zip() -> None:
    print(f"\n{BOLD}Packaging Submission Archive{RESET}")
    print(f"Target: {OUTPUT_ZIP.name}\n")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    files_added = 0
    total_size_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through project root recursively
        for root, dirs, files in os.walk(PROJECT_ROOT):
            root_path = Path(root)
            
            # Prune excluded folders in-place so os.walk stops traversing them
            dirs[:] = [d for d in dirs if d not in EXCLUDED_FOLDERS]

            for file in files:
                file_path = root_path / file
                
                # Exclude the zip file itself
                if file_path == OUTPUT_ZIP:
                    continue

                # Determine whether file should be included:
                # 1. It is a root pipeline script (*.py) or requirements.txt
                # 2. It lives inside one of the explicitly INCLUDED_FOLDERS
                rel_path = file_path.relative_to(PROJECT_ROOT)
                top_level_dir = rel_path.parts[0]

                include_file = False
                if len(rel_path.parts) == 1:
                    # It's in the root folder
                    if file_path.suffix == ".py" or file_path.name == "requirements.txt":
                        include_file = True
                elif top_level_dir in INCLUDED_FOLDERS:
                    # It's inside an allowed folder
                    include_file = True

                if include_file:
                    try:
                        filesize = file_path.stat().st_size
                        zipf.write(file_path, arcname=rel_path)
                        total_size_bytes += filesize
                        files_added += 1
                        print(f"  {GREEN}+{RESET} {rel_path}")
                    except PermissionError:
                        print(f"  {YELLOW}⚠{RESET} Permission denied: {rel_path}")
                    except Exception as e:
                        print(f"  {YELLOW}⚠{RESET} Error adding {rel_path}: {e}")

    # Final Summary
    size_mb = total_size_bytes / (1024 * 1024)
    print(f"\n{'─' * 55}")
    print(f"  {BOLD}Packaging Complete{RESET}")
    print(f"  Total Files : {BLUE}{files_added}{RESET}")
    print(f"  Total Size  : {BLUE}{size_mb:.2f} MB{RESET}")
    print(f"  Saved to    : {GREEN}{OUTPUT_ZIP}{RESET}\n")


if __name__ == "__main__":
    create_submission_zip()
