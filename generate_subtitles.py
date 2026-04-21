#!/usr/bin/env python3
"""
generate_subtitles.py
─────────────────────
Batch-reads Whisper JSON transcript files from the transcripts/ folder,
converts each to a valid .srt subtitle file using utils.convert_to_srt(),
and saves results in the subtitles/ folder.

Expected JSON format (produced by Whisper / transcribe.py):
    {
      "segments": [
        {"start": 0.0,  "end": 3.52,  "text": "Hello everyone."},
        {"start": 3.52, "end": 7.81,  "text": "Welcome to today's lecture."},
        ...
      ]
    }

Naming convention:
    transcripts/lecture1_segments.json  →  subtitles/lecture1.srt

Usage:
    python generate_subtitles.py
    python generate_subtitles.py --input_dir transcripts --output_dir subtitles
"""

import argparse
import json
import sys
import time
from pathlib import Path

from utils import convert_to_srt

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "transcripts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "subtitles"


def load_segments(json_path: Path) -> list[dict]:
    """
    Load Whisper segments from a JSON transcript file.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file containing a ``"segments"`` key.

    Returns
    -------
    list[dict]
        List of segment dicts, each with 'start', 'end', 'text'.

    Raises
    ------
    ValueError
        If the JSON structure is unexpected or the segment list is empty.
    FileNotFoundError
        If the file does not exist.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept both {"segments": [...]} and a bare [...]
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    else:
        raise ValueError(
            f"Unexpected JSON structure in '{json_path.name}'. "
            f"Expected a 'segments' key or a top-level list."
        )

    if not segments:
        raise ValueError(f"No segments found in '{json_path.name}'.")

    return segments


def derive_srt_name(json_filename: str) -> str:
    """
    Derive the .srt filename from the JSON transcript filename.

    Strips common suffixes like ``_segments`` before adding ``.srt``.

    Parameters
    ----------
    json_filename : str
        Original JSON filename (e.g., ``lecture1_segments.json``).

    Returns
    -------
    str
        SRT filename (e.g., ``lecture1.srt``).
    """
    stem = Path(json_filename).stem          # lecture1_segments
    for suffix in ("_segments", "_transcript", "_segs"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}.srt"


def batch_generate(input_dir: Path, output_dir: Path) -> None:
    """
    Convert every JSON transcript in *input_dir* to an SRT file in *output_dir*.

    Parameters
    ----------
    input_dir : Path
        Folder containing ``*_segments.json`` files.
    output_dir : Path
        Folder where ``.srt`` files will be written.
    """
    print(f"\n{BOLD}SRT Subtitle Generation — Batch Processing{RESET}")
    print(f"  Input :  {input_dir}")
    print(f"  Output:  {output_dir}\n")

    # ── validate input directory ──────────────────────────────────
    if not input_dir.exists():
        print(f"  {RED}✘{RESET} Input directory does not exist: {input_dir}")
        print(f"    Run transcribe.py first to generate JSON transcripts.")
        sys.exit(1)

    # ── collect JSON files ────────────────────────────────────────
    json_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".json"
    ])

    if not json_files:
        print(f"  {YELLOW}⚠{RESET} No .json transcript files found in {input_dir}")
        print(f"    Run transcribe.py first to generate transcripts.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Found {len(json_files)} transcript(s). Generating SRT files…\n")

    success = 0
    failed  = 0

    for idx, json_path in enumerate(json_files, start=1):
        srt_name    = derive_srt_name(json_path.name)
        output_path = output_dir / srt_name

        print(f"  [{idx}/{len(json_files)}] {json_path.name} → {srt_name} … ",
              end="", flush=True)
        start_time = time.time()

        try:
            segments = load_segments(json_path)
            convert_to_srt(segments, output_path)
            elapsed      = time.time() - start_time
            seg_count    = len(segments)
            print(f"{GREEN}done{RESET}  ({seg_count} segments, {elapsed:.2f}s)")
            success += 1

        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            elapsed = time.time() - start_time
            print(f"{RED}FAILED{RESET}  ({elapsed:.2f}s)")
            print(f"        ↳ {e}\n")
            failed += 1

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{RED}FAILED{RESET}  ({elapsed:.2f}s)")
            print(f"        ↳ Unexpected error: {e}\n")
            failed += 1

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  {GREEN}✔{RESET} {success} SRT file(s) generated successfully.")
    if failed:
        print(f"  {RED}✘{RESET} {failed} file(s) failed — see errors above.")
    print()


# ── CLI entry point ───────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Whisper JSON transcripts to .srt subtitle files."
    )
    parser.add_argument(
        "--input_dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Directory containing JSON transcripts (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output .srt files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_generate(args.input_dir, args.output_dir)
