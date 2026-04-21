#!/usr/bin/env python3
"""
extract_audio.py
────────────────
Batch-extracts audio from every .mp4 file in the videos/ folder and saves
each as a mono, 16 kHz .wav file in the audio/ folder — the optimal input
format for OpenAI Whisper.

Naming convention:
    videos/lecture1.mp4  →  audio/lecture1.wav

Usage:
    python extract_audio.py                        # defaults: videos/ → audio/
    python extract_audio.py --input_dir my_vids    # custom input folder
    python extract_audio.py --output_dir my_wavs   # custom output folder
"""

import argparse
import os
import sys
import time
from pathlib import Path

import ffmpeg  # ffmpeg-python

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "videos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "audio"

# ── audio parameters (Whisper-optimal) ────────────────────────────
SAMPLE_RATE = 16_000   # 16 kHz
CHANNELS    = 1        # mono


def extract_audio(video_path: Path, output_path: Path) -> None:
    """
    Extract audio from a single video file and save as a mono 16 kHz .wav.

    Parameters
    ----------
    video_path : Path
        Absolute path to the source .mp4 file.
    output_path : Path
        Absolute path for the output .wav file.

    Raises
    ------
    ffmpeg.Error
        If ffmpeg encounters an issue with the input file.
    FileNotFoundError
        If the source video does not exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Probe the file first to catch corrupt / unsupported files early
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Cannot read '{video_path.name}' — file may be corrupt or not a valid video.\n"
            f"ffmpeg stderr: {e.stderr.decode().strip() if e.stderr else 'N/A'}"
        ) from e

    # Verify at least one audio stream exists
    audio_streams = [s for s in probe.get("streams", []) if s["codec_type"] == "audio"]
    if not audio_streams:
        raise RuntimeError(f"No audio stream found in '{video_path.name}'.")

    # Run the extraction
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(output_path),
                ac=CHANNELS,       # mono
                ar=SAMPLE_RATE,    # 16 kHz
                format="wav",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"Audio extraction failed for '{video_path.name}'.\n"
            f"ffmpeg stderr: {e.stderr.decode().strip() if e.stderr else 'N/A'}"
        ) from e


def batch_extract(input_dir: Path, output_dir: Path) -> None:
    """
    Iterate over all .mp4 files in *input_dir*, extract audio, and save
    corresponding .wav files in *output_dir*.

    Parameters
    ----------
    input_dir : Path
        Directory containing source .mp4 videos.
    output_dir : Path
        Directory where extracted .wav files are saved.
    """
    print(f"\n{BOLD}Audio Extraction — Batch Processing{RESET}")
    print(f"  Input :  {input_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Format:  mono, {SAMPLE_RATE // 1000} kHz WAV\n")

    # Validate input directory
    if not input_dir.exists():
        print(f"  {RED}✘{RESET} Input directory does not exist: {input_dir}")
        print(f"    Create it and place your .mp4 files inside, then re-run.")
        sys.exit(1)

    # Collect .mp4 files (case-insensitive)
    videos = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".mp4"
    ])

    if not videos:
        print(f"  {YELLOW}⚠{RESET} No .mp4 files found in {input_dir}")
        print(f"    Place your lecture videos there and re-run.")
        return

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Found {len(videos)} video(s). Starting extraction…\n")

    success = 0
    failed  = 0

    for idx, video in enumerate(videos, start=1):
        wav_name    = video.stem + ".wav"
        output_path = output_dir / wav_name

        print(f"  [{idx}/{len(videos)}] Extracting {video.name} … ", end="", flush=True)
        start_time = time.time()

        try:
            extract_audio(video, output_path)
            elapsed  = time.time() - start_time
            size_mb  = output_path.stat().st_size / (1024 * 1024)
            print(f"{GREEN}done{RESET}  ({elapsed:.1f}s, {size_mb:.1f} MB)")
            success += 1

        except (RuntimeError, FileNotFoundError) as e:
            elapsed = time.time() - start_time
            print(f"{RED}FAILED{RESET}  ({elapsed:.1f}s)")
            print(f"        ↳ {e}\n")
            failed += 1

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"{RED}FAILED{RESET}  ({elapsed:.1f}s)")
            print(f"        ↳ Unexpected error: {e}\n")
            failed += 1

    # ── summary ────────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    print(f"  {GREEN}✔{RESET} {success} file(s) extracted successfully.")
    if failed:
        print(f"  {RED}✘{RESET} {failed} file(s) failed — see errors above.")
    print()


# ── CLI entry point ───────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract mono 16 kHz WAV audio from .mp4 lecture videos."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing .mp4 files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output .wav files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_extract(args.input_dir, args.output_dir)
