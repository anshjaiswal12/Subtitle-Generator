#!/usr/bin/env python3
"""
transcribe.py
─────────────
Loops over all .wav files in /audio, processes them using OpenAI Whisper ("base"),
and outputs a plain .txt transcript and a .json file containing segment details.

Naming convention:
    audio/lecture1.wav  →  transcripts/lecture1.txt
                        →  transcripts/lecture1_segments.json

Usage:
    python transcribe.py
"""

import os
import sys
import json
import time
from pathlib import Path
import warnings

import torch
import whisper

# Suppress expected FP16 warning on CPUs
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "audio"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "transcripts"

def batch_transcribe(input_dir: Path, output_dir: Path):
    if not input_dir.exists():
        print(f"✘ Input directory missing: {input_dir}")
        sys.exit(1)

    wav_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"])
    if not wav_files:
        print(f"⚠ No .wav files found in {input_dir}. Run extract_audio.py first.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper 'base' on {device}...")
    model = whisper.load_model("base", device=device)

    for idx, wav_path in enumerate(wav_files, start=1):
        txt_path = output_dir / f"{wav_path.stem}.txt"
        json_path = output_dir / f"{wav_path.stem}_segments.json"
        
        print(f"[{idx}/{len(wav_files)}] Transcribing {wav_path.name}... ", end="", flush=True)
        start = time.time()
        
        try:
            result = model.transcribe(str(wav_path))
            
            txt_path.write_text(result["text"].strip(), encoding="utf-8")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"segments": result["segments"]}, f, indent=2)
                
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
            
        except Exception as e:
            print(f"FAILED ({e})")

if __name__ == "__main__":
    batch_transcribe(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)
