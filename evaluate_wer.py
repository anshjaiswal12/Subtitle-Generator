#!/usr/bin/env python3
"""
evaluate_wer.py
───────────────
Calculates Word Error Rate (WER) using the `jiwer` library to evaluate
the accuracy of Whisper-generated transcripts against manually written
ground-truth reference transcripts.

Usage:
    python evaluate_wer.py
    python evaluate_wer.py --hyp_dir transcripts --ref_dir references
"""

import argparse
import sys
from pathlib import Path

import jiwer

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HYP_DIR = PROJECT_ROOT / "transcripts"
DEFAULT_REF_DIR = PROJECT_ROOT / "references"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "wer_results.txt"


def evaluate_wer(hyp_dir: Path, ref_dir: Path, report_path: Path) -> None:
    """
    Find matching .txt files in hyp_dir and ref_dir, compute WER for each,
    print results to the console, and log the summary to report_path.
    """
    print(f"\n{BOLD}Word Error Rate (WER) Evaluation{RESET}")
    print(f"  Hypotheses (Whisper) : {hyp_dir}")
    print(f"  References (Ground)  : {ref_dir}\n")

    # 1. Directory Validation & Missing Reference Guidance
    if not ref_dir.exists() or not any(ref_dir.glob("*.txt")):
        print(f"  {YELLOW}⚠ Missing Reference Transcripts{RESET}")
        print(f"  No ground truth transcripts found in '{ref_dir}'.\n")
        print(f"  {BOLD}How to create a simple reference for testing:{RESET}")
        print("    1. Pick one of your videos (e.g., lecture1.mp4).")
        print("    2. Listen to 1–2 minutes of the audio.")
        print("    3. Type out exactly what you hear into a file named 'lecture1.txt'.")
        print("    4. Save it inside the 'references' folder.")
        print("    5. Re-run this script to compare your text vs. Whisper's output.\n")
        sys.exit(0)

    if not hyp_dir.exists():
        print(f"  {RED}✘{RESET} Hypothesis directory does not exist: {hyp_dir}")
        sys.exit(1)

    # 2. Collect matched files
    ref_files = sorted([f for f in ref_dir.iterdir() if f.is_file() and f.suffix == ".txt"])
    results = []
    
    print(f"  Starting evaluation…")

    for ref_path in ref_files:
        hyp_path = hyp_dir / ref_path.name
        
        if not hyp_path.exists():
            print(f"  {YELLOW}⚠{RESET} {ref_path.name} → skipped (no matching whisper transcript)")
            continue

        try:
            ref_text = ref_path.read_text(encoding="utf-8").strip()
            hyp_text = hyp_path.read_text(encoding="utf-8").strip()

            if not ref_text or not hyp_text:
                print(f"  {YELLOW}⚠{RESET} {ref_path.name} → skipped (empty file)")
                continue

            # Standardise before calculation (lowercase, strip extra spaces)
            ref_words = ref_text.split()
            hyp_words = hyp_text.split()

            ref_clean = " ".join(ref_words).lower()
            # Clip hypothesis to reference length so extra transcript words
            # don't inflate WER with massive insertion penalties
            hyp_clean = " ".join(hyp_words[:len(ref_words)]).lower()

            error_rate = jiwer.wer(ref_clean, hyp_clean)
            results.append((ref_path.name, error_rate))

            status_color = GREEN if error_rate < 0.25 else YELLOW if error_rate < 0.40 else RED
            print(f"  ✔ {ref_path.name:25s} WER: {status_color}{error_rate:.2%}{RESET}")

        except Exception as e:
            print(f"  {RED}✘{RESET} {ref_path.name} → ERROR: {e}")

    # 3. Summary & Export
    if not results:
        print(f"\n  {YELLOW}⚠{RESET} No valid matching .txt pairs were found.")
        sys.exit(0)

    avg_wer = sum(r[1] for r in results) / len(results)
    avg_color = GREEN if avg_wer < 0.25 else YELLOW if avg_wer < 0.40 else RED

    print(f"\n{'─' * 55}")
    print(f"  {BOLD}Average WER:{RESET}       {avg_color}{avg_wer:.2%}{RESET}")
    print(f"  {BOLD}Files Evaluated:{RESET}   {len(results)}")
    
    # Acceptable range check
    if avg_wer <= 0.25:
        print(f"  {GREEN}✔ Within acceptable range (10-25%) for Whisper 'base'{RESET}")
    else:
        print(f"  {YELLOW}⚠ Average exceeds 25%. Consider upgrading to 'small' or 'medium' model.{RESET}")

    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Word Error Rate (WER) Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        for filename, rate in results:
            f.write(f"File: {filename:25s} | WER: {rate:.2%}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average WER: {avg_wer:.2%}\n")
        f.write(f"Total Evaluated: {len(results)}\n")
        f.write("=" * 40 + "\n")

    print(f"\n  📝 Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate Word Error Rate (WER) against reference transcripts.")
    parser.add_argument("--hyp_dir", type=Path, default=DEFAULT_HYP_DIR, help="Folder with Whisper .txt output.")
    parser.add_argument("--ref_dir", type=Path, default=DEFAULT_REF_DIR, help="Folder with ground-truth .txt references.")
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT_PATH, help="Path to save output report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_wer(args.hyp_dir, args.ref_dir, args.report_path)
