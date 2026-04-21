#!/usr/bin/env python3
"""
evaluate_rouge.py
─────────────────
Calculates ROUGE-1, ROUGE-2, and ROUGE-L F1 scores using the `rouge-score`
library to evaluate the quality of AI-generated summaries against
human-written reference summaries.

Usage:
    python evaluate_rouge.py
    python evaluate_rouge.py --hyp_dir summaries --ref_dir reference_summaries
"""

import argparse
import sys
from pathlib import Path

from rouge_score import rouge_scorer

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HYP_DIR = PROJECT_ROOT / "summaries"
DEFAULT_REF_DIR = PROJECT_ROOT / "reference_summaries"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "rouge_results.txt"


def evaluate_rouge(hyp_dir: Path, ref_dir: Path, report_path: Path) -> None:
    """
    Compute ROUGE scores for all matching abstract pairs, print them to the
    terminal, and export a clean table structure to the reports folder.
    """
    print(f"\n{BOLD}ROUGE Score Evaluation{RESET}")
    print(f"  Generated Summaries : {hyp_dir}")
    print(f"  Reference Summaries : {ref_dir}\n")

    # 1. Directory Validation & Missing Reference Guidance
    if not ref_dir.exists() or not any(ref_dir.glob("*.txt")):
        print(f"  {YELLOW}⚠ Missing Reference Summaries{RESET}")
        print(f"  No ground truth summaries found in '{ref_dir}'.\n")
        print(f"  {BOLD}How to create a simple reference for testing:{RESET}")
        print("    1. Pick one of your videos (e.g., lecture1.mp4).")
        print("    2. Manually write a fast 2–3 sentence (≤100 words) summary of the content.")
        print("    3. Save your text into a file named 'lecture1_summary.txt' (or matching generated name).")
        print("    4. Save it inside the 'reference_summaries' folder.")
        print("    5. Re-run this script to evaluate summarization quality.\n")
        sys.exit(0)

    if not hyp_dir.exists():
        print(f"  {RED}✘{RESET} Output directory does not exist: {hyp_dir}")
        sys.exit(1)

    # 2. Collect matched files
    ref_files = sorted([f for f in ref_dir.iterdir() if f.is_file() and f.suffix == ".txt"])
    results = []

    # Initialize ROUGE scorer (measuring 1-gram, 2-gram, and Longest Common Subsequence)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    print(f"  Starting evaluation…\n")

    for ref_path in ref_files:
        hyp_path = hyp_dir / ref_path.name
        
        if not hyp_path.exists():
            print(f"  {YELLOW}⚠{RESET} {ref_path.name} → skipped (no matching AI summary)")
            continue

        try:
            ref_text = ref_path.read_text(encoding="utf-8").strip()
            hyp_text = hyp_path.read_text(encoding="utf-8").strip()

            if not ref_text or not hyp_text:
                print(f"  {YELLOW}⚠{RESET} {ref_path.name} → skipped (empty file)")
                continue

            # Calculate scores
            scores = scorer.score(ref_text, hyp_text)
            
            r1_f1 = scores['rouge1'].fmeasure
            r2_f1 = scores['rouge2'].fmeasure
            rl_f1 = scores['rougeL'].fmeasure

            results.append((ref_path.name, r1_f1, r2_f1, rl_f1))

            print(f"  ✔ {BOLD}{ref_path.name:25s}{RESET}")
            print(f"      ROUGE-1: {GREEN}{r1_f1:.4f}{RESET} | ROUGE-2: {GREEN}{r2_f1:.4f}{RESET} | ROUGE-L: {GREEN}{rl_f1:.4f}{RESET}")

        except Exception as e:
            print(f"  {RED}✘{RESET} {ref_path.name} → ERROR: {e}")

    # 3. Summary & Export
    if not results:
        print(f"\n  {YELLOW}⚠{RESET} No valid matching .txt pairs were found.")
        sys.exit(0)

    # Calculate robust averages
    avg_r1 = sum(r[1] for r in results) / len(results)
    avg_r2 = sum(r[2] for r in results) / len(results)
    avg_rl = sum(r[3] for r in results) / len(results)

    print(f"\n{'─' * 60}")
    print(f"  {BOLD}Global Averages (F1 Score):{RESET}")
    print(f"    ROUGE-1: {GREEN}{avg_r1:.4f}{RESET}")
    print(f"    ROUGE-2: {GREEN}{avg_r2:.4f}{RESET}")
    print(f"    ROUGE-L: {GREEN}{avg_rl:.4f}{RESET}")
    print(f"\n  {BOLD}Files Evaluated:{RESET} {len(results)}")

    # Write report table
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ROUGE Evaluation Report\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Filename':^25} | {'ROUGE-1':^10} | {'ROUGE-2':^10} | {'ROUGE-L':^10}\n")
        f.write("-" * 65 + "\n")
        
        for filename, r1, r2, rl in results:
            f.write(f"{filename:25s} | {r1:10.4f} | {r2:10.4f} | {rl:10.4f}\n")
            
        f.write("=" * 65 + "\n")
        f.write(f"{'AVERAGE':25s} | {avg_r1:10.4f} | {avg_r2:10.4f} | {avg_rl:10.4f}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Total Evaluated: {len(results)}\n")

    print(f"  📝 Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate ROUGE scores against reference summaries.")
    parser.add_argument("--hyp_dir", type=Path, default=DEFAULT_HYP_DIR, help="Folder with model-generated summaries.")
    parser.add_argument("--ref_dir", type=Path, default=DEFAULT_REF_DIR, help="Folder with human-written references.")
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT_PATH, help="Path to save output report.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_rouge(args.hyp_dir, args.ref_dir, args.report_path)
