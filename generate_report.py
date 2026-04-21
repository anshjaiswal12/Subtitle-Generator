#!/usr/bin/env python3
"""
generate_report.py
──────────────────
Produces a final aggregated evaluation report by parsing the output files from
evaluate_wer.py and evaluate_rouge.py. 

Combines data into a polished Markdown document inside /reports.
"""

from pathlib import Path

# ── project paths ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "reports"
WER_PATH = REPORTS_DIR / "wer_results.txt"
ROUGE_PATH = REPORTS_DIR / "rouge_results.txt"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"

def generate_report():
    print(f"\nGenerating final evaluation report...")

    wer_content = "WER data not found. Run evaluate_wer.py first."
    if WER_PATH.exists():
        wer_content = WER_PATH.read_text(encoding="utf-8")

    rouge_content = "ROUGE data not found. Run evaluate_rouge.py first."
    if ROUGE_PATH.exists():
        rouge_content = ROUGE_PATH.read_text(encoding="utf-8")

    report = f"""# Subtitle Generator and Summarizer — Final Evaluation Report

## 1. Project Overview

This project implements a fully automated pipeline to process educational lecture videos
into structured subtitle files (.srt) and concise text summaries (.txt). The system is
designed for edtech platforms seeking to improve student engagement and accessibility.
Given a raw .mp4 lecture video, the pipeline extracts audio, transcribes it using
OpenAI Whisper, generates SRT-formatted captions preserving Whisper segment timestamps,
chunks the transcript at sentence boundaries, and finally runs BART-based abstractive
summarization constrained to ≤100 words per video. Outputs are evaluated against
human references using WER (transcription accuracy) and ROUGE (summary quality) metrics.

## 2. Model Choices and Justification

| Stage          | Model                        | Justification |
|----------------|------------------------------|---------------|
| Audio extract  | ffmpeg-python (16kHz mono)   | Industry standard; Whisper requires exactly 16kHz mono WAV input |
| Transcription  | OpenAI Whisper `base`        | Balances speed and accuracy; GPU-accelerated on CUDA; works offline |
| Summarization  | `facebook/bart-large-cnn`    | State-of-the-art abstractive seq2seq; pre-trained on news summaries; produces coherent 3–5 sentence abstracts |
| Fallback       | `google/flan-t5-base`        | Lighter alternative for CPU-only environments with <8GB RAM |

## 3. Word Error Rate (WER) Evaluation
*Measures transcription fidelity against manually typed ground-truth transcripts.*
*Target: WER < 30% per video.*

```text
{wer_content.strip()}
```

## 4. ROUGE Score Evaluation
*Measures semantic overlap (F1) of AI-generated summaries vs human-written references.*
*Targets: ROUGE-1 > 0.25, ROUGE-L > 0.20.*

```text
{rouge_content.strip()}
```

## 5. Pipeline Limitations

1. **Whisper Base Accuracy**: The `base` model can hallucinate on heavily accented speech,
   background noise, or domain-specific vocabulary (e.g., advanced mathematics). Upgrading
   to `whisper medium` or `large-v3` significantly reduces WER at the cost of inference time.

2. **Non-English Content**: Whisper auto-detects language but performs best on English.
   Lectures with code-switching or non-English segments may produce garbled transcripts.

3. **BART Compression Artefacts**: With a strict ≤100-word cap on summaries, very dense
   lectures (>45 minutes) may lose important contextual information in the abstraction pass.

4. **Short Clip Hallucination**: Clips under 2 minutes may cause Whisper to repeat phrases
   or output in another language due to insufficient audio context.

5. **Reference Dependency**: WER and ROUGE scores require manually curated ground-truth files.
   Without them, quantitative evaluation cannot be performed and scores will not be reported.
"""

    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"✅ Final report saved to {FINAL_REPORT_PATH.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    generate_report()
