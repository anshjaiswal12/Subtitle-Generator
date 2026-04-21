# Subtitle Generator and Summarizer

An AI pipeline that generates subtitles and concise summaries for lecture videos using OpenAI Whisper and BART.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
1. python extract_audio.py       # Extract mono 16kHz WAV from .mp4
2. python transcribe.py          # Whisper transcription → .txt + .json segments
3. python summarize.py           # BART summarization → .txt per video
4. python evaluate_wer.py        # Word Error Rate against reference transcripts
5. python evaluate_rouge.py      # ROUGE-1/2/L against reference summaries
6. python generate_report.py     # Final Markdown evaluation report
7. python package_submission.py  # Bundle all outputs into a .zip
```

## Output

| Folder | Contents |
|--------|----------|
| `/audio` | Extracted `.wav` files (mono, 16kHz) |
| `/transcripts` | `.txt` transcripts + `.json` segment files |
| `/subtitles` | `.srt` subtitle files (SRT spec compliant) |
| `/summaries` | `.txt` summaries (≤100 words, 3–5 sentences) |
| `/reports` | `wer_results.txt`, `rouge_results.txt`, `final_report.md` |
| `/notebooks` | `subtitle_summarizer.ipynb` end-to-end demo |

## Models

| Stage | Model |
|-------|-------|
| Speech-to-Text | OpenAI Whisper `base` (GPU accelerated via CUDA) |
| Summarization | `facebook/bart-large-cnn` (Hugging Face) |
| Fallback | `google/flan-t5-base` (CPU-only alternative) |

## Evaluation Targets

- **WER** (Word Error Rate): < 30% per video
- **ROUGE-1** F1: > 0.25
- **ROUGE-L** F1: > 0.20

## Requirements

See `requirements.txt`. Key dependencies: `openai-whisper`, `transformers`, `torch`, `ffmpeg-python`, `jiwer`, `rouge-score`, `nltk`.
