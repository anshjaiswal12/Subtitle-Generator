#!/usr/bin/env python3
"""
utils.py
────────
Shared utility functions for the Subtitle Generator & Summarizer pipeline.
Imported by individual stage scripts and the Jupyter notebook.
"""

from pathlib import Path
from typing import Any


# ══════════════════════════════════════════════════════════════════════
#  SRT GENERATION
# ══════════════════════════════════════════════════════════════════════

def format_timestamp(seconds: float) -> str:
    """
    Convert a time value in seconds to SRT timestamp format.

    Parameters
    ----------
    seconds : float
        Time in seconds (e.g., 63.217).

    Returns
    -------
    str
        Timestamp string in ``HH:MM:SS,mmm`` format (e.g., ``00:01:03,217``).
    """
    if seconds < 0:
        seconds = 0.0

    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int(round((seconds - int(seconds)) * 1000))

    # Guard against rounding 999.5+ → 1000 ms
    if millis >= 1000:
        millis = 0
        secs += 1
        if secs >= 60:
            secs = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def convert_to_srt(segments: list[dict[str, Any]], output_path: str | Path) -> Path:
    """
    Convert a list of Whisper transcript segments to a valid .srt subtitle file.

    Each segment must contain:
      - ``start`` (float): start time in seconds
      - ``end``   (float): end time in seconds
      - ``text``  (str)  : subtitle text

    SRT spec followed:
      - Sequential 1-based index
      - ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` timestamp line
      - Subtitle text (leading/trailing whitespace stripped)
      - Blank line separating entries

    Parameters
    ----------
    segments : list[dict]
        Whisper-style segments with 'start', 'end', and 'text' keys.
    output_path : str | Path
        Destination path for the .srt file.

    Returns
    -------
    Path
        The resolved path of the written .srt file.

    Raises
    ------
    ValueError
        If *segments* is empty or a segment is missing required keys.
    """
    output_path = Path(output_path)

    if not segments:
        raise ValueError("Segment list is empty — cannot generate SRT file.")

    required_keys = {"start", "end", "text"}
    srt_blocks: list[str] = []

    for idx, seg in enumerate(segments, start=1):
        missing = required_keys - seg.keys()
        if missing:
            raise ValueError(
                f"Segment {idx} is missing required key(s): {missing}"
            )

        start_ts = format_timestamp(seg["start"])
        end_ts   = format_timestamp(seg["end"])
        text     = seg["text"].strip()

        srt_blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")

    # Join with blank lines; ensure file ends with a newline
    srt_content = "\n".join(srt_blocks) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt_content, encoding="utf-8")

    return output_path.resolve()


# ══════════════════════════════════════════════════════════════════════
#  TRANSCRIPT CHUNKING  (placeholder — will be fleshed out later)
# ══════════════════════════════════════════════════════════════════════

def chunk_transcript(text: str, max_words: int = 200) -> list[str]:
    """
    Split *text* into chunks of approximately *max_words* words,
    breaking only at sentence boundaries using NLTK.

    Parameters
    ----------
    text : str
        The full transcript text.
    max_words : int
        Target maximum words per chunk.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    if not text or not text.strip():
        raise ValueError("Empty transcript passed to chunk_transcript")

    import nltk
    
    # Ensure the required punkt tokeniser data is available
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    sentences = nltk.sent_tokenize(text.strip())
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        
        # If adding this sentence pushes us over the limit (and we already have sentences),
        # save the current chunk and start a new one.
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    # Append any remaining sentences as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
