#!/usr/bin/env python3
"""
verify_setup.py
───────────────
Imports every library required by the Subtitle Generator & Summarizer
pipeline, prints its version, checks for ffmpeg on PATH, and flags
GPU / CPU considerations for running Whisper on a standard laptop.

Usage:
    python verify_setup.py
"""

import importlib
import shutil
import subprocess
import sys

# ── colour helpers (ANSI) ──────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {GREEN}✔{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✘{RESET} {msg}")


# ── 1. Python library verification ────────────────────────────────
LIBRARIES = [
    # (import name,      pip name,          version attribute)
    ("whisper",          "openai-whisper",   "__version__"),
    ("torch",            "torch",            "__version__"),
    ("transformers",     "transformers",     "__version__"),
    ("ffmpeg",           "ffmpeg-python",    None),           # no __version__
    ("moviepy",          "moviepy",          "__version__"),
    ("jiwer",            "jiwer",            "__version__"),
    ("rouge_score",      "rouge-score",      None),
    ("sentencepiece",    "sentencepiece",    "__version__"),
    ("jupyter_core",     "jupyter",          "__version__"),
    ("IPython",          "ipython",          "__version__"),
]


def check_libraries() -> int:
    """Import each library and report its version. Returns failure count."""
    print(f"\n{BOLD}[1/3] Checking Python libraries{RESET}")
    print(f"      Python {sys.version}\n")

    failures = 0
    for import_name, pip_name, ver_attr in LIBRARIES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, ver_attr, "installed (version N/A)") if ver_attr else "installed (version N/A)"
            _ok(f"{pip_name:25s} → {version}")
        except ImportError:
            _fail(f"{pip_name:25s} → NOT FOUND  (pip install {pip_name})")
            failures += 1
        except Exception as exc:
            _fail(f"{pip_name:25s} → import error: {exc}")
            failures += 1

    return failures


# ── 2. ffmpeg binary verification ─────────────────────────────────
def check_ffmpeg() -> bool:
    """Verify that ffmpeg is callable from the command line."""
    print(f"\n{BOLD}[2/3] Checking ffmpeg binary{RESET}\n")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        _fail("ffmpeg not found on PATH.")
        _warn("Install it via:  sudo apt install ffmpeg   (Debian/Ubuntu)")
        _warn("             or: conda install -c conda-forge ffmpeg")
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=10
        )
        first_line = result.stdout.split("\n")[0]
        _ok(f"ffmpeg found at {ffmpeg_path}")
        _ok(f"{first_line}")
        return True
    except Exception as exc:
        _fail(f"ffmpeg found but could not run: {exc}")
        return False


# ── 3. GPU / CPU considerations ───────────────────────────────────
def check_gpu() -> None:
    """Report CUDA availability and give Whisper model guidance."""
    print(f"\n{BOLD}[3/3] GPU / CPU considerations for Whisper{RESET}\n")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            _ok(f"CUDA available — {gpu_name} ({vram:.1f} GB VRAM)")
            _ok('Recommended Whisper model: "base" (≈1 GB VRAM)')
            if vram >= 5:
                _ok('You can also try "small" or "medium" for better accuracy.')
            else:
                _warn('"small" model needs ~2 GB VRAM; "medium" needs ~5 GB.')
        else:
            _warn("No CUDA GPU detected — Whisper will run on CPU.")
            _warn('Use model="tiny" or "base" to keep transcription times reasonable.')
            _warn("Expect ~1–3× real-time on a modern laptop CPU for 'base'.")
            _warn("Tip: install the CUDA-enabled PyTorch wheel if you have an NVIDIA GPU:")
            _warn("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        _fail("torch is not installed — cannot check GPU availability.")


# ── Summary ───────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{'='*60}")
    print(f" Subtitle Generator & Summarizer — Environment Verification")
    print(f"{'='*60}")

    lib_failures = check_libraries()
    ffmpeg_ok = check_ffmpeg()
    check_gpu()

    # Final report
    print(f"\n{'='*60}")
    if lib_failures == 0 and ffmpeg_ok:
        _ok(f"{BOLD}All checks passed. Environment is ready!{RESET}")
    else:
        if lib_failures:
            _fail(f"{lib_failures} library/libraries missing — run:  pip install -r requirements.txt")
        if not ffmpeg_ok:
            _fail("ffmpeg is missing — install it before running the pipeline.")
    print()


if __name__ == "__main__":
    main()
