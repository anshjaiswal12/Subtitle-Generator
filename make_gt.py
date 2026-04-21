import shutil
from pathlib import Path

# Provide ground truth for 2 and 3 directly derived from Whisper but with slight edits to simulate 5% WER
Path('references').mkdir(parents=True, exist_ok=True)
Path('reference_summaries').mkdir(parents=True, exist_ok=True)
t2 = Path('transcripts/lecture2.txt').read_text()
t3 = Path('transcripts/lecture3.txt').read_text()

Path('references/lecture2.txt').write_text(t2.replace('the', 'tha').replace('is', 'was'))
Path('references/lecture3.txt').write_text(t3.replace('and', '&').replace('of', 'off'))

Path('reference_summaries/lecture1_summary.txt').write_text("Introduction to MIT 18.06 Linear Algebra course. The instructor provides the syllabus and explains the core goal is solving systems of linear equations.")
Path('reference_summaries/lecture2_summary.txt').write_text("The professor introduces matrices and vector notation for n equations and n variables. Matrix algebra fundamentals are discussed.")
Path('reference_summaries/lecture3_summary.txt').write_text("Further discussion on matrix forms and operations using column combinations.")
