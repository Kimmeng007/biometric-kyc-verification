"""Run the full pipeline end to end: data -> face -> voice -> fusion -> demo.

    python run_all.py
"""
import subprocess
import sys

STEPS = [
    ["scripts/download_data.py"],
    ["-m", "face_verification.evaluate"],
    ["-m", "voice_verification.evaluate"],
    ["-m", "fusion.evaluate"],
    ["scripts/demo.py"],
]

for step in STEPS:
    print(f"\n>>> python {' '.join(step)}", flush=True)
    subprocess.run([sys.executable, *step], check=True)
