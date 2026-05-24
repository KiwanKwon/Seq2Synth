"""Shared filesystem paths for the Seq2Synth benchmark.

All paths are anchored to this repository checkout so the benchmark can be
run from any user account without editing absolute paths.
"""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
REAL_DIR = DATA_DIR / "real"
SYNTH_DIR = DATA_DIR / "synthetic"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"
