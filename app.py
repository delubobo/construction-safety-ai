"""
Hugging Face Spaces entry point.
HF Spaces expects the Streamlit app at the repo root named app.py.
This simply re-runs the actual app module.
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
runpy.run_module("app.main", run_name="__main__", alter_sys=True)
