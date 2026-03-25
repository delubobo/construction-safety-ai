import sys
import os
from pathlib import Path

# Add repo root to path so all imports resolve
root = Path(__file__).parent
sys.path.insert(0, str(root))

# Load .env for local dev (HF Spaces injects secrets as env vars automatically)
_env = root / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# Import main — executes all Streamlit commands
from app import main  # noqa: F401, E402
