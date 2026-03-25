import sys
import os
import importlib.util
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))

# Load .env for local dev (HF Spaces uses secrets as env vars automatically)
_env = root / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Load app/main.py by file path — avoids conflict between app.py and app/ package
spec = importlib.util.spec_from_file_location(
    "streamlit_main", root / "app" / "main.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
