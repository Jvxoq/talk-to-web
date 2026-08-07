import sys
from pathlib import Path

# Make `app` and `tests` importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
