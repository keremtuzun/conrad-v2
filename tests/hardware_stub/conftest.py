from __future__ import annotations

import sys
from pathlib import Path

_STUB = str(Path(__file__).resolve().parent)
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)
