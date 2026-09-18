from __future__ import annotations

import faulthandler
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_STUB = str(Path(__file__).resolve().parents[3] / "hardware_stub")
if _STUB not in sys.path:
    sys.path.insert(0, _STUB)

SOCKET_TEST_HARD_TIMEOUT_S = 30.0


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:
    """pytest-level guard (no pytest-timeout dependency): a hung socket test dumps stacks and exits."""
    faulthandler.dump_traceback_later(SOCKET_TEST_HARD_TIMEOUT_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()
