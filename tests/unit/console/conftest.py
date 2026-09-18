"""A real run bundle built once per module from the Phase-1 fake full system."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from console_bundle_fixture import make_bundle


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_bundle(tmp_path_factory.mktemp("console") / "run-a")


@pytest.fixture
def bundle_copy(bundle_dir: Path, tmp_path: Path) -> Path:
    dest = tmp_path / "run-copy"
    shutil.copytree(bundle_dir, dest)
    return dest
