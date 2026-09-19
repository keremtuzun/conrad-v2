"""SS-09: CI catches schema/migration/lock/secret violations."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext

from conrad.persistence.db import make_engine, metadata, migrate

ROOT = Path(__file__).resolve().parent.parent.parent


def test_no_untracked_migration(tmp_path: Path) -> None:
    """Table metadata and the Alembic revisions must describe the same schema."""
    db_path = tmp_path / "drift.sqlite"
    migrate(db_path)
    engine = make_engine(db_path)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), metadata)
    engine.dispose()
    assert diff == [], f"schema changed without a migration: {diff}"


def test_lockfile_is_committed_and_current() -> None:
    assert (ROOT / "uv.lock").exists()
    uv = os.environ.get("UV") or shutil.which("uv")  # `uv run` exports UV=<path to the uv binary>
    if uv is None:
        pytest.skip("uv binary not discoverable outside `uv run`")
    result = subprocess.run([uv, "lock", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-400:]


def test_secret_scanner_detects_planted_credentials() -> None:
    spec = importlib.util.spec_from_file_location("secret_scan", ROOT / "scripts" / "secret_scan.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    assert module.PATTERNS["aws access key"].search(f"key = {fake_key}")
    assert module.PATTERNS["private key"].search("-----BEGIN " + "PRIVATE KEY-----")
    assert module.PATTERNS["assigned secret"].search('password = "' + "hunter2hunter2hunter2" + '"')
    assert module.FORBIDDEN_NAMES.search("configs/.env") and module.FORBIDDEN_NAMES.search("deploy/robot.pem")
    assert not module.PATTERNS["assigned secret"].search("required_secrets: tuple[str, ...] = ()")


def test_repository_is_currently_clean_of_secrets() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/secret_scan.py"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-600:]


def test_artifacts_are_gitignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/*" in ignore and ".env" in ignore and "configs/local.yaml" in ignore
