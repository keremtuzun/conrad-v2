"""Run bundles, backup/restore and digest-verified replay inputs (ch2 Deterministic replay, ch34, SS-02/04/08).

A run bundle manifest lists every object the run references. Replay and evaluation load
nothing until every digest exists and hashes correctly; a missing artifact fails closed with
its digest named. Missing original evidence is never synthesized.

implementation_status: FROZEN_CONTRACT
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from conrad.persistence import db
from conrad.persistence.object_store import MissingObjectError, ObjectStore, ObjectStoreError
from conrad.schemas.base import ARCHITECTURE_ID, SCHEMA_VERSION, STACK_ID, ConradModel

REQUIRED_REPLAY_KEYS = (
    "scenario_seed",
    "scenario_version",
    "twin_versions",
    "model_versions",
    "robot_config_digest",
    "sensor_configuration",
    "config_digest",
    "git_commit",
    "seeds",
    "architecture_id",
    "stack_id",
)


class ReplayIntegrityError(RuntimeError):
    """A replay/restore input is missing, corrupt or incompatible. Carries the offending names."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("replay integrity failure: " + "; ".join(problems))
        self.problems = problems


class BundleManifest(ConradModel):
    run_id: str
    architecture_id: str = ARCHITECTURE_ID
    stack_id: str = STACK_ID
    schema_version: str = SCHEMA_VERSION
    replay_inputs: dict[str, Any]
    object_digests: tuple[str, ...]
    files: dict[str, str]  # relative path in run dir -> sha256

    def manifest_digest(self) -> str:
        return self.content_digest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bundle_manifest(
    run_dir: Path, run_id: str, replay_inputs: dict[str, Any], object_digests: list[str]
) -> BundleManifest:
    missing = [k for k in REQUIRED_REPLAY_KEYS if k not in replay_inputs]
    if missing:
        raise ReplayIntegrityError([f"replay input missing: {k}" for k in missing])
    files = {
        p.relative_to(run_dir).as_posix(): _sha256_file(p)
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != "bundle_manifest.json" and "notes" not in p.relative_to(run_dir).parts
    }
    manifest = BundleManifest(
        run_id=run_id,
        replay_inputs=replay_inputs,
        object_digests=tuple(sorted(set(object_digests))),
        files=files,
    )
    (run_dir / "bundle_manifest.json").write_text(manifest.canonical_json(), encoding="utf-8")
    return manifest


def load_bundle_manifest(run_dir: Path) -> BundleManifest:
    path = run_dir / "bundle_manifest.json"
    if not path.exists():
        raise ReplayIntegrityError([f"bundle manifest missing: {path}"])
    return BundleManifest.model_validate_json(path.read_text(encoding="utf-8"))


def verify_bundle(run_dir: Path, store: ObjectStore) -> BundleManifest:
    """SS-04: fail closed before any processing, naming every missing or corrupt artifact."""
    manifest = load_bundle_manifest(run_dir)
    problems: list[str] = []
    if manifest.architecture_id != ARCHITECTURE_ID or manifest.stack_id != STACK_ID:
        problems.append(f"identity mismatch: {manifest.architecture_id}/{manifest.stack_id}")
    for rel, digest in manifest.files.items():
        path = run_dir / rel
        if not path.exists():
            problems.append(f"run file missing: {rel}")
        elif _sha256_file(path) != digest:
            problems.append(f"run file corrupt: {rel} (expected sha256:{digest})")
    for digest in manifest.object_digests:
        try:
            store.get_bytes(digest)
        except MissingObjectError:
            problems.append(f"object missing: sha256:{digest}")
        except ObjectStoreError as exc:
            problems.append(str(exc))
    if problems:
        raise ReplayIntegrityError(problems)
    return manifest


class BackupReport(ConradModel):
    backup_dir: str
    database_sha256: str
    object_count: int
    restore_verified: bool = False
    problems: tuple[str, ...] = ()


def backup(db_path: Path, store: ObjectStore, digests: list[str], dest: Path) -> BackupReport:
    """Online-consistent SQLite backup plus every referenced object, as one manifest."""
    dest.mkdir(parents=True, exist_ok=True)
    target_db = dest / "conrad.sqlite"
    src = sqlite3.connect(str(db_path))
    out = sqlite3.connect(str(target_db))
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    obj_dir = dest / "objects" / "sha256"
    obj_dir.mkdir(parents=True, exist_ok=True)
    for digest in sorted(set(digests)):
        data = store.get_bytes(digest)  # verifies before copying
        (obj_dir / digest).write_bytes(data)
    report = BackupReport(
        backup_dir=str(dest), database_sha256=_sha256_file(target_db), object_count=len(set(digests))
    )
    (dest / "backup_manifest.json").write_text(
        json.dumps({**report.model_dump(), "digests": sorted(set(digests))}, indent=2), encoding="utf-8"
    )
    return report


def restore(backup_dir: Path, db_target: Path, store_root: Path) -> BackupReport:
    """SS-08: restore is valid only if integrity, digests and migration state all check out."""
    meta = json.loads((backup_dir / "backup_manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    src_db = backup_dir / "conrad.sqlite"
    if _sha256_file(src_db) != meta["database_sha256"]:
        problems.append("backup database hash mismatch")
    db_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_db, db_target)
    store = ObjectStore(store_root)
    for digest in meta["digests"]:
        blob = backup_dir / "objects" / "sha256" / digest
        if not blob.exists():
            problems.append(f"backup object missing: sha256:{digest}")
            continue
        data = blob.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            problems.append(f"backup object corrupt: sha256:{digest}")
            continue
        store.put_bytes(data, "application/octet-stream")
    engine = db.make_engine(db_target)
    try:
        db.integrity_check(engine)
        db.check_migration_state(engine)
    except db.DatabaseError as exc:
        problems.append(str(exc))
    finally:
        engine.dispose()
    problems += store.verify(meta["digests"])
    report = BackupReport(
        backup_dir=str(backup_dir),
        database_sha256=meta["database_sha256"],
        object_count=len(meta["digests"]),
        restore_verified=not problems,
        problems=tuple(problems),
    )
    if problems:
        raise ReplayIntegrityError(list(problems))
    return report
