"""Read-only run-bundle loading for the console (ch37 Scope and authority, SS-04).

The bundle is verified with ``verify_bundle`` before anything else is read. The SQLite database is
copied to a private temporary directory and opened there, so the console never writes into the run
directory (not even WAL/SHM side files).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from conrad.persistence import db
from conrad.persistence.object_store import ObjectStore
from conrad.persistence.replay_store import BundleManifest, ReplayIntegrityError, verify_bundle
from conrad.persistence.repository import Repository
from conrad.runtime.event_log import read_events
from conrad.schemas.events import RuntimeEvent
from conrad.settings import REPO_ROOT, PathSettings

DB_SUFFIXES = (".sqlite", ".sqlite3", ".db")


@dataclass(frozen=True)
class IntegrityFailure:
    """The bundle failed verification. Only these problems may be shown."""

    run_dir: Path
    problems: tuple[str, ...]


@dataclass
class LoadedBundle:
    run_dir: Path
    manifest: BundleManifest
    events: list[RuntimeEvent]
    repo: Repository | None
    db_path: Path | None
    db_note: str
    config: dict[str, Any] | None
    mission: dict[str, Any] | None  # relative file name -> parsed JSON (or {"__error__": ...})
    truth: dict[str, Any] | None  # EVALUATION ONLY; loaded only when include_truth=True (ch37 UI-08)
    truth_present: bool
    metrics: Any | None
    warnings: list[str] = field(default_factory=list)
    _tmp: tempfile.TemporaryDirectory[str] | None = None

    def close(self) -> None:
        if self.repo is not None:
            self.repo.engine.dispose()
            self.repo = None
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_store(run_dir: Path, config: dict[str, Any] | None, override: Path | None) -> Path | None:
    candidates: list[Path] = []
    if override is not None:
        candidates.append(override)
    candidates.append(run_dir / "objects")
    if config:
        for key in ("object_store", "objects_dir"):
            val = _find_key(config, key)
            if isinstance(val, str):
                p = Path(val)
                candidates += [p] if p.is_absolute() else [run_dir / p, Path.cwd() / p]
    candidates.append(REPO_ROOT / PathSettings().object_store)
    return next((c for c in candidates if (c / "sha256").is_dir()), None)


def _find_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def _db_strings(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _db_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _db_strings(v)]
    return [obj] if isinstance(obj, str) and obj.lower().endswith(DB_SUFFIXES) else []


def _locate_db(run_dir: Path, config: dict[str, Any] | None) -> Path | None:
    for s in _db_strings(config or {}):
        p = Path(s)
        for cand in [p] if p.is_absolute() else [run_dir / p, Path.cwd() / p]:
            if cand.is_file():
                return cand
    found = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.suffix.lower() in DB_SUFFIXES)
    return found[0] if found else None


def _read_json_dir(folder: Path) -> dict[str, Any] | None:
    if not folder.is_dir():
        return None
    out: dict[str, Any] = {}
    for path in sorted(folder.rglob("*.json")):
        rel = path.relative_to(folder).as_posix()
        try:
            out[rel] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            out[rel] = {"__error__": f"unreadable JSON: {exc}"}
    return out


def _open_db(bundle: LoadedBundle, db_path: Path) -> None:
    tmp = tempfile.TemporaryDirectory(prefix="conrad-console-")
    bundle._tmp = tmp
    copy = Path(tmp.name) / db_path.name
    shutil.copyfile(db_path, copy)
    for side in ("-wal",):
        extra = db_path.with_name(db_path.name + side)
        if extra.exists():
            shutil.copyfile(extra, copy.with_name(copy.name + side))
    try:
        rel = db_path.resolve().relative_to(bundle.run_dir.resolve()).as_posix()
    except ValueError:
        rel = None
    expected = bundle.manifest.files.get(rel) if rel else None
    if expected is None:
        bundle.warnings.append(f"database {db_path} is not covered by the bundle manifest digests")
    elif _sha256(copy) != expected:  # changed between verification and copy
        raise ReplayIntegrityError([f"run file corrupt: {rel} (expected sha256:{expected})"])
    engine = db.make_engine(copy)
    try:
        bundle.repo = Repository(engine)
        bundle.db_note = f"read from a private copy of {db_path.name}"
    except db.DatabaseError as exc:
        engine.dispose()
        bundle.db_note = f"database unavailable: {exc}"


def open_bundle(
    run_dir: str | Path, object_store: str | Path | None = None, include_truth: bool = False
) -> LoadedBundle | IntegrityFailure:
    """Verify, then load. Returns ``IntegrityFailure`` (and loads nothing) if verification fails.

    ``truth/`` is never parsed unless ``include_truth`` is set (evaluation mode); otherwise only its
    existence is recorded.
    """
    run_dir = Path(run_dir)
    config: dict[str, Any] | None = None
    cfg_path = run_dir / "config.resolved.yaml"
    try:
        if cfg_path.is_file():
            loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else {"value": loaded}
        store_root = _resolve_store(run_dir, config, Path(object_store) if object_store else None)
        with tempfile.TemporaryDirectory(prefix="conrad-console-store-") as empty:
            # With no store on disk, an empty private store makes every referenced object fail closed.
            manifest = verify_bundle(run_dir, ObjectStore(store_root or empty))
    except ReplayIntegrityError as exc:
        return IntegrityFailure(run_dir, tuple(exc.problems))
    except (ValidationError, ValueError, OSError, yaml.YAMLError) as exc:
        return IntegrityFailure(run_dir, (f"bundle unreadable: {type(exc).__name__}: {exc}",))

    events_path = run_dir / "events.jsonl"
    bundle = LoadedBundle(
        run_dir=run_dir,
        manifest=manifest,
        events=list(read_events(events_path)) if events_path.is_file() else [],
        repo=None,
        db_path=None,
        db_note="not present",
        config=config,
        mission=_read_json_dir(run_dir / "mission"),
        truth=_read_json_dir(run_dir / "truth") if include_truth else None,
        truth_present=(run_dir / "truth").is_dir(),
        metrics=None,
    )
    if not events_path.is_file():
        bundle.warnings.append("events.jsonl not present")
    metrics_path = run_dir / "reports" / "metrics.json"
    if metrics_path.is_file():
        bundle.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    db_path = _locate_db(run_dir, config)
    if db_path is not None:
        bundle.db_path = db_path
        try:
            _open_db(bundle, db_path)
        except ReplayIntegrityError as exc:
            bundle.close()
            return IntegrityFailure(run_dir, tuple(exc.problems))
    return bundle
