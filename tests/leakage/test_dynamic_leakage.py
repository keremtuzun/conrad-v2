"""Dynamic leakage scan of a real FLAGSHIP-I4 run (INV-ARCH-01/07, CC-07).

Every runtime-side artifact (events, observations, evidence, belief revisions, provenance, commands and the
deployment mission/ artifacts) must contain no world-entity UUID and no twin-only key. Only truth/ (the
evaluation record), reports/ (evaluation output) and the bundle manifest may mention truth.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tests.acceptance._runs import flagship

TWIN_ONLY_KEYS = ("true_world_entity_id", "entity_id_table", "true_robot_pose", "supervision", "twin2t_truth")
TRUTH_TABLE_COLUMNS = {
    "observations": "payload_json",
    "evidence": "payload_json",
    "belief_revisions": "payload_json",
    "provenance_nodes": "payload_json",
    "commands": "payload_json",
    "relationships": "payload_json",
}


def _keys(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def _runtime_texts(run_dir: Path):
    yield "events.jsonl", (run_dir / "events.jsonl").read_text(encoding="utf-8")
    for p in sorted((run_dir / "mission").glob("*")):
        yield f"mission/{p.name}", p.read_text(encoding="utf-8")
    con = sqlite3.connect(str(run_dir / "conrad.sqlite"))
    try:
        for table, col in TRUTH_TABLE_COLUMNS.items():
            for (payload,) in con.execute(f"SELECT {col} FROM {table}"):
                yield table, payload
    finally:
        con.close()


def test_no_world_entity_id_or_twin_key_on_the_runtime_side():
    run_dir = Path(flagship()["run_dir"])
    meta = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
    world_ids = set(meta["world_entity_ids"])
    assert world_ids and meta["target_world_id"] in world_ids
    violations = []
    scanned = 0
    for where, text in _runtime_texts(run_dir):
        scanned += 1
        for wid in world_ids:
            if wid in text:
                violations.append(f"{where}: world entity {wid}")
        if "visibility:" in text:
            violations.append(f"{where}: twin visibility key")
        for line in text.splitlines() or [text]:
            try:
                keys = _keys(json.loads(line), set())
            except json.JSONDecodeError:
                continue
            bad = [k for k in keys if k in TWIN_ONLY_KEYS]
            if bad:
                violations.append(f"{where}: twin-only keys {bad}")
    assert scanned > 100
    assert not violations, "\n".join(sorted(set(violations))[:20])


def test_registry_ids_are_not_world_ids():
    meta = json.loads(
        (Path(flagship()["run_dir"]) / "truth" / "truth_record.json").read_text(encoding="utf-8")
    )["meta"]
    assert not set(meta["registry_to_world"]) & set(meta["world_entity_ids"])
