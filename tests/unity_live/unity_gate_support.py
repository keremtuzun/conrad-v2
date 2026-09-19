"""Shared helpers of the formal Unity gate tests I1 / I2 / I3 (ADR-0008 FORMAL path; U0 has its own harness).

* Seeds: held-out FINAL partition of ``configs/eval/partitions.yaml`` (domain ``mission``), purpose
  final_evaluation. The NAV benchmark noise seeds come from the ``nav`` final_test split
  (``configs/eval/partitions_nav.yaml``; see the I2 repair section of docs/audits/UNITY_INTEGRATION_I1_I3.md).
* Every test records what it measured with ``measured(gate, criterion, ...)``; the numbers land in
  ``artifacts/gates/<gate>/unity_measured.json`` and are copied into the formal evidence by
  ``scripts/record_unity_gate_evidence.py``.
* Run bundles are kept under ``artifacts/unity/gate_runs/<gate>/`` so the write-up can cite them.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from conrad.evaluation.partitions import split
from conrad.sim.mission.unity_faults import fault_cases

REPO = Path(__file__).resolve().parents[2]
GATE_RUNS = REPO / "artifacts" / "unity" / "gate_runs"
GATES = REPO / "artifacts" / "gates"
MISSION_CONFIG = "configs/sim/mission_test_small.yaml"
# NAV noise seeds: held-out final_test split of configs/eval/partitions_nav.yaml (7400001..7400006, digest-pinned,
# disjoint from partitions.yaml). The former I2 seeds 7300001..7300006 are CONTAMINATED_FOR_FINAL_EVALUATION
# (seen before the I2 repair) and are development seeds of that file now.
_NAV_FINAL = split("nav", "final_test", "final_evaluation")
NAV_SEEDS = dict(zip(_NAV_FINAL.families, _NAV_FINAL.world_seeds, strict=True))
# Fault cases (configs/sim/nav_fault_cases.yaml): case k runs on the k-th held-out NAV noise seed.
FAULT_CASES = tuple(fault_cases()["cases"])
FAULT_SEEDS = dict(zip(FAULT_CASES, _NAV_FINAL.world_seeds[: len(FAULT_CASES)], strict=True))
TWIN_ONLY_KEYS = ("true_world_entity_id", "entity_id_table", "true_robot_pose", "supervision", "twin2t_truth")
RUNTIME_TABLES = (
    "observations",
    "evidence",
    "belief_revisions",
    "provenance_nodes",
    "commands",
    "relationships",
)


def final_seeds() -> tuple[tuple[int, ...], tuple[str, ...]]:
    s = split("mission", "final_test", "final_evaluation")
    return s.world_seeds, s.families


def i1_seed() -> int:
    return final_seeds()[0][-1]


def i3_seed() -> int:
    return final_seeds()[0][-2]


def _to_json(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _to_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_json(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float):
        return round(v, 6)
    return v


def reset_measured(gate: str, meta: dict[str, Any]) -> None:
    path = GATES / gate / "unity_measured.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": _to_json(meta), "criteria": {}}, indent=1), encoding="utf-8")


def measured(gate: str, criterion: str, **values: Any) -> None:
    """Merge measured values for one criterion into ``artifacts/gates/<gate>/unity_measured.json``."""
    path = GATES / gate / "unity_measured.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"meta": {}, "criteria": {}}
    data["criteria"].setdefault(criterion, {}).update(_to_json(values))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def runtime_texts(run_dir: Path) -> Iterator[tuple[str, str]]:
    """Every deployment-side artifact of a bundle (events, mission/, SQLite payloads)."""
    yield "events.jsonl", (run_dir / "events.jsonl").read_text(encoding="utf-8")
    for p in sorted((run_dir / "mission").glob("*")):
        yield f"mission/{p.name}", p.read_text(encoding="utf-8")
    con = sqlite3.connect(str(run_dir / "conrad.sqlite"))
    try:
        for table in RUNTIME_TABLES:
            for (payload,) in con.execute(f"SELECT payload_json FROM {table}"):
                yield table, payload
    finally:
        con.close()


def _keys(obj: Any, out: set[str]) -> set[str]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            _keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _keys(v, out)
    return out


def leakage_scan(run_dir: Path, world_ids: set[str], twin_keys: set[str]) -> tuple[int, list[str]]:
    """(texts scanned, violations): world-entity UUIDs, twin-only keys or Twin2S visibility keys on the runtime side."""
    violations: list[str] = []
    scanned = 0
    for where, text in runtime_texts(run_dir):
        scanned += 1
        violations += [f"{where}: world entity {w}" for w in world_ids if w in text]
        if "visibility:" in text:
            violations.append(f"{where}: twin visibility key")
        for line in text.splitlines() or [text]:
            try:
                keys = _keys(json.loads(line), set())
            except json.JSONDecodeError:
                continue
            bad = sorted(k for k in keys if k in TWIN_ONLY_KEYS or k in twin_keys)
            if bad:
                violations.append(f"{where}: twin-only keys {bad}")
    return scanned, sorted(set(violations))
