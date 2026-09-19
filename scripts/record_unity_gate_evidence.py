"""Record FORMAL evidence for gates I1, I2, I3 from the live Unity tests (ADR-0008).

Runs ``tests/unity_live/test_i{1,2,3}_unity.py`` against the built player, maps every gate criterion (names exactly
as in ``conrad.evaluation.gates``) to the pytest nodes that decide it, and writes
``artifacts/gates/<gate>/evidence_formal.json`` with the numbers the tests measured
(``artifacts/gates/<gate>/unity_measured.json``). A criterion is PASS only if all its nodes passed; a skipped node
(player binary absent) makes it NOT_RUN. The replay test of each gate is attached to every criterion.

Usage: python -m uv run python scripts/record_unity_gate_evidence.py [I1 I2 I3] [--no-run]
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conrad.evaluation.gates import (  # noqa: E402
    GATE_BY_ID,
    CriterionResult,
    CriterionStatus,
    EvidenceClass,
    GateEvidence,
    evaluate_gates,
    write_evidence,
)
from conrad.sim.mission.unity_run import git_commit, player_identity  # noqa: E402

T1 = "tests/unity_live/test_i1_unity.py::"
T2 = "tests/unity_live/test_i2_unity.py::"
T3 = "tests/unity_live/test_i3_unity.py::"
MODULES = {
    "I1": "tests/unity_live/test_i1_unity.py",
    "I2": "tests/unity_live/test_i2_unity.py",
    "I3": "tests/unity_live/test_i3_unity.py",
}
REPLAY = {
    "I1": [T1 + "test_bundle_replays_deterministically"],
    "I2": [
        T2 + "test_bundle_replays_deterministically[NAV-003]",
        T2 + "test_bundle_replays_deterministically[NAV-006]",
    ],
    "I3": [T3 + "test_bundle_replays_deterministically"],
}
PLAN: dict[str, list[tuple[str, list[str], list[str]]]] = {
    # (criterion, deciding nodes, measured keys in unity_measured.json)
    "I1": [
        (
            "robot moves through synthetic world",
            [T1 + "test_robot_moves_through_synthetic_world"],
            ["robot moves through synthetic world"],
        ),
        ("persistent map", [T1 + "test_persistent_map"], ["persistent map"]),
        ("geometry", [T1 + "test_geometry_occupied_claims_lie_on_twin2s_surfaces"], ["geometry"]),
        ("occupancy", [T1 + "test_occupancy_free_claims_agree_with_truth"], ["occupancy"]),
        ("coverage", [T1 + "test_coverage"], ["coverage"]),
        (
            "observed/inferred/unknown",
            [T1 + "test_observed_inferred_unknown_and_hidden_is_unknown"],
            ["observed/inferred/unknown"],
        ),
        ("uncertainty", [T1 + "test_uncertainty_is_decomposed_and_responds"], ["uncertainty"]),
        ("provenance", [T1 + "test_provenance_to_raw_unity_frames"], ["provenance"]),
        (
            "no Twin truth leakage",
            [T1 + "test_no_twin_truth_leakage_on_runtime_side"],
            ["no Twin truth leakage"],
        ),
    ],
    "I2": [
        ("reach waypoint", [T2 + "test_primitive[reach waypoint-benchmarks0]"], ["reach waypoint"]),
        ("avoid obstacle", [T2 + "test_primitive[avoid obstacle-benchmarks1]"], ["avoid obstacle"]),
        ("follow pipeline", [T2 + "test_primitive[follow pipeline-benchmarks2]"], ["follow pipeline"]),
        ("station keep", [T2 + "test_primitive[station keep-benchmarks3]"], ["station keep"]),
        ("uses estimated state", [T2 + "test_uses_estimated_state"], ["uses estimated state"]),
        ("NAV-001..NAV-006", [T2 + "test_nav_001_to_006"], ["NAV-001..NAV-006"]),
    ],
    "I3": [
        (
            "Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T",
            [
                T3 + "test_twin2t_twin2s_unity_to_model2s_and_model2t",
                T3 + "test_no_twin_truth_leakage_on_runtime_side",
            ],
            ["Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T", "leakage"],
        ),
        (
            "robot sees only partial infrastructure",
            [T3 + "test_robot_sees_only_partial_infrastructure"],
            ["robot sees only partial infrastructure"],
        ),
        (
            "persistent technical belief",
            [T3 + "test_persistent_technical_belief"],
            ["persistent technical belief"],
        ),
    ],
}


def run_module(gate: str) -> dict[str, str]:
    xml_path = ROOT / "artifacts" / "gates" / gate / "unity_pytest.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    log = xml_path.with_suffix(".txt")
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-rA",
                "-p",
                "no:cacheprovider",
                f"--junitxml={xml_path}",
                MODULES[gate],
            ],
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return read_outcomes(xml_path)


def read_outcomes(xml_path: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    if not xml_path.exists():
        return results
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        module = case.get("classname", "").split(".")[-1]  # rootdir-relative or bare module name
        skip = case.find("skipped")
        # a strict-xfail node encodes a gate criterion that currently FAILS: it is a failure, not a skip
        xfailed = skip is not None and skip.get("type") == "pytest.xfail"
        bad = case.find("failure") is not None or case.find("error") is not None or xfailed
        skipped = skip is not None and not xfailed
        results[f"{module}::{case.get('name', '')}"] = "FAIL" if bad else ("SKIP" if skipped else "PASS")
    return results


def _key(node: str) -> str:
    path, name = node.split("::", 1)
    return f"{Path(path).stem}::{name}"


def record(gate: str, rerun: bool = True) -> GateEvidence:
    outcome = (
        run_module(gate) if rerun else read_outcomes(ROOT / "artifacts" / "gates" / gate / "unity_pytest.xml")
    )
    mpath = ROOT / "artifacts" / "gates" / gate / "unity_measured.json"
    data = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {"meta": {}, "criteria": {}}
    names = {c for c, _, _ in PLAN[gate]}
    if names != set(GATE_BY_ID[gate].criteria):
        raise SystemExit(f"{gate}: criterion names differ from conrad.evaluation.gates")
    criteria = []
    for name, nodes, keys in PLAN[gate]:
        nodes = nodes + REPLAY[gate]
        statuses = [outcome.get(_key(n), "NOT_RUN") for n in nodes]
        if any(s == "FAIL" for s in statuses):
            status = CriterionStatus.FAIL
        elif any(s in ("NOT_RUN", "SKIP") for s in statuses):
            status = CriterionStatus.NOT_RUN
        else:
            status = CriterionStatus.PASS
        values = {k: data["criteria"].get(k) for k in keys}
        replay = {k: v for k, v in data["criteria"].items() if k.startswith("replay")}
        text = "; ".join(f"{n.split('::')[-1]}={s}" for n, s in zip(nodes, statuses, strict=True))
        text += "; measured: " + json.dumps({**values, **replay}, sort_keys=True)
        criteria.append(CriterionResult(criterion=name, status=status, measured=text))
    note = (
        f"meta={json.dumps(data.get('meta', {}), sort_keys=True)}; "
        f"player={json.dumps(player_identity(), sort_keys=True)}; validity=L1_APPROXIMATE_PHYSICS, "
        "every physical parameter SYNTHETIC_ONLY"
    )
    if gate == "I3":
        note += "; official I3 status also needs gate 2T (produced by another workstream)"
    ev = GateEvidence(
        gate_id=gate,
        evidence_class=EvidenceClass.FORMAL,
        execution_path=GATE_BY_ID[gate].formal_path + " (built Unity V2 player, lock-step TCP)",
        git_commit=git_commit(),
        criteria=tuple(criteria),
        artifacts=(
            MODULES[gate],
            f"artifacts/gates/{gate}/unity_measured.json",
            f"artifacts/gates/{gate}/unity_pytest.xml",
            f"artifacts/unity/gate_runs/{gate}",
        ),
        notes=note,
    )
    write_evidence(ev)
    return ev


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rerun = "--no-run" not in sys.argv
    for g in args or ["I1", "I2", "I3"]:
        e = record(g, rerun)
        print(g, [f"{c.criterion}:{c.status.value}" for c in e.criteria])
    reports = evaluate_gates()
    for g in ("U0", "2S-FIRST", "I1", "I2", "2T", "I3"):
        r = reports[g]
        print(
            f"{g:<9} official={r.official_status.value} formal={r.formal_status.value} "
            f"blocked_by={','.join(r.blocking_upstream)}"
        )
