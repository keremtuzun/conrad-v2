"""Record FORMAL evidence for the integration gates from the live Unity tests (ADR-0008).

Runs ``tests/unity_live/test_i<N>_unity.py`` against the built player, maps every gate criterion (names exactly
as in ``conrad.evaluation.gates``) to the pytest nodes that decide it, and writes
``artifacts/gates/<gate>/evidence_formal.json`` with the numbers the tests measured
(``artifacts/gates/<gate>/unity_measured.json``). A criterion is PASS only if all its nodes passed; a skipped node
(player binary absent) makes it NOT_RUN. The replay test of each gate is attached to every criterion, except for
I5, whose action-matrix half is decided by held-out belief fixtures and needs no world (``REPLAY["I5"] = []``;
those nodes are listed on I5's three integrated-mission criteria instead).

Usage: python -m uv run python scripts/record_unity_gate_evidence.py [I1 I2 I3 I4 I5 I6 I7] [--no-run]
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
from conrad.sim.mission.unity_faults import fault_cases  # noqa: E402
from conrad.sim.mission.unity_run import git_commit, player_identity  # noqa: E402

I2_FAULT_CASES = tuple(fault_cases()["cases"])  # configs/sim/nav_fault_cases.yaml

T1 = "tests/unity_live/test_i1_unity.py::"
T2 = "tests/unity_live/test_i2_unity.py::"
T3 = "tests/unity_live/test_i3_unity.py::"
T4 = "tests/unity_live/test_i4_unity.py::"
T5 = "tests/unity_live/test_i5_unity.py::"
T6 = "tests/unity_live/test_i6_unity.py::"
T7 = "tests/unity_live/test_i7_unity.py::"
I4_BASES = ("A-B1_fixed_inspection", "A-B0_random", "A-B2_coverage")
# I5's action-matrix half is belief-level fixtures (M1-ACTION-E001), so those six class criteria are decided
# without the player; the module marks them ``no_unity_player``.
I5_MATRIX = ("continue", "request evidence", "replan", "change sensing", "return", "escalate")
MODULES = {
    "I1": "tests/unity_live/test_i1_unity.py",
    "I2": "tests/unity_live/test_i2_unity.py",
    "I3": "tests/unity_live/test_i3_unity.py",
    "I4": "tests/unity_live/test_i4_unity.py",
    "I5": "tests/unity_live/test_i5_unity.py",
    "I6": "tests/unity_live/test_i6_unity.py",
    "I7": "tests/unity_live/test_i7_unity.py",
}
REPLAY = {
    "I1": [T1 + "test_bundle_replays_deterministically"],
    "I2": [
        T2 + "test_bundle_replays_deterministically[NAV-003]",
        T2 + "test_bundle_replays_deterministically[NAV-006]",
    ],
    "I3": [T3 + "test_bundle_replays_deterministically"],
    "I4": [T4 + "test_bundle_replays_deterministically", T4 + "test_no_twin_truth_leakage_on_runtime_side"],
    # I5: empty ON PURPOSE. The replay, the leakage scan and the grid check belong to the Unity missions and
    # are listed on the three mission criteria below; attaching them to every criterion would make the six
    # action-matrix criteria, which are world-independent belief fixtures, depend on the player.
    "I5": [],
    "I6": [
        T6 + "test_bundle_replays_deterministically",
        T6 + "test_no_twin_truth_leakage_on_runtime_side",
        T6 + "test_twin2e_reaches_the_unity_camera",
    ],
    # I7: every criterion also needs the replay, the leakage scan, the sweep declaration (a reduced sweep is
    # evidence only when it is recorded as reduced, with the levels it drops named) and the results file.
    "I7": [
        T7 + "test_bundle_replays_deterministically",
        T7 + "test_no_twin_truth_leakage_on_runtime_side",
        T7 + "test_declared_sweep_is_recorded",
        T7 + "test_results_file_is_written",
    ],
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
        (
            "faults reach defined safe states",
            [T2 + f"test_fault_reaches_defined_safe_state[{c}]" for c in I2_FAULT_CASES],
            [f"fault {c}" for c in I2_FAULT_CASES],
        ),
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
    # I4: 12 held-out worlds x {PRODUCTION, fixed, random, coverage}, sequential Unity flights (ACTIVE-MCBR-E004
    # worlds and budgets); paired bootstrap over worlds, "beats" = CI95 lower bound > 0 (pre-declared).
    "I4": [
        (
            "critical structure partly hidden -> uncertain -> MCBR view -> navigation -> new evidence -> belief improves",
            [T4 + "test_closed_loop_mcbr_view_improves_hidden_target"],
            ["closed loop"],
        ),
        (
            "beats fixed views on actual hidden-state reconstruction",
            [T4 + "test_beats_fixed_views_on_hidden_state_reconstruction"],
            ["beats fixed views"],
        ),
        ("beats random views", [T4 + "test_beats_random_views"], ["beats random views"]),
        ("beats coverage-only", [T4 + "test_beats_coverage_only"], ["beats coverage-only"]),
        (
            "beats simple views on information/time/energy",
            [T4 + f"test_beats_simple_views_on_information_per_time_and_energy[{b}]" for b in I4_BASES],
            [f"information/time/energy vs {b}" for b in I4_BASES],
        ),
    ],
    # I5: both halves of the ch25 formal path. The six action-matrix class criteria and "hard constraints
    # inviolable" read the stored M1-ACTION-E001 final artifact (held-out belief fixtures, no world, no
    # player). The three integrated-mission criteria (ch25 I5 + ch26 Phase 9) fly the eight I5-* scenarios x
    # 3 arms x the 2 worlds declared in configs/eval/i5_unity.yaml before any run; the replay, the leakage
    # scan and the grid-coverage check are attached to each of them.
    "I5": [
        *[(c, [T5 + f"test_action_matrix_class[{c}]"], [c]) for c in I5_MATRIX],
        (
            "hard constraints inviolable",
            [T5 + "test_matrix_hard_constraints_inviolable"],
            ["hard constraints inviolable"],
        ),
        *[
            (
                criterion,
                [
                    T5 + node,
                    T5 + "test_every_scenario_and_arm_flew_on_every_world",
                    T5 + "test_no_twin_truth_leakage_on_runtime_side",
                    T5 + "test_bundle_replays_deterministically",
                ],
                [criterion, "coverage of the declared grid", "leakage"],
            )
            for criterion, node in (
                (
                    "actions exercised correctly inside integrated missions",
                    "test_actions_exercised_correctly_inside_integrated_missions",
                ),
                (
                    "traceable decisions with low measured UIR",
                    "test_traceable_decisions_with_low_measured_uir",
                ),
                (
                    "competitive mission outcomes vs decision baselines",
                    "test_competitive_mission_outcomes_vs_decision_baselines",
                ),
            )
        ],
    ],
    # I6: worlds 7800014-7800016 x {TURBID, CLEAR}, declared in configs/eval/i6_multidomain.yaml before any run; a
    # criterion passes iff it passes on all three worlds (decision rule in that config).
    "I6": [
        (
            "one mission produces 2S, 2T and 2E beliefs",
            [T6 + "test_one_mission_produces_2s_2t_2e_beliefs"],
            ["one mission produces 2S, 2T and 2E beliefs"],
        ),
        (
            "Model1 reasons across all three via the Belief Bus",
            [T6 + "test_model1_reasons_across_all_three_via_the_belief_bus"],
            ["Model1 reasons across all three via the Belief Bus", "twin2e -> unity"],
        ),
        (
            "children remain authoritative within domains",
            [T6 + "test_children_remain_authoritative_within_domains"],
            ["children remain authoritative within domains"],
        ),
    ],
    # I7: worlds 7800018-7800019 x {bandwidth sweep, outage} x the declared bandwidth levels, declared in
    # configs/eval/i7_unity.yaml before any run. One flight measures all five policies (BAAC primary + four
    # shadow arms over the identical offer stream). A criterion passes iff it passes on both worlds (decision
    # rule in that config). CONRAD_I7_SWEEP=reduced selects the declared reduced sweep.
    "I7": [
        (
            "full mission under constrained bandwidth",
            [T7 + "test_full_mission_under_constrained_bandwidth"],
            ["full mission under constrained bandwidth", "sweep"],
        ),
        (
            "full mission under outages",
            [T7 + "test_full_mission_under_outages"],
            ["full mission under outages"],
        ),
        (
            "BAAC retains more mission-relevant information than raw/FIFO/fixed-priority",
            [T7 + "test_baac_retains_more_than_raw_fifo_fixed_priority"],
            ["BAAC retains more mission-relevant information than raw/FIFO/fixed-priority"],
        ),
        (
            "critical latency and sync error compared against baselines",
            [T7 + "test_critical_latency_and_sync_error_compared_against_baselines"],
            ["critical latency and sync error compared against baselines"],
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
    if gate == "I6":
        note += (
            "; worlds and arms declared before any run in configs/eval/i6_multidomain.yaml; per-run results in "
            "artifacts/gates/I6/unity_i6_results.json"
        )
    if gate == "I4":
        note += (
            "; worlds declared before any run in configs/eval/active_mcbr_e004.yaml; per-world results in "
            "artifacts/gates/I4/unity_i4_results.json"
        )
    if gate == "I5":
        note += (
            "; the action-matrix half is M1-ACTION-E001 on held-out BELIEF FIXTURES (no world, no player), "
            "the integrated-mission half is Unity on the worlds declared before any run in "
            "configs/eval/i5_unity.yaml (reduced sweep: 2 worlds, declared as reduced); per-flight results in "
            "artifacts/gates/I5/unity_i5_results.json; the success floor, the 4 s latency budget, the UIR "
            "bound and the contact clearance are ENGINEERING_ESTIMATE values (ch25/ch26 leave the I5 bounds "
            "OPEN); I5-NOMINAL is declared NOT APPLICABLE to the continue criterion (measured: its warrant "
            "cannot arise by construction) and is still scored for everything else"
        )
    if gate == "I7":
        sweep = data["criteria"].get("sweep", {})
        note += (
            "; worlds, arms, bandwidth levels and both sweeps declared before any run in "
            "configs/eval/i7_unity.yaml; per-flight results in artifacts/gates/I7/unity_i7_results.json; "
            f"sweep={json.dumps({k: sweep.get(k) for k in ('name', 'reduced', 'bandwidth_levels', 'outage_levels', 'dropped_bandwidth_levels', 'dropped_closed_loop_levels', 'flights')}, sort_keys=True)}"
            "; all link numbers SYNTHETIC_ONLY; deadlines are ENGINEERING_ESTIMATE"
        )
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
            *((f"artifacts/gates/{gate}/unity_i4_results.json",) if gate == "I4" else ()),
            *(
                ("artifacts/gates/I5/unity_i5_results.json", "configs/eval/i5_unity.yaml")
                if gate == "I5"
                else ()
            ),
            *((f"artifacts/gates/{gate}/unity_i6_results.json",) if gate == "I6" else ()),
            *(
                ("artifacts/gates/I7/unity_i7_results.json", "configs/eval/i7_unity.yaml")
                if gate == "I7"
                else ()
            ),
        ),
        notes=note,
    )
    write_evidence(ev)
    return ev


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        raise SystemExit(0)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    rerun = "--no-run" not in sys.argv
    for g in args or ["I1", "I2", "I3"]:
        e = record(g, rerun)
        print(g, [f"{c.criterion}:{c.status.value}" for c in e.criteria])
    reports = evaluate_gates()
    for g in ("U0", "2S-FIRST", "I1", "I2", "2T", "I3", "I4"):
        r = reports[g]
        print(
            f"{g:<9} official={r.official_status.value} formal={r.formal_status.value} "
            f"blocked_by={','.join(r.blocking_upstream)}"
        )
