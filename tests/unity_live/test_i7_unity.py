"""Gate I7 FORMAL path: the frozen I7 communication harness flown through the built Unity player.

Worlds, arms, bandwidth levels, the sweep declarations and the per-criterion decision rule are declared in
``configs/eval/i7_unity.yaml`` BEFORE any run: unity_gate final_test worlds 7800018 and 7800019 (the last two
unspent seeds of that digest-pinned split), 240 s missions at a 0.1 s Unity control period, and one flight per
(world, scenario, bandwidth level). One flight measures all five policies: BAAC is the primary arm and drives
the mission, raw/send-all, FIFO, fixed priority and value-per-bit are shadow arms of the same ``ShoreLink``
that receive the identical offer stream at identical times over a link with the identical profile and channel
seed and never feed back (``conrad/orchestration/comms.py``). Nothing here is tuned on a final world: the BAAC
design was frozen on mission development seed 5100000 during the surrogate pass.

Two sweeps are declared. ``full`` is the surrogate's levels (100/50/10/1/0.1/0 % plus the outage scenario at
100 % and 10 % and the closed-loop baseline primaries). ``reduced`` is the declared fallback for when Unity
wall clock is short; it keeps every ARM and every qualitative regime and drops only levels the surrogate
showed to be bracketed by a kept level. Pick it with ``CONRAD_I7_SWEEP=reduced``; the recorded evidence always
says which sweep ran and whether it was reduced.

Flights are strictly sequential: one Unity player at a time.
"""

from __future__ import annotations

import faulthandler
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from unity_gate_support import GATE_RUNS, GATES, leakage_scan, measured, reset_measured

from conrad.evaluation.decision_experiments import com_i7_unity as H
from conrad.settings import REPO_ROOT
from conrad.sim.mission.unity_run import player_identity, replay_unity_run
from conrad.sim.unity.player import find_player

GATE = "I7"
RESULTS = GATES / GATE / "unity_i7_results.json"
SWEEP_ENV = "CONRAD_I7_SWEEP"
# Wall-clock ceiling per Unity flight (ENGINEERING_ESTIMATE): a 240 s mission at a 0.1 s control period is
# 2400 lock-step player round trips plus the five-arm python mission. Measured reference on this machine:
# the gate I4 Unity flights ran 1000 control steps in 83 s each, and a kernel I7 mission of 2400 steps with
# five arms takes minutes. The ceiling is deliberately generous; it only guards against a hung bridge.
FLIGHT_CEILING_S = 1800.0
REPLAY_CEILING_S = 2400.0


def _player_has_ecology() -> bool:
    """The I7 mission runs the full 2S + 2T + 2E stack, so it needs the Twin2E player (EcologyScene.cs).

    Guards the held-out worlds: a stale player would be launched on a final world only to refuse the
    ``optics_grid`` primitive (same guard as gate I6)."""
    exe = find_player()
    if exe is None:
        return False
    dll = exe.parent / f"{exe.stem}_Data" / "Managed" / "ConradUnityV2.dll"
    if not dll.is_file():
        return False
    data = dll.read_bytes()  # C# string literals live as UTF-16 in the assembly's user-string heap
    return "optics_grid".encode("utf-16-le") in data or b"optics_grid" in data


pytestmark = pytest.mark.skipif(
    not _player_has_ecology(),
    reason="Unity player absent or built before EcologyScene.cs (no optics_grid support): rebuild it with "
    "BuildScript.BuildWindows64Player before recording I7",
)


def _config() -> dict[str, Any]:
    return H.load_config()


def _sweep(cfg: dict[str, Any]) -> dict[str, Any]:
    return H.sweep_of(cfg, os.environ.get(SWEEP_ENV) or None)


@pytest.fixture(autouse=True)
def _hard_timeout() -> Iterator[None]:  # overrides the conftest guard for this module only
    cfg = _config()
    jobs = len(H.flight_jobs(cfg, H.final_worlds(cfg), _sweep(cfg), H.UNITY))
    faulthandler.dump_traceback_later(jobs * FLIGHT_CEILING_S + REPLAY_CEILING_S, exit=True)
    yield
    faulthandler.cancel_dump_traceback_later()


@pytest.fixture(scope="module")
def flights() -> dict[str, Any]:
    cfg = _config()
    sweep = _sweep(cfg)
    worlds = H.final_worlds(cfg)
    result = H.run_gate(cfg, worlds, sweep, H.UNITY, GATE_RUNS / GATE, results_path=RESULTS)
    reset_measured(
        GATE,
        {
            "worlds": worlds,
            "arms": result["arms"],
            "sweep": sweep["name"],
            "reduced_sweep": result["reduced_sweep"],
            "dropped_bandwidth_levels": sweep.get("dropped_bandwidth_levels", []),
            "dropped_closed_loop_levels": sweep.get("dropped_closed_loop_levels", []),
            "bandwidth_levels": sweep["bandwidth_levels"],
            "outage_levels": sweep["outage_levels"],
            "flights": result["flights"],
            "wall_clock_s": result["wall_clock_s"],
            "partition": "configs/eval/partitions_unity_gates.yaml final_test (declared in i7_unity.yaml)",
            "results": str(RESULTS.relative_to(REPO_ROOT)),
        },
    )
    return result


def _per_world(flights: dict[str, Any], criterion: str) -> dict[str, bool]:
    return {w: bool(m[criterion]["ok"]) for w, m in flights["per_world"].items()}


def _summary(flights: dict[str, Any], criterion: str) -> dict[str, Any]:
    return {w: m[criterion] for w, m in flights["per_world"].items()}


def test_full_mission_under_constrained_bandwidth(flights):
    per = _per_world(flights, H.C_BANDWIDTH)
    measured(GATE, H.C_BANDWIDTH, per_world=per, detail=_summary(flights, H.C_BANDWIDTH))
    assert all(per.values()), per


def test_full_mission_under_outages(flights):
    per = _per_world(flights, H.C_OUTAGE)
    measured(GATE, H.C_OUTAGE, per_world=per, detail=_summary(flights, H.C_OUTAGE))
    assert all(per.values()), per


def test_baac_retains_more_than_raw_fifo_fixed_priority(flights):
    per = _per_world(flights, H.C_RETAIN)
    measured(GATE, H.C_RETAIN, per_world=per, detail=_summary(flights, H.C_RETAIN))
    assert all(per.values()), per


def test_critical_latency_and_sync_error_compared_against_baselines(flights):
    per = _per_world(flights, H.C_LATENCY)
    measured(GATE, H.C_LATENCY, per_world=per, detail=_summary(flights, H.C_LATENCY))
    assert all(per.values()), per


def test_declared_sweep_is_recorded(flights):
    """A reduced sweep is evidence only when it is declared as reduced, with its dropped levels named."""
    sweep = flights["sweep"]
    measured(
        GATE,
        "sweep",
        name=sweep["name"],
        reduced=flights["reduced_sweep"],
        bandwidth_levels=sweep["bandwidth_levels"],
        outage_levels=sweep["outage_levels"],
        closed_loop_levels=sweep.get("closed_loop_levels", []),
        dropped_bandwidth_levels=sweep.get("dropped_bandwidth_levels", []),
        dropped_closed_loop_levels=sweep.get("dropped_closed_loop_levels", []),
        flights=flights["flights"],
        wall_clock_s=flights["wall_clock_s"],
        player=player_identity(),
    )
    assert flights["flights"] == len(H.flight_jobs(_config(), flights["worlds"], sweep, H.UNITY)), (
        "every declared flight must have run"
    )
    assert set(flights["arms"]) == set(H.POLICIES), "no baseline arm may be dropped from a flight"
    if flights["reduced_sweep"]:
        assert sweep.get("dropped_bandwidth_levels"), "a reduced sweep must name the levels it drops"


def test_no_twin_truth_leakage_on_runtime_side(flights):
    rows, bad = {}, []
    for r in flights["runs"]:
        run_dir = REPO_ROOT / r["run_dir"]
        meta = json.loads((run_dir / "truth" / "truth_record.json").read_text(encoding="utf-8"))["meta"]
        scanned, violations = leakage_scan(run_dir, set(meta["world_entity_ids"]), set())
        rows[r["run_id"]] = {"texts_scanned": scanned, "violations": violations[:10]}
        bad += violations
        assert scanned > 100
    measured(GATE, "leakage", runs=rows)
    assert not bad


def test_bundle_replays_deterministically(flights):
    """The outage flight at the highest declared level: the bundle that carries the reconnection trace."""
    outage = [r for r in flights["runs"] if r["role"] == H.OUTAGE and r["primary"] == "baac"]
    r = max(outage, key=lambda x: float(x["bandwidth_factor"]))
    rep = replay_unity_run(REPO_ROOT / r["run_dir"], GATE_RUNS / "replay" / GATE)
    measured(
        GATE,
        "replay",
        run=r["run_dir"],
        **{
            k: rep[k]
            for k in (
                "events",
                "decisions",
                "revisions",
                "trajectory",
                "equal",
                "verified_files",
                "verified_objects",
                "same_player_binary",
            )
            if k in rep
        },
    )
    assert rep["equal"], rep


def test_results_file_is_written(flights):
    assert Path(RESULTS).is_file()
    stored = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert stored["backend"] == H.UNITY and stored["evidence_class"].startswith("FORMAL")
    assert stored["worlds"] == flights["worlds"]
