"""Registered integrated scenario IDs (gates I1-I7, flagship I4, INT-001..INT-010). TRUTH-SIDE harness.

Each entry is a pair of overrides (``world`` for MissionWorldOptions, ``runtime`` for MissionRuntimeConfig)
applied on top of the run configuration's ``sim.mission`` section. Scenario overrides win: they define what
the scenario IS (e.g. the fixed-view baseline swaps the planner), while configs size it (duration, grids).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from typing import Any

Override = dict[str, dict[str, Any]]
FIXED_VIEW = "A-B1_fixed_inspection"
ON_INSPECTION = "INSPECTION_GOAL"

SCENARIOS: dict[str, Override] = {
    "GOLDEN-SMOKE": {"world": {}, "runtime": {"duration_s": 24.0}},
    "I1-SPATIAL": {"world": {"ecological_enabled": False}, "runtime": {"model2e_enabled": False}},
    "I3-STRUCTURAL": {"world": {}, "runtime": {}},
    "FLAGSHIP-I4": {"world": {}, "runtime": {}},
    "FLAGSHIP-I4-FIXEDVIEW": {"world": {}, "runtime": {"planner": FIXED_VIEW}},
    "I6-MULTIDOMAIN": {
        "world": {
            "eco_events": [
                {
                    "t_s": 15.0,
                    "event_type": "TURBIDITY_SPIKE",
                    "parameters": {
                        "delta_ntu": 12.0,
                        "units": "NTU",
                        "center_m": [0.0, 0.0, 1.0],
                        "radius_m": 30.0,
                    },
                }
            ]
        },
        "runtime": {},
    },
    "I7-COMMS-OUTAGE": {
        "world": {},
        "runtime": {"link": {"bandwidth_bps": 300.0, "outages_s": [[20.0, 70.0]]}},
    },
    # Gate I7 harness (COM-I7-E001/E002). The defect is on the lane side, so the critical structural finding is
    # made by the lane pass itself (not by MCBR); I7-OUTAGE-CRITICAL drops the link at 6 s, before the robot
    # reaches a view of the defect, and restores it at 60 s. The link stays the declared mission link.
    "I7-BANDWIDTH": {"world": {"defect": {"side": "near"}}, "runtime": {"duration_s": 240.0}},
    "I7-OUTAGE-CRITICAL": {
        "world": {"defect": {"side": "near"}},
        "runtime": {"duration_s": 240.0, "link": {"outages_s": [[6.0, 60.0]]}},
    },
    # ch20 "Integrated intelligence benchmarks" (spec lines ~9681-9703): one-line definitions, no criteria.
    "INT-001": {"world": {"defect": {"side": "near"}}, "runtime": {}},
    "INT-002": {"world": {}, "runtime": {}},
    "INT-003": {"world": {"contradiction": {"enabled": True}}, "runtime": {}},
    "INT-004": {
        "world": {
            "structural": {"degradation": {"corruption": 0.5}},
            "geometric_degradation": {"turbidity": 0.5, "range_noise_scale": 3.0},
            "geometric_dropout": [[10.0, 25.0]],
        },
        "runtime": {},
    },
    "INT-005": {"world": {}, "runtime": {"link": {"bandwidth_bps": 100.0, "packet_loss": 0.15}}},
    "INT-006": {"world": {"family": "bent_pipeline", "survey_sigma_m": 0.3}, "runtime": {}},
    "INT-007": {
        "world": {
            "eco_events": [
                {
                    "t_s": 15.0,
                    "event_type": "TURBIDITY_SPIKE",
                    "parameters": {
                        "delta_ntu": 15.0,
                        "units": "NTU",
                        "center_m": [0.0, 0.0, 1.0],
                        "radius_m": 30.0,
                    },
                }
            ]
        },
        "runtime": {},
    },
    "INT-008": {
        "world": {
            "faults": [{"t_s": 3.0, "trigger": ON_INSPECTION, "type": "THRUSTER_FAILURE", "target": "H1"}]
        },
        "runtime": {},
    },
    "INT-009": {
        "world": {
            "faults": [
                {"t_s": 1.0, "trigger": ON_INSPECTION, "type": "FIX_OUTAGE", "duration_s": 40.0},
                {
                    "t_s": 1.0,
                    "trigger": ON_INSPECTION,
                    "type": "SENSOR_NOISE",
                    "target": "imu",
                    "magnitude": 5.0,
                },
            ]
        },
        "runtime": {},
    },
    "INT-010": {"world": {}, "runtime": {"link": {"bandwidth_bps": 300.0, "outages_s": [[25.0, 85.0]]}}},
    # Gate I5 integrated missions (M1-ACTION-E002): each scenario makes one Model 1 action class warranted.
    # The trigger is scripted on the truth side; the runtime only sees its consequences through sensors,
    # power readings, the link state and its own mission clock.
    "I5-NOMINAL": {
        "world": {"defect": {"side": "near", "corrosion_depth_m": 0.0, "crack_length_m": 0.0}},
        "runtime": {"duration_s": 120.0},
    },
    "I5-CRITICAL-FINDING": {"world": {"defect": {"side": "near"}}, "runtime": {"duration_s": 120.0}},
    "I5-UNCERTAIN-BELIEF": {"world": {}, "runtime": {"duration_s": 120.0}},
    "I5-ROUTE-BLOCKED": {"world": {"lane_obstacle": {}}, "runtime": {"duration_s": 120.0}},
    "I5-BATTERY-RESERVE": {
        "world": {"faults": [{"t_s": 30.0, "type": "LOW_POWER", "magnitude": 0.12}]},
        "runtime": {"duration_s": 120.0},
    },
    "I5-TIME-RESERVE": {"world": {}, "runtime": {"duration_s": 120.0, "time_budget_s": 150.0}},
    "I5-COMMS-OUTAGE": {
        "world": {"defect": {"side": "near"}},
        "runtime": {"duration_s": 120.0, "link": {"outages_s": [[6.0, 80.0]]}},
    },
}

INT_DESCRIPTIONS = {
    "INT-001": "visible infrastructure (defect on the lane side)",
    "INT-002": "occluded critical region (defect on the far side)",
    "INT-003": "contradictory evidence (second degraded, biased structural sensor)",
    "INT-004": "sensor degradation (structural corruption, turbid sonar/depth, geometric dropout 10-25 s)",
    "INT-005": "communication constrained (100 bps, 15 % packet loss)",
    "INT-006": "unknown geometry (bent pipeline, 0.3 m registry survey error)",
    "INT-007": "multi-domain anomaly (turbidity spike at 15 s affects 2E fields and 2T inspection quality)",
    "INT-008": "thruster H1 failure 3 s after the first inspection goal",
    "INT-009": "position-fix outage + IMU noise 1 s after the first inspection goal (MCBR manoeuvre)",
    "INT-010": "critical finding under a 25-85 s acoustic link outage",
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve(scenario_id: str, mission_section: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"unknown scenario {scenario_id!r}; known: {sorted(SCENARIOS)}")
    over = SCENARIOS[scenario_id]
    world = _merge(dict(mission_section.get("world", {})), over.get("world", {}))
    runtime = _merge(dict(mission_section.get("runtime", {})), over.get("runtime", {}))
    return world, runtime
