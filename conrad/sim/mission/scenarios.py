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
    # I5 iteration 2 (M1-ACTION-E003): the nominal mission with a READABLE intact surface. In I5-NOMINAL a view
    # of the target yields one averaged reading, so Model2T's surface coverage (80 % of cells) can never be
    # reached and the "continue" warrant (critical component OBSERVED INTACT) never arises. Here the target's
    # non-defect surface is read per tile (8 axial x 8 around), so a view covers every tile it sees, and it
    # starts without corrosion or crack (in I5-NOMINAL it is sampled from the priors, about 1 mm corrosion).
    "I5-NOMINAL-READABLE": {
        "world": {
            "defect": {
                "side": "near",
                "corrosion_depth_m": 0.0,
                "crack_length_m": 0.0,
                "pristine_rest": True,
            },
            "structural": {"region_tiles": [8, 8]},
        },
        # Same runtime as I5-NOMINAL. A DEV trial with 240 s and 10 plans per need did not raise coverage to the
        # 80 % Model2T needs and added dead-end escalations, so it was dropped (docs/audits/I5_ACTION_MATRIX.md).
        "runtime": {"duration_s": 120.0},
    },
}

# Gate I6 (docs/audits/I6_MULTI_DOMAIN.md, configs/eval/i6_multidomain.yaml): one world, one mission, 2S + 2T + 2E
# beliefs; Model 1 cites all three. TURBID: a turbidity spike (Twin2E truth) during the lane survey, so the
# critical inspection that EGDC requests after the survey faces low visibility. CLEAR: the same world without the
# spike (paired control: the 2E belief, not the world geometry, changes the outcome). The gate settings are the
# deployment's ENGINEERING_ESTIMATE values, fixed on development worlds before any final run.
I6_RUNTIME: dict[str, Any] = {
    "duration_s": 120.0,
    "multidomain": {"enabled": True, "inspection_range_m": 1.0, "min_visibility": 0.5},
}
I6_SPIKE: dict[str, Any] = {
    "t_s": 15.0,
    "event_type": "TURBIDITY_SPIKE",
    "parameters": {"delta_ntu": 25.0, "units": "NTU", "center_m": [0.0, 0.0, 1.0], "radius_m": 30.0},
}
SCENARIOS["I6-MULTIDOMAIN-TURBID"] = {"world": {"eco_events": [I6_SPIKE]}, "runtime": dict(I6_RUNTIME)}
SCENARIOS["I6-MULTIDOMAIN-CLEAR"] = {"world": {}, "runtime": dict(I6_RUNTIME)}

# Flagship integrated Unity mission (docs/audits/FLAGSHIP_UNITY.md, configs/eval/flagship_unity.yaml).
# ONE mission on ONE held-out world carrying all three declared stressors at declared times. It is a
# demonstration run, not a gate: no criterion is derived from it. Every time below is a SYNTHETIC_ONLY
# simulation setting, fixed on development worlds (unity_gate development split) before the final world ran.
#
#  1. communication outage  link down [6 s, 70 s); the defect is on the LANE side (as in I7-OUTAGE-CRITICAL),
#     so the critical structural finding is made by the lane pass itself while the link is down and can only
#     be delivered after reconnection.
#  2. contradiction         the second structural payload starts at 16 s, in the middle of the lane view of the
#     target, so the first readings are single-sensor and every later view yields TWO readings of the same
#     component that disagree. Unlike INT-003 the second payload is NOT corrupted (corruption 0), so Model2T
#     rates it reliable and the disagreement is credible: it carries a persistent +10 mm wall-loss calibration
#     offset and a -0.8 surface-appearance offset. A credible disagreement must raise U_C rather than be
#     averaged away (spec ch33 uncertainty rule 5).
#  3. localization          the synthetic USBL-like position fix stops at 90 s and never returns, driving the
#     estimator to DEGRADED and then LOCALIZATION_LOST with the declared safe state.
FLAGSHIP_UNITY_DURATION_S = 150.0
FLAGSHIP_UNITY_LINK_OUTAGE_S = (6.0, 70.0)
FLAGSHIP_UNITY_CONTRADICTION_S = 16.0
FLAGSHIP_UNITY_CONTRADICTION_BIAS_M = 0.010
FLAGSHIP_UNITY_FIX_OUTAGE_S = 90.0
FLAGSHIP_UNITY_FIX_OUTAGE_DURATION_S = 90.0  # past the end of the mission: the fix never returns
SCENARIOS["FLAGSHIP-UNITY"] = {
    "world": {
        "defect": {"side": "near"},
        "contradiction": {
            "enabled": True,
            "start_s": FLAGSHIP_UNITY_CONTRADICTION_S,
            "sensor_bias_m": FLAGSHIP_UNITY_CONTRADICTION_BIAS_M,
            "corruption": 0.0,
            "contradiction": -0.8,
        },
        "faults": [
            {
                "t_s": FLAGSHIP_UNITY_FIX_OUTAGE_S,
                "type": "FIX_OUTAGE",
                "duration_s": FLAGSHIP_UNITY_FIX_OUTAGE_DURATION_S,
            }
        ],
    },
    "runtime": {
        "duration_s": FLAGSHIP_UNITY_DURATION_S,
        "link": {"outages_s": [list(FLAGSHIP_UNITY_LINK_OUTAGE_S)]},
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
