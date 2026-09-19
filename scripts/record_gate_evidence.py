"""Record formal gate evidence for the gates whose criteria are decided by executed tests/experiments.

Each criterion maps to pytest node IDs (all must pass) and/or an experiment check reading a retained artifact.
Gates that need Unity or integrated missions (U0, I1-I7) are recorded by their own harnesses, not here.

Usage: python -m uv run python scripts/record_gate_evidence.py [P0 I0 C1 2S-FIRST 2T 2E]
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conrad.evaluation.gates import (  # noqa: E402
    CriterionResult,
    CriterionStatus,
    EvidenceClass,
    GateEvidence,
    write_evidence,
)

P0 = "tests/contract/test_p0_contracts.py::"
I0 = "tests/integration/test_i0_fake_full_system.py::"
CC = "tests/contract/test_cc_persistence.py::"
UC = "tests/unit/core/"
SP = "tests/unit/domains/spatial/"


def _exp(path: str, check: Callable[[dict], tuple[bool, str]]) -> Callable[[], tuple[CriterionStatus, str]]:
    def run() -> tuple[CriterionStatus, str]:
        p = ROOT / path
        if not p.exists():
            return CriterionStatus.NOT_RUN, f"artifact missing: {path}"
        ok, measured = check(json.loads(p.read_text(encoding="utf-8")))
        return (CriterionStatus.PASS if ok else CriterionStatus.FAIL), measured

    return run


def _persist_check(d: dict) -> tuple[bool, str]:
    """Main PMBL arm over long sequences: identity, re-identification, provenance, dedupe, CC-09 all at 1.0."""
    s = d.get("summary", d)
    keys = [
        "pmbl_full.identity_purity",
        "pmbl_full.reidentification_after_gap",
        "pmbl_full.provenance_integrity",
        "pmbl_full.duplicate_rejection_rate",
        "cc09_reset_equivalence",
    ]
    vals = {k: float(s[k]["mean"]) for k in keys if k in s}
    ok = len(vals) == len(keys) and all(v >= 0.999 for v in vals.values())
    return ok, json.dumps(vals) if vals else "keys missing"


I5A = "tests/acceptance/test_i5_action_matrix.py::"
I5_ARTIFACT = "artifacts/experiments/M1-ACTION-E001/m1_action_e001.json"


def _i5_class_check(cls: str) -> Callable[[dict], tuple[bool, str]]:
    """M1-ACTION-E001 final split: the class's canonical-scenario recall meets the stored floor (EGDC arm)."""

    def check(d: dict) -> tuple[bool, str]:
        if d.get("partition") != "final_test":
            return False, f"artifact partition is {d.get('partition')!r}, not final_test"
        s = d["summary"]["egdc_structured"]["per_class"].get(cls, {})
        floor = float(d["acceptance"]["canonical_recall_floor"])
        recall, n = s.get("recall"), int(s.get("n_canonical", 0))
        ok = recall is not None and n >= 50 and recall >= floor
        return ok, f"{cls}: recall={recall} n={n} floor={floor} (ENGINEERING_ESTIMATE; spec bound OPEN)"

    return check


def _i5_constraints_check(d: dict) -> tuple[bool, str]:
    s = d["summary"]["egdc_structured"]
    hc = s["hard_constraint_violations"]
    uir = s["uir"]["unsupported_inference_rate"]
    ok = d.get("partition") == "final_test" and hc["total"] == 0 and uir == 0.0
    return ok, (
        f"violations={hc['total']} (audit={hc['chosen_action_audit']}, injected accepted="
        f"{hc['injected_invalid_accepted']}/{hc['injected_invalid_proposals']}); UIR={uir}"
    )


I5M = "tests/acceptance/test_i5_integrated_missions.py::"
I5_E002 = "artifacts/experiments/M1-ACTION-E002/m1_action_e002.json"
I5_MISSION_CRITERIA = (
    "actions exercised correctly inside integrated missions",
    "traceable decisions with low measured UIR",
    "competitive mission outcomes vs decision baselines",
)


def _i5_formal_mission(criterion: str) -> Callable[[], tuple[CriterionStatus, str]]:
    """The formal path is integrated missions through Unity. Only python-kernel SURROGATE runs exist, so the
    formal criterion stays NOT_RUN; the surrogate result lives in artifacts/gates/I5/evidence_surrogate.json."""

    def run() -> tuple[CriterionStatus, str]:
        return CriterionStatus.NOT_RUN, (
            f"{criterion}: no Unity integrated-mission run exists; M1-ACTION-E002 is SURROGATE (python L1 kernel) "
            "and is recorded only with --surrogate"
        )

    return run


def _i5_e002_final(d: dict) -> str | None:
    if d.get("partition") != "final_test":
        return f"artifact partition is {d.get('partition')!r}, not final_test"
    if "SURROGATE" not in str(d.get("evidence_class", "")):
        return "artifact is not labelled SURROGATE"
    return None


def _i5_actions_check(d: dict) -> tuple[bool, str]:
    bad = _i5_e002_final(d)
    if bad:
        return False, bad
    v = d["verdicts"]
    e = d["closed_loop"]["egdc_structured"]
    rows = "; ".join(
        f"{sc}: correct {e[sc]['correct']}/{e[sc]['n']} latency_max={e[sc]['latency_s_max']} "
        f"forbidden={e[sc]['forbidden_after_onset']}"
        for sc in d["scenarios"]
    )
    return bool(v["actions_exercised_correctly"]), (
        f"{rows}; violations={v['violations_total']}; nominal over-escalations={v['nominal_over_escalations']}; "
        f"floor={v['success_floor']} (ENGINEERING_ESTIMATE)"
    )


def _i5_trace_check(d: dict) -> tuple[bool, str]:
    bad = _i5_e002_final(d)
    if bad:
        return False, bad
    v = d["verdicts"]
    u = d["closed_loop"]["egdc_structured"]["ALL"]["uir"]
    return bool(v["traceable_low_uir"]), (
        f"traceable_fraction={v['traceable_fraction']}; UIR={v['uir']} ({u['relied_unsupported_claims']}/"
        f"{u['relied_world_claims']} relied world claims) max={v['uir_max']}"
    )


def _i5_competitive_check(d: dict) -> tuple[bool, str]:
    bad = _i5_e002_final(d)
    if bad:
        return False, bad
    v = d["verdicts"]
    rows = "; ".join(
        f"vs {k}: task success {c['egdc_task_success']} vs {c['baseline_task_success']}, safety events "
        f"{c['egdc_safety_events']} vs {c['baseline_safety_events']}"
        for k, c in v["competitive"].items()
    )
    return bool(v["competitive_outcomes"]), rows


def _walk(o: object, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    if isinstance(o, dict):
        for k, v in o.items():
            out.append((f"{prefix}{k}", v))
            out += _walk(v, f"{prefix}{k}.")
    return out


PLAN: dict[str, list[tuple[str, list[str], Callable[[], tuple[CriterionStatus, str]] | None]]] = {
    "P0": [
        (
            "serialization/deserialization",
            [P0 + "test_round_trip_serialization", P0 + "test_unknown_keys_rejected"],
            None,
        ),
        ("schema version", [P0 + "test_schema_version_gate"], None),
        ("timestamps", [P0 + "test_time_semantics", P0 + "test_evidence_cannot_predate_observation"], None),
        (
            "coordinate frames",
            [
                P0 + "test_frame_contract",
                P0 + "test_frame_graph_round_trip",
                P0 + "test_every_spatial_type_requires_a_frame",
            ],
            None,
        ),
        (
            "UUID uniqueness",
            [P0 + "test_uuid_uniqueness_and_determinism", P0 + "test_namespaced_id_factories_never_collide"],
            None,
        ),
        ("provenance graph validity", [P0 + "test_provenance_dag"], None),
        (
            "uncertainty contract",
            [P0 + "test_uncertainty_contract", P0 + "test_unknown_is_a_status_not_a_value"],
            None,
        ),
        ("RobotConfig validation", [P0 + "test_robot_config_validation"], None),
    ],
    "I0": [
        ("complete architectural loop executes", [I0 + "test_i0_loop_closes_and_belief_resolves"], None),
        ("timestamps", [I0 + "test_i0_boundaries_preserve_identity"], None),
        ("IDs", [I0 + "test_i0_boundaries_preserve_identity"], None),
        ("frames", [I0 + "test_i0_boundaries_preserve_identity"], None),
        ("provenance", [I0 + "test_i0_full_causal_trace_from_command_to_raw_observation"], None),
        ("uncertainty", [I0 + "test_i0_loop_closes_and_belief_resolves"], None),
        ("logging", [I0 + "test_i0_deterministic_replay"], None),
        ("configs", [I0 + "test_i0_boundaries_preserve_identity"], None),
    ],
    "C1": [
        (
            "persistent identity",
            [
                UC + "test_core_pmbl.py::test_lifecycle_candidate_confirm_active_dormant_reactivated",
                UC + "test_core_pmbl.py::test_cc09_reset_working_memory_preserves_persistence",
            ],
            None,
        ),
        (
            "evidence association",
            [
                UC + "test_core_association.py::test_retrieval_gates_frame_lifecycle_domain_distance_and_k",
                UC + "test_core_association.py::test_nearest_neighbour_baseline_and_engine",
            ],
            None,
        ),
        (
            "no-match/new entity",
            [
                UC + "test_core_association.py::test_no_match_with_empty_candidate_set",
                UC + "test_core_pipeline.py::test_no_match_creates_candidate_then_matches",
            ],
            None,
        ),
        (
            "credible contradiction handling",
            [
                UC
                + "test_core_buo_analytic.py::test_credible_contradiction_raises_uc_and_is_not_averaged_away",
                UC + "test_core_buo_analytic.py::test_unreliable_disagreement_is_ambiguous_not_contradiction",
            ],
            None,
        ),
        (
            "decomposed uncertainty",
            [
                UC + "test_core_buo_analytic.py::test_corrupted_evidence_raises_ua",
                UC + "test_core_buo_analytic.py::test_ood_score_raises_ue_and_unmeasured_ood_keeps_it",
                UC + "test_core_buo_analytic.py::test_low_coverage_raises_uo",
            ],
            None,
        ),
        (
            "relational inference",
            [
                UC
                + "test_core_analytic_tbd_rbp.py::test_analytic_rbp_labels_inferred_and_keeps_observational_uncertainty",
                UC + "test_core_analytic_tbd_rbp.py::test_analytic_rbp_never_overwrites_direct_observation",
                UC + "test_core_pipeline.py::test_relational_step_is_inferred_with_provenance",
            ],
            None,
        ),
        (
            "temporal prediction/correction",
            [
                UC + "test_core_analytic_tbd_rbp.py::test_analytic_tbd_uses_delta_t_and_marks_predicted",
                UC + "test_core_pipeline.py::test_prediction_is_persisted_as_predicted_and_kept_apart",
                UC + "test_core_pipeline.py::test_sequence_index_is_irrelevant_only_physical_time_matters",
            ],
            None,
        ),
        (
            "provenance",
            [
                CC + "test_cc04_merge_preserves_lineage_to_raw_observations",
                CC + "test_cc03_dangling_provenance_rejected",
                UC + "test_core_pmbl.py::test_cc04_merge_preserves_lineage_evidence_and_provenance",
            ],
            None,
        ),
        (
            "long sequence persistence",
            [
                CC + "test_cc01_duplicate_delivery_is_one_contribution",
                CC + "test_cc02_late_observation_never_masquerades_as_current",
            ],
            _exp("artifacts/experiments/core/CORE-PERSIST-E001.json", _persist_check),
        ),
    ],
    "2S-FIRST": [
        (
            "observed != inferred != unknown",
            [
                SP + "test_m2s_inference_temporal.py::test_small_gap_is_inferred_with_relational_provenance",
                SP + "test_m2s_inference_temporal.py::test_large_gap_and_hidden_side_stay_unknown",
                SP + "test_m2s_map_updates.py::test_unknown_is_never_occupied_and_is_free_follows_policy",
            ],
            None,
        ),
        (
            "counterfactual worlds produce uncertainty, not confident hallucination",
            [
                "tests/simulation/spatial/test_m2s_twin2s_loop.py::test_counterfactual_region_is_not_confidently_resolved"
            ],
            None,
        ),
    ],
    # Model1 action-matrix half of the I5 formal path (M1-ACTION-E001, final split). The integrated-mission half
    # is not covered here; the official status stays BLOCKED_UPSTREAM until I4 passes.
    "I5": [
        (name, [I5A + test], _exp(I5_ARTIFACT, _i5_class_check(name)))
        for name, test in (
            ("continue", "test_i5_continue"),
            ("request evidence", "test_i5_request_evidence"),
            ("replan", "test_i5_replan"),
            ("change sensing", "test_i5_change_sensing"),
            ("return", "test_i5_return"),
            ("escalate", "test_i5_escalate"),
        )
    ]
    + [
        (
            "hard constraints inviolable",
            [
                I5A + "test_i5_hard_constraints_inviolable",
                I5A + "test_artifact_is_final_split_and_large_enough",
                "tests/unit/decision/test_action_semantics.py::test_every_injected_invalid_kind_is_rejected",
            ],
            _exp(I5_ARTIFACT, _i5_constraints_check),
        )
    ]
    # Integrated-mission half (ch25 I5 + ch26 Phase 9): formal needs Unity missions, so these stay NOT_RUN here.
    # M1-ACTION-E002 (python kernel) is recorded in SURROGATE_PLAN["I5"].
    + [(name, [], _i5_formal_mission(name)) for name in I5_MISSION_CRITERIA],
}

# ---------------------------------------------------------------- 2E (functional) and 2E-CEFD (research)
EM = "tests/unit/domains/ecological/test_eco_model2e.py::"
EB = "tests/unit/domains/ecological/test_eco_field_eb.py::"
EP = "tests/property/domains/ecological/test_eco_belief_properties.py::"
ECO = "artifacts/experiments/ecological/"


def _eco_entity_check(d: dict) -> tuple[bool, str]:
    """2E-E002-R3 (thresholds as set for R2) production arm: cover error well below the prior sd in clear/moderate water, calibrated
    (coverage95 >= 0.85, mean z^2 <= 2) at every turbidity level. Thresholds fixed before the FINAL run."""
    s, cand = d["summary"], d.get("candidate", "production")
    levels = sorted({k.split(".")[0] for k in s if k.startswith("turbidity_")}, key=lambda x: float(x[10:]))
    got, ok = {}, bool(levels)
    for lv in levels:
        rmse = s[f"{lv}.{cand}.cover.rmse"]["mean"]
        cov = s[f"{lv}.{cand}.cover.coverage95"]["mean"]
        z2 = s[f"{lv}.{cand}.cover.mean_z2"]["mean"]
        got[lv] = {"rmse": round(rmse, 4), "coverage95": round(cov, 3), "mean_z2": round(z2, 3)}
        ok &= cov >= 0.85 and z2 <= 2.0 and (float(lv[10:]) > 4.0 or rmse <= 0.10)
    return ok, json.dumps({"candidate": cand, **got})


def _eco_field_check(d: dict) -> tuple[bool, str]:
    """2E-E001-R3 (thresholds as set for R2) production arm, per field: RMSE not significantly increasing with sensor count (paired 95 % CI
    low <= 0 at each step) and lower at 8 than at 1 sensor; not significantly worse than the static field at
    >= 4 sensors; calibrated at every count (coverage95 >= 0.85 and 1/3 <= mean z^2 <= 3)."""
    s, p = d["summary"], d.get("paired", {})
    cand = p.get("candidate", d.get("candidate"))
    ok, got = bool(p), {}
    for name in ("temperature", "turbidity"):
        f = p.get(name, {})
        steps = {k: v for k, v in f.items() if "_minus_k" in k}
        rm = {k: s[f"{cand}.k{k}.{name}.rmse"]["mean"] for k in (1, 2, 4, 8)}
        mono = bool(steps) and all(v["ci_low"] <= 0 for v in steps.values()) and rm[8] < rm[1]
        keys = [f"k{k}_minus_static" for k in (4, 8)]
        static = all(k in f and f[k]["ci_low"] <= 0 for k in keys)
        cal = {
            k: (s[f"{cand}.k{k}.{name}.coverage95"]["mean"], s[f"{cand}.k{k}.{name}.mean_z2"]["mean"])
            for k in rm
        }
        calib = all(c >= 0.85 and 1 / 3 <= z <= 3 for c, z in cal.values())
        ok &= mono and static and calib
        got[name] = {
            "rmse": {k: round(v, 4) for k, v in rm.items()},
            "non_increasing": mono,
            "not_worse_than_static_k>=4": static,
            "calibrated": calib,
            "coverage95/mean_z2": {k: [round(c, 3), round(z, 3)] for k, (c, z) in cal.items()},
        }
    return ok, json.dumps(got)


def _cefd_check(d: dict) -> tuple[bool, str]:
    """2E-E003-R3 (thresholds as set for R2): CEFD cover benefit over BOTH uncoupled arms > 0 with the pooled paired 95 % CI above 0, AND
    zero confident stress claims on healthy entities AND zero non-UNKNOWN damage claims. Otherwise FAIL."""
    p = d.get("paired", {})
    got: dict[str, object] = {}
    ok = bool(p)
    for key in ("CB_entity_cover_rmse", "CB_entity_cover_rmse_vs_production"):
        pooled = p.get(key, {}).get("pooled")
        if pooled is None:
            ok = False
            continue
        got[key] = {k: round(float(v), 5) for k, v in pooled.items()}
        ok &= pooled["mean"] > 0 and pooled["ci_low"] > 0
    spur = p.get("cefd_confident_stress_on_healthy", {})
    got["cefd_confident_stress_on_healthy"] = spur
    got["cefd_UEI_max"] = p.get("cefd_UEI_max")
    ok &= spur.get("max", 1.0) == 0.0 and p.get("cefd_UEI_max", 1.0) == 0.0
    return ok, json.dumps(got)


PLAN["2E"] = [
    (
        "entity model works",
        [
            EM + "test_entity_and_field_beliefs_are_separate_state_types",
            EM + "test_stress_is_inferred_and_never_observed_damage",
            EM + "test_observability_context_sets_survey_noise_but_never_couples_ecology",
            EP + "test_cover_belief_stays_bounded",
        ],
        _exp(ECO + "2E-E002-R3.json", _eco_entity_check),
    ),
    (
        "field model works",
        [
            EM + "test_variance_grows_away_from_sensor_and_with_time",
            EB + "test_error_does_not_grow_with_more_sensors_on_a_near_uniform_field",
            EB + "test_tau2_shrinks_when_stations_agree_and_grows_when_they_differ",
            EB + "test_noise_scale_and_drift_are_learned_from_the_readings",
            EB + "test_abrupt_change_is_detected_and_not_over_confident",
            EB + "test_change_detection_is_quiet_on_a_stationary_field",
            EP + "test_field_variance_stays_between_floor_and_prior",
            EP + "test_prediction_variance_is_monotone_in_delta_t",
        ],
        _exp(ECO + "2E-E001-R3.json", _eco_field_check),
    ),
    (
        "persistent inference works",
        [
            EM + "test_reset_working_memory_preserves_persistent_state",
            EM + "test_persisted_revisions_and_query",
            EM + "test_each_observation_uses_its_own_timestamp",
            EM + "test_internal_exception_sets_unavailable_and_is_raised",
        ],
        None,
    ),
]
PLAN["2E-CEFD"] = [
    (
        "CEFD beats uncoupled baselines without unsupported ecological claims",
        [EM + "test_stress_is_inferred_and_never_observed_damage"],
        _exp(ECO + "2E-E003-R3.json", _cefd_check),
    ),
]

# ---------------------------------------------------------------- 2T (functional) and 2T-TCDP (research)
# FINAL-3 artifacts of the R3 structural experiments (docs/audits/MODEL2T_REPAIR.md, iteration 3). The R2 FINAL
# seeds are spent; R3 runs on the config-local final_3 split 6500000-6500059 (partition name "final_3").
TD = "tests/unit/domains/technical/"
T2_E001 = "artifacts/experiments/2T-E001-R3/2T-E001-R3.json"
T2_E003 = "artifacts/experiments/2T-E003-R3/2T-E003-R3.json"
T2_FINAL_PARTITIONS = ("final_test", "final_3")


def _t2_direct_check(d: dict) -> tuple[bool, str]:
    """PASS iff on the FINAL partition, at every sensor-degradation level, Model2T direct inference beats
    LATEST_OBSERVATION for corrosion AND crack (paired bootstrap 95 % CI of the per-seed MAE gain above 0)
    and its 95 % interval coverage lies in the band declared in the config before the run."""
    v = d["verdicts"]
    band = v.get("coverage_band")
    got: dict[str, object] = {"partition": v.get("partition"), "coverage_band": band}
    ok = v.get("partition") in T2_FINAL_PARTITIONS
    levels = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in v if k.startswith("level_")})
    for lv in levels:
        for q in ("corrosion_depth_m", "crack_length_m"):
            gain = v.get(f"{lv}.{q}.mae_gain_vs_LATEST_OBSERVATION")
            cov = v.get(f"{lv}.{q}.model_cov95")
            beats = bool(v.get(f"{lv}.{q}.beats_latest_ci_above_0"))
            calibrated = bool(v.get(f"{lv}.{q}.model_cov95_in_band"))
            ok = ok and beats and calibrated
            got[f"{lv}.{q}"] = {
                "mae_gain_mm": None if gain is None else {k: round(float(x), 4) for k, x in gain.items()},
                "cov95": None if cov is None else round(float(cov), 3),
                "beats_latest": beats,
                "calibrated": calibrated,
            }
    return ok and bool(levels), json.dumps(got)


def _t2_tcdp_run() -> tuple[CriterionStatus, str]:
    """Benefit vs no propagation (CI above 0, declared rule in 2T-E003-R2) AND contamination below GENERIC.
    The 'excessive contamination' bound is OPEN: benefit + lower-than-generic -> NOT_EVALUABLE; no benefit, or
    contamination not below generic -> FAIL."""
    p = ROOT / T2_E003
    if not p.exists():
        return CriterionStatus.NOT_RUN, f"artifact missing: {T2_E003}"
    v = json.loads(p.read_text(encoding="utf-8"))["verdicts"]
    keys = [k for k in v if k.endswith(("RB_tcdp_ci", "rc_generic_minus_tcdp_ci")) and "." in k]
    got = {k: v[k] for k in keys if k.count(".") == 1}
    got.update(
        partition=v.get("partition"),
        tcdp_benefit=v.get("tcdp_benefit"),
        tcdp_contamination_lower_than_generic=v.get("tcdp_contamination_lower_than_generic"),
        excessive_contamination_threshold=v.get("excessive_contamination_threshold"),
    )
    measured = json.dumps(got, default=str)
    if v.get("partition") not in T2_FINAL_PARTITIONS or not v.get("tcdp_benefit"):
        return CriterionStatus.FAIL, measured
    if not v.get("tcdp_contamination_lower_than_generic"):
        return CriterionStatus.FAIL, measured
    if str(v.get("excessive_contamination_threshold", "OPEN")).upper() == "OPEN":
        return CriterionStatus.NOT_EVALUABLE, measured
    return CriterionStatus.PASS, measured


PLAN["2T"] = [
    (
        "direct inference works before TCDP gets credit",
        [
            TD + "test_m2t_direct.py::test_direct_update_is_observed_and_reduces_uo",
            TD + "test_m2t_direct.py::test_contradiction_raises_uc_and_is_not_averaged_away",
            TD + "test_m2t_measurement.py::test_missed_detection_never_drags_a_large_crack_to_zero",
            TD + "test_m2t_measurement.py::test_same_sensor_bias_floor_is_not_averaged_away",
        ],
        _exp(T2_E001, _t2_direct_check),
    ),
]
PLAN["2T-TCDP"] = [
    (
        "TCDP improves hidden-state reconstruction vs generic/no propagation without excessive contamination",
        [TD + "test_m2t_tcdp.py::test_no_propagation_over_invalid_relation_types"],
        _t2_tcdp_run,
    ),
]


I7A = "tests/acceptance/test_i7_constrained_comms.py::"
I7_E001 = "artifacts/experiments/COM-I7-E001/com_i7_e001.json"
I7_E002 = "artifacts/experiments/COM-I7-E002/com_i7_e002.json"


def _i7_bandwidth_check(d: dict) -> tuple[bool, str]:
    s = d["summary"]
    parts = [
        f"{c}: baac={s[c]['baac']['mission_information_retained']:.3f} "
        f"crit_latency={s[c]['baac']['critical_alert_latency_s']}"
        for c in s
    ]
    return d.get("partition") == "final_test", "; ".join(parts)


def _i7_outage_check(d: dict) -> tuple[bool, str]:
    runs = [r for r in d["per_run"] if r["run_id"].endswith("-shadow")]
    first = [r["reconnection"]["baac"][0]["critical_delivered_first"] for r in runs]
    lat = [r["arms"]["baac"]["critical_delta_latency_s"] for r in runs]
    ok = d.get("partition") == "final_test" and bool(runs) and all(first)
    return ok, f"critical_delivered_first={sum(first)}/{len(first)}; critical delta latency s={lat}"


def _i7_retention_check(d: dict) -> tuple[bool, str]:
    comp = d["comparisons"]
    rows, ok = [], d.get("partition") == "final_test"
    for cond, row in comp.items():
        if cond == "bw_0pct":
            continue
        for p in ("raw", "fifo", "fixed_priority"):
            ok = ok and bool(row[p]["baac_strictly_more_every_seed"])
        rows.append(f"{cond}: " + ", ".join(f"{p} {row[p]['baac_minus_policy_mean']:+.3f}" for p in row))
    return ok, "BAAC minus policy retained (mean): " + "; ".join(rows)


def _i7_latency_sync_check(d: dict) -> tuple[bool, str]:
    """ch26 Phase 11 asks for a COMPARISON of critical latency and sync error (superiority is the retention
    criterion). PASS = every arm has both measures at every non-zero level; cases where BAAC is worse are listed."""
    s, ok, worse = d["summary"], d.get("partition") == "final_test", []
    inf = float("inf")
    for cond, arms in s.items():
        if cond in ("bw_0pct", "bw_0.0pct"):
            continue
        for m in arms.values():
            ok = ok and "critical_alert_latency_s" in m and "sync_critical_all_equal_fraction" in m
        b = arms["baac"]
        b_lat = b["critical_alert_latency_s"] if b["critical_alert_latency_s"] is not None else inf
        for arm, m in arms.items():
            if arm == "baac":
                continue
            lat = m["critical_alert_latency_s"] if m["critical_alert_latency_s"] is not None else inf
            if lat < b_lat:
                worse.append(f"{cond} alert latency: {arm} {lat:.2f}s < baac {b_lat:.2f}s")
            if m["sync_critical_all_equal_fraction"] > b["sync_critical_all_equal_fraction"]:
                worse.append(f"{cond} sync: {arm} better than baac")
    return ok, "compared for every arm and level; BAAC worse in: " + ("; ".join(worse) or "none")


I1S = "tests/integration/test_i1_spatial_loop.py::"
I3S = "tests/integration/test_i3_structural.py::"
# SURROGATE evidence: the same criteria exercised through the Python L1 kernel instead of Unity (ADR-0008).
# It is reported next to the gate and can never promote it.
SURROGATE_PLAN: dict[str, list[tuple[str, list[str], Callable[[], tuple[CriterionStatus, str]] | None]]] = {
    "I1": [
        (
            "robot moves through synthetic world",
            [I1S + "test_spatial_beliefs_persist_with_direct_provenance"],
            None,
        ),
        ("persistent map", [I1S + "test_spatial_beliefs_persist_with_direct_provenance"], None),
        ("geometry", [I1S + "test_free_space_claims_agree_with_truth_where_claimed"], None),
        ("occupancy", [I1S + "test_free_space_claims_agree_with_truth_where_claimed"], None),
        ("coverage", [I1S + "test_map_keeps_unknown_space_next_to_observed_space"], None),
        ("observed/inferred/unknown", [I1S + "test_map_keeps_unknown_space_next_to_observed_space"], None),
        ("uncertainty", [I1S + "test_map_keeps_unknown_space_next_to_observed_space"], None),
        ("provenance", [I1S + "test_spatial_beliefs_persist_with_direct_provenance"], None),
        (
            "no Twin truth leakage",
            [
                I1S + "test_spatial_beliefs_carry_only_registry_identities",
                "tests/leakage/test_dynamic_leakage.py::test_no_world_entity_id_or_twin_key_on_the_runtime_side",
                "tests/leakage/test_dynamic_leakage.py::test_registry_ids_are_not_world_ids",
            ],
            None,
        ),
    ],
    # I7: COM-I7-E001 (bandwidth sweep) / COM-I7-E002 (critical finding during an outage), FINAL seeds.
    "I7": [
        (
            "full mission under constrained bandwidth",
            [
                I7A + "test_artifacts_are_final_split_surrogate",
                I7A + "test_full_mission_under_constrained_bandwidth",
                I7A + "test_no_duplicate_contribution_at_the_receiver",
            ],
            _exp(I7_E001, _i7_bandwidth_check),
        ),
        (
            "full mission under outages",
            [
                I7A + "test_outage_finding_is_created_while_the_link_is_down",
                I7A + "test_critical_delta_is_delivered_first_after_reconnection",
                I7A + "test_receiver_synchronised_for_critical_beliefs_after_reconnection",
                I7A + "test_stale_and_redundant_units_are_coalesced_or_dropped",
            ],
            _exp(I7_E002, _i7_outage_check),
        ),
        (
            "BAAC retains more mission-relevant information than raw/FIFO/fixed-priority",
            [
                I7A + "test_baac_retains_more_than_raw_fifo_fixed_priority_every_nonzero_level",
                I7A + "test_value_per_bit_comparison_is_reported",
            ],
            _exp(I7_E001, _i7_retention_check),
        ),
        (
            "critical latency and sync error compared against baselines",
            [I7A + "test_value_per_bit_comparison_is_reported"],
            _exp(I7_E001, _i7_latency_sync_check),
        ),
    ],
    # I5: the action-matrix criteria (belief-level fixtures, identical to the formal record) plus the integrated
    # missions of M1-ACTION-E002 (python kernel, FINAL seeds of configs/eval/partitions_i5.yaml).
    "I5": [
        *[c for c in PLAN["I5"] if c[0] not in I5_MISSION_CRITERIA],
        (
            I5_MISSION_CRITERIA[0],
            [
                I5M + "test_artifact_is_final_split_surrogate",
                I5M + "test_every_scenario_and_arm_ran_on_every_final_seed",
            ],
            _exp(I5_E002, _i5_actions_check),
        ),
        (
            I5_MISSION_CRITERIA[1],
            [I5M + "test_artifact_is_final_split_surrogate", I5M + "test_uir_is_measured_on_the_e001_basis"],
            _exp(I5_E002, _i5_trace_check),
        ),
        (
            I5_MISSION_CRITERIA[2],
            [
                I5M + "test_artifact_is_final_split_surrogate",
                I5M + "test_baselines_ran_closed_loop_on_the_same_seeds",
            ],
            _exp(I5_E002, _i5_competitive_check),
        ),
    ],
    "I3": [
        (
            "Twin2T + Twin2S -> Unity -> ECMER -> 2S + 2T",
            [I3S + "test_lane_views_associate_structural_observations_correctly"],
            None,
        ),
        (
            "robot sees only partial infrastructure",
            [I3S + "test_hidden_target_stays_unknown_from_the_lane"],
            None,
        ),
        (
            "persistent technical belief",
            [I3S + "test_observed_components_get_persistent_direct_beliefs"],
            None,
        ),
    ],
}


def run_nodes(nodes: list[str]) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "r.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={xml_path}",
                *nodes,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        results: dict[str, str] = {}
        if not xml_path.exists():
            return results
        for case in ET.parse(xml_path).getroot().iter("testcase"):
            file = case.get("classname", "").replace(".", "/") + ".py"
            name = case.get("name", "")
            bad = case.find("failure") is not None or case.find("error") is not None
            skipped = case.find("skipped") is not None
            results[f"{file}::{name}"] = "FAIL" if bad else ("SKIP" if skipped else "PASS")
        return results


def record(gate: str, surrogate: bool = False) -> GateEvidence:
    plan = (SURROGATE_PLAN if surrogate else PLAN)[gate]
    nodes = sorted({n for _, ns, _ in plan for n in ns})
    outcome = run_nodes(nodes)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    criteria = []
    for name, ns, check in plan:
        statuses = [outcome.get(n, "NOT_RUN") for n in ns]
        measured = "; ".join(f"{n.split('::')[-1]}={s}" for n, s in zip(ns, statuses, strict=True))
        if any(s == "FAIL" for s in statuses):
            status = CriterionStatus.FAIL
        elif any(s in ("NOT_RUN", "SKIP") for s in statuses):
            status = CriterionStatus.NOT_RUN
        else:
            status = CriterionStatus.PASS
        if check is not None and status is CriterionStatus.PASS:
            status, extra = check()
            measured += f"; experiment: {extra}"
        criteria.append(CriterionResult(criterion=name, status=status, measured=measured))
    ev = GateEvidence(
        gate_id=gate,
        evidence_class=EvidenceClass.SURROGATE if surrogate else EvidenceClass.FORMAL,
        execution_path=("python L1 kernel instead of Unity; " if surrogate else "")
        + f"pytest nodes + retained experiment artifacts ({gate})",
        git_commit=commit,
        criteria=tuple(criteria),
        artifacts=tuple(nodes),
    )
    write_evidence(ev)
    return ev


if __name__ == "__main__":
    args = sys.argv[1:]
    surrogate = "--surrogate" in args
    args = [a for a in args if a != "--surrogate"]
    for gate in args or list(SURROGATE_PLAN if surrogate else PLAN):
        ev = record(gate, surrogate)
        print(gate, [f"{c.criterion}:{c.status.value}" for c in ev.criteria])
