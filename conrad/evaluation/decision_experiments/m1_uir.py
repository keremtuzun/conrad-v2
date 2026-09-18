"""M1-UIR-E001: UIR and safe behaviour of EGDC under injected upstream faults vs a naive baseline.

The naive arm (``NaiveActOnClaimsPolicy`` + grounding enforcement off) reads claimed values at face
value. Truth lives in ``conrad.evaluation.oracle.decision_oracle``; EGDC only sees belief contexts.
SYNTHETIC_ONLY belief-level fixtures; no perception.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from conrad.decision import (
    EGDC,
    DecisionConfig,
    NaiveActOnClaimsPolicy,
    uir_report,
)
from conrad.decision.context import DecisionContext
from conrad.evaluation.decision_experiments.fixtures import (
    make_belief,
    make_context,
    make_requirement,
    unc,
)
from conrad.evaluation.oracle.decision_oracle import DecisionTruth, Fault, judge, sample_truth
from conrad.schemas.belief import Availability
from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Domain

EXPERIMENT_ID = "M1-UIR-E001"


def build_context(ids: IdFactory, truth: DecisionTruth, now_s: float = 200.0) -> DecisionContext:
    reported = (
        truth.true_condition
        if truth.belief_value_correct
        else ("NOMINAL" if truth.true_condition == "DAMAGED" else "DAMAGED")
    )
    f = truth.fault
    target_entity = ids.new()
    kwargs: dict[str, Any] = {
        "properties": {"condition": reported},
        "time_s": now_s - 1.0,
        "uncertainty": unc(),
        "world_entity_id": target_entity,
    }
    if f is Fault.STALE:
        kwargs["time_s"] = now_s - 120.0
    elif f is Fault.WRONG_ASSOCIATION:
        kwargs["world_entity_id"] = ids.new()
    elif f is Fault.CONTRADICTION:
        kwargs["uncertainty"] = unc(uc=0.8)
        kwargs["n_conflicts"] = 2
    elif f is Fault.MISCALIBRATED_FLAGGED:
        kwargs["uncertainty"] = unc(calibrated=False)
    elif f is Fault.OOD:
        kwargs["uncertainty"] = unc(ue=0.8)
    elif f is Fault.NO_EVIDENCE_PATH:
        kwargs["n_evidence"] = 0
    belief = make_belief(ids, **kwargs)
    coverage = 0.1 if f is Fault.CROSS_DOMAIN_DISAGREEMENT else 0.95
    spatial = make_belief(
        ids,
        domain=Domain.SPATIAL,
        properties={"occupied": True},
        coverage=coverage,
        time_s=now_s - 1.0,
        uncertainty=unc(uo=0.9 if coverage < 0.5 else 0.05),
    )
    messages = [spatial] if f is Fault.MISSING else [belief, spatial]
    req = make_requirement(
        ids,
        belief_ids=[belief.belief_id],
        entity_ids=[target_entity],
        consequence=0.9,
        context_domains=[Domain.SPATIAL],
    )
    availability = {Domain.TECHNICAL: Availability.UNAVAILABLE} if f is Fault.DOMAIN_UNAVAILABLE else None
    return make_context(ids, messages, [req], now_s=now_s, availability=availability)


def _arms(ids: IdFactory) -> dict[str, EGDC]:
    naive_cfg = DecisionConfig(enforce_grounding=False, model_version="naive-baseline-0.2")
    return {
        "egdc_structured": EGDC(ids),
        "naive_act_on_claims": EGDC(ids, config=naive_cfg, policy=NaiveActOnClaimsPolicy()),
    }


def run_seed(seed: int, n_per_fault: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = IdFactory(seed)
    arms = _arms(ids)
    records: dict[str, list[Any]] = {k: [] for k in arms}
    rows: dict[str, list[dict[str, Any]]] = {k: [] for k in arms}
    for fault in Fault:
        for _ in range(n_per_fault):
            truth = sample_truth(rng, fault)
            ctx = build_context(ids, truth)
            for name, egdc in arms.items():
                out = egdc.decide(ctx)
                action = None if out.record.chosen is None else out.record.chosen.action_type
                j = judge(truth, action)
                records[name].append(out.record)
                rows[name].append(
                    {
                        "fault": fault.value,
                        "action": None if action is None else action.value,
                        "abstained": out.record.abstained,
                        **j.model_dump(),
                    }
                )
    result: dict[str, Any] = {}
    for name in arms:
        r = rows[name]
        uir = uir_report(records[name])
        clean_nominal = [
            x for x in r if x["fault"] == Fault.CLEAN.value and x["unnecessary_action"] is not None
        ]
        per_fault = {f.value: float(np.mean([x["safe"] for x in r if x["fault"] == f.value])) for f in Fault}
        result[name] = {
            "uir": uir.unsupported_inference_rate,
            "relied_world_claims": uir.relied_world_claims,
            "relied_unsupported_claims": uir.relied_unsupported_claims,
            "unsupported_claim_fraction": uir.unsupported_claim_fraction,
            "safe_rate": float(np.mean([x["safe"] for x in r])),
            "safe_rate_faults_only": float(
                np.mean([x["safe"] for x in r if x["fault"] != Fault.CLEAN.value])
            ),
            "missed_damage_rate": float(np.mean([x["missed_damage"] for x in r])),
            "unnecessary_action_rate_clean_nominal": float(
                np.mean([x["unnecessary_action"] for x in clean_nominal if x["fault"] == Fault.CLEAN.value])
            )
            if clean_nominal
            else None,
            "abstention_rate": float(np.mean([x["abstained"] for x in r])),
            "safe_rate_by_fault": per_fault,
            "actions_by_fault": {
                f.value: dict(Counter(str(x["action"]) for x in r if x["fault"] == f.value)) for f in Fault
            },
        }
    return result


def run(config: dict[str, Any], seeds: list[int], out_dir: str | Path) -> dict[str, Any]:
    n = int(config.get("n_per_fault", 20))
    per_seed = {str(s): run_seed(s, n) for s in seeds}
    arms = list(next(iter(per_seed.values())))
    summary: dict[str, Any] = {}
    for arm in arms:
        summary[arm] = {}
        for key in (
            "uir",
            "safe_rate",
            "safe_rate_faults_only",
            "missed_damage_rate",
            "abstention_rate",
            "unnecessary_action_rate_clean_nominal",
        ):
            vals = [per_seed[s][arm][key] for s in per_seed if per_seed[s][arm][key] is not None]
            summary[arm][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        summary[arm]["safe_rate_by_fault_mean"] = {
            f.value: float(np.mean([per_seed[s][arm]["safe_rate_by_fault"][f.value] for s in per_seed]))
            for f in Fault
        }
    result = {
        "experiment_id": EXPERIMENT_ID,
        "data_status": "SYNTHETIC_ONLY",
        "config": config,
        "seeds": seeds,
        "summary": summary,
        "per_seed": per_seed,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "m1_uir_e001.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
