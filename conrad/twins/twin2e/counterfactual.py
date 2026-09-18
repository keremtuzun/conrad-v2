"""Twin2E counterfactual pairs and controlled confounders (ch12 Coupling supervision, ch13).

A pair is two scenarios identical in world, seed and every sampled prior, except ONE causal
factor. Priors are resolved before the toggle so the two runs draw identical random streams.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from conrad.schemas.ids import IdFactory
from conrad.schemas.world import Scenario
from conrad.twins.twin2e.ecology import SESSILE
from conrad.twins.twin2e.priors import ENVIRONMENT_PRIORS, populate_ecological_state

CONFOUNDER_KINDS = (
    "HIGH_TEMPERATURE_STABLE_ECOLOGY",
    "NORMAL_TEMPERATURE_DISTURBED_ECOLOGY",
    "MULTIPLE_CAUSES",
)


@dataclass(frozen=True)
class CounterfactualPair:
    factual: Scenario
    counterfactual: Scenario
    factor: str
    factual_value: Any
    counterfactual_value: Any


def _resolved(scenario: Scenario, rng: np.random.Generator | None) -> Scenario:
    return populate_ecological_state(scenario, rng or np.random.default_rng(scenario.seed + 7919))


def _edit(scenario: Scenario, factor: str, value: Any) -> tuple[Scenario, Any]:
    data = scenario.model_dump(mode="python")
    head, _, key = factor.partition(".")
    if head == "environment":
        if key not in ENVIRONMENT_PRIORS:
            raise KeyError(f"unknown environment factor {key!r}")
        old = data["environment"][key]["value"]
        data["environment"][key] = {
            "value": value,
            "units": ENVIRONMENT_PRIORS[key].units,
            "source": "COUNTERFACTUAL",
        }
    elif head == "parameters":
        old = data["ecological_state"]["parameters"][key]
        data["ecological_state"]["parameters"][key] = value
    elif head == "entities":
        eid, _, field = key.partition(".")
        old = data["ecological_state"]["entities"][eid][field]
        data["ecological_state"]["entities"][eid][field] = value
    elif head == "remove_event":
        old = [e for e in data["events"] if str(e["event_id"]) == key]
        if not old:
            raise KeyError(f"no event {key}")
        data["events"] = [e for e in data["events"] if str(e["event_id"]) != key]
    else:
        raise KeyError(f"unsupported counterfactual factor {factor!r}")
    return Scenario.model_validate(data), old


def make_counterfactual_pair(
    scenario: Scenario, factor: str, value: Any = None, rng: np.random.Generator | None = None
) -> CounterfactualPair:
    """``factor`` is 'environment.<key>', 'parameters.<key>', 'entities.<id>.<key>' or 'remove_event.<id>'."""
    factual = _resolved(scenario, rng)
    cf, old = _edit(factual, factor, value)
    meta = {**cf.metadata, "counterfactual_of": str(factual.scenario_id), "counterfactual_factor": factor}
    cf = cf.model_copy(update={"metadata": meta})
    return CounterfactualPair(factual, cf, factor, old, value)


def make_confounded_variant(
    scenario: Scenario,
    kind: str,
    ids: IdFactory,
    magnitude: float = 4.0,
    rng: np.random.Generator | None = None,
) -> Scenario:
    """Break the 'high temperature -> poor condition' shortcut with controlled confounders."""
    if kind not in CONFOUNDER_KINDS:
        raise KeyError(f"unknown confounder kind {kind!r}")
    base = _resolved(scenario, rng)
    data = base.model_dump(mode="python")
    ents = data["ecological_state"]["entities"]
    sessile = [k for k, v in ents.items() if v["entity_kind"] in SESSILE]
    if kind == "HIGH_TEMPERATURE_STABLE_ECOLOGY":
        data["environment"]["temperature"]["value"] = (
            float(data["environment"]["temperature"]["value"]) + magnitude
        )
        for k in sessile:  # a thermally tolerant community tracks the warmer regime
            ents[k]["thermal_optimum_c"] = float(ents[k]["thermal_optimum_c"]) + magnitude
    elif kind == "NORMAL_TEMPERATURE_DISTURBED_ECOLOGY":
        for k in sessile:  # non-environmental cause (e.g. disease pressure), temperature unchanged
            ents[k]["independent_mortality_per_day"] = 0.02 * magnitude
    else:
        for k in sessile:
            ents[k]["independent_mortality_per_day"] = 0.01 * magnitude
        data["events"] = [
            *data["events"],
            {
                "event_id": ids.new(),
                "time_s": 0.0,
                "event_type": "TEMPERATURE_ANOMALY",
                "target_entity_id": None,
                "parameters": {"delta_c": magnitude, "duration_s": 5 * 86400.0, "units": "degC"},
            },
        ]
    data["metadata"] = {**data["metadata"], "confounder": kind, "confounder_magnitude": magnitude}
    return Scenario.model_validate(data)
