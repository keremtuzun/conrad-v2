"""Generator baselines (H-T2E-01, ch12 MEIFE baselines, ch13 Twin2E baselines) and ablations.

Each baseline is the SAME simulator with switches changed, so only the generator mechanism
differs between training sets (H-T2E-01 key experiment).

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from conrad.twins.twin2e.config import MeifeSwitches, Twin2EConfig

_OFF_COUPLING = {"field_to_entity": False, "entity_to_field": False}

BASELINE_SWITCHES: dict[str, MeifeSwitches] = {
    "full_meife": MeifeSwitches(),
    "static_ecology": MeifeSwitches(entity_dynamics=False),
    "static_fields": MeifeSwitches(field_dynamics=False),
    "independent_noise_ecology": MeifeSwitches(random_ecology=True, **_OFF_COUPLING),
    "independent_entities": MeifeSwitches(field_dynamics=False, **_OFF_COUPLING),
    "independent_fields": MeifeSwitches(entity_dynamics=False, **_OFF_COUPLING),
    "uncoupled_stochastic": MeifeSwitches(**_OFF_COUPLING),
    "one_way_field_to_entity": MeifeSwitches(entity_to_field=False),
    "single_scale_coupled": MeifeSwitches(multi_scale=False),
    "mechanistic_no_stochasticity": MeifeSwitches(stochasticity=False),
}
"""H-T2E-01 list: static ecology, independent entities, independent fields, one-way field->entity,
single-scale coupling, full MEIFE; plus ch12/13 extras (static fields, independent-noise ecology,
uncoupled stochastic, mechanistic without stochasticity)."""

ABLATIONS: dict[str, MeifeSwitches] = {
    f"no_{name}": MeifeSwitches(**{name: False})
    for name in (
        "field_to_entity",
        "entity_to_field",
        "multi_scale",
        "disturbances",
        "stochasticity",
        "recovery",
    )
}
"""ch13 'And ablate' (the learned residual is already off by default and has its own switch)."""


def baseline_config(name: str, base: Twin2EConfig | None = None) -> Twin2EConfig:
    table = {**BASELINE_SWITCHES, **ABLATIONS}
    if name not in table:
        raise KeyError(f"unknown Twin2E baseline/ablation {name!r}; known: {sorted(table)}")
    return (base or Twin2EConfig()).model_copy(update={"switches": table[name]})
