"""2E baselines behind ONE interface: each is Model2E with different CEFD switches (ch12 E-B*).

* ``production``    the production default switches (``CefdSwitches()``; uncoupled per ADR-0007)
* ``cefd``          full analytic CEFD (field->entity observability + stress context, entity->field sink)
* ``entity_only``   entity beliefs only; field evidence ignored; turbidity-blind survey noise (E-B2-like)
* ``field_only``    field beliefs only; survey evidence ignored (E-B3)
* ``uncoupled``     entity + field beliefs, no coupling in either direction (E-B7)
* ``static_field``  CEFD with a static, spatially uniform field model (no delta_t dynamics, no kernel)

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from conrad.domains.ecological.config import CefdSwitches, Model2EConfig
from conrad.domains.ecological.model2e import Model2E
from conrad.persistence.repository import Repository
from conrad.schemas.ids import IdFactory

_NO_COUPLING = {"field_to_entity": False, "entity_to_field": False}

BASELINES: dict[str, CefdSwitches] = {
    "production": CefdSwitches(),  # the shipped default (ADR-0007: uncoupled), whatever it currently is
    "cefd": CefdSwitches(field_to_entity=True, entity_to_field=True),
    "entity_only": CefdSwitches(fields=False, **_NO_COUPLING),
    "field_only": CefdSwitches(entities=False, **_NO_COUPLING),
    "uncoupled": CefdSwitches(**_NO_COUPLING),
    "static_field": CefdSwitches(
        field_to_entity=True, entity_to_field=True, field_dynamics=False, spatial_correlation=False
    ),
}


def baseline_config(name: str, base: Model2EConfig | None = None) -> Model2EConfig:
    if name not in BASELINES:
        raise KeyError(f"unknown 2E baseline {name!r}; known: {sorted(BASELINES)}")
    cfg = base or Model2EConfig()
    return cfg.model_copy(
        update={"switches": BASELINES[name], "model_version": f"{cfg.model_version}+{name}"}
    )


def make_baseline(
    name: str, ids: IdFactory, base: Model2EConfig | None = None, repo: Repository | None = None
) -> Model2E:
    return Model2E(baseline_config(name, base), ids, repo)
