"""2E baselines behind ONE interface: each is Model2E with different CEFD switches (ch12 E-B*).

* ``production``    the production default switches (``CefdSwitches()``: observability context only, ADR-0007)
* ``cefd``          full analytic CEFD (observability + ecological stress coupling + entity->field sink)
* ``entity_only``   entity beliefs only; field evidence ignored; turbidity-blind survey noise (E-B2-like)
* ``field_only``    field beliefs only; survey evidence ignored (E-B3)
* ``uncoupled``     entity + field beliefs, no cross-type link at all, turbidity-blind survey noise (E-B7)
* ``static_field``  CEFD with a static, spatially uniform field model (no delta_t dynamics, no kernel)

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from conrad.domains.ecological.config import CefdSwitches, Model2EConfig
from conrad.domains.ecological.model2e import Model2E
from conrad.persistence.repository import Repository
from conrad.schemas.ids import IdFactory

_NO_COUPLING = {"observability_context": False, "ecological_coupling": False, "entity_to_field": False}
_FULL_CEFD = {"observability_context": True, "ecological_coupling": True, "entity_to_field": True}

BASELINES: dict[str, CefdSwitches] = {
    # the shipped default (ADR-0007 + addendum): observability context ON, ecological coupling OFF
    "production": CefdSwitches(),
    "cefd": CefdSwitches(**_FULL_CEFD),
    "entity_only": CefdSwitches(fields=False, **_NO_COUPLING),
    "field_only": CefdSwitches(entities=False, **_NO_COUPLING),
    "uncoupled": CefdSwitches(**_NO_COUPLING),
    "static_field": CefdSwitches(**_FULL_CEFD, field_dynamics=False, spatial_correlation=False),
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
