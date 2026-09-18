"""Cross-domain context from 2E / 2S (consumed only as CROSS_DOMAIN_CONTEXT).

Biofouling or poor visibility make a component harder to inspect: U_O and U_A rise and surface
measurements get noisier. Context NEVER changes a degradation value or severity: biofouling does not
imply corrosion.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from conrad.domains.technical.config import ContextConfig
from conrad.domains.technical.state import ComponentBelief
from conrad.schemas.belief import BeliefMessage
from conrad.schemas.world import Domain


@dataclass(frozen=True)
class ContextEffect:
    message_id: UUID
    provenance_refs: tuple[UUID, ...]
    target: UUID | None
    """Registry id of the affected component; None = asset-wide."""
    obscuration: float
    """0 = clear, 1 = the component cannot be inspected."""


def _unit(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def context_effects(messages: Sequence[BeliefMessage], cfg: ContextConfig) -> list[ContextEffect]:
    out: list[ContextEffect] = []
    for m in messages:
        if m.domain is Domain.TECHNICAL:
            continue
        levels: list[float] = []
        if m.ecological is not None:
            qty = m.ecological.quantities
            levels += [_unit(qty[k]) for k in cfg.biofouling_keys + cfg.turbidity_keys if k in qty]
            levels += [1.0 - _unit(qty[k]) for k in cfg.visibility_keys if k in qty]
        if m.spatial is not None and m.spatial.observation_count > 0:
            levels.append(1.0 - _unit(m.spatial.coverage))
        if levels:
            out.append(ContextEffect(m.message_id, m.provenance_refs, m.world_entity_id, max(levels)))
    return out


def apply_context(belief: ComponentBelief, obscuration: float, cfg: ContextConfig) -> bool:
    """Raise U_O / U_A only. Returns True when the belief's uncertainty changed."""
    uo = _unit(cfg.uo_gain * obscuration)
    ua = _unit(cfg.ua_gain * obscuration)
    gain = 1.0 + cfg.surface_variance_gain * obscuration
    changed = uo != belief.uo_context or ua != belief.ua_context
    belief.uo_context, belief.ua_context, belief.surface_var_gain = uo, ua, gain
    return changed
