"""2E observability context for other children: CROSS_DOMAIN_CONTEXT messages derived from beliefs.

* surface biofouling cover belief per registered structure (affects observability; NOT corrosion);
* turbidity belief affecting observability of each horizontal quadrant of the belief grid.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from conrad.domains.ecological import messages as mb
from conrad.domains.ecological.cefd_analytic import visibility
from conrad.domains.ecological.entity_belief import BIOFOULING
from conrad.domains.ecological.readout import CONTEXT_NOTE, _message
from conrad.domains.ecological.views import cover_prior_var, entity_type_of
from conrad.schemas.belief import BeliefMessage, EcologicalPayload, KnowledgeStatus
from conrad.schemas.provenance import ProvenanceRecord, SourceType
from conrad.schemas.timebase import TimeStamp

if TYPE_CHECKING:
    from conrad.domains.ecological.model2e import Model2E

K = KnowledgeStatus


def observability_context(m: Model2E, now: TimeStamp) -> list[BeliefMessage]:
    out: list[BeliefMessage] = []
    records: list[ProvenanceRecord] = []
    pv = cover_prior_var(m)
    for bid, b in m.entities.beliefs.items():
        if b.eco_class != BIOFOULING or b.asset is None or b.n_hits == 0 or bid not in m.ledger.heads:
            continue
        rec = m.ledger.record(
            SourceType.CROSS_DOMAIN_CONTEXT,
            (bid,),
            bid,
            now,
            (m.ledger.heads[bid].root,),
            "context.surface_biofouling_cover",
        )
        records.append(rec)
        u = mb.entity_uncertainty(b, b.cover_var, pv)
        claims = (
            mb.claim("surface_biofouling_cover", b.cover_mean, "1", K.INFERRED, u, rec.record_id),
            mb.claim(
                "surface_biofouling_cover.sd", math.sqrt(b.cover_var), "1", K.INFERRED, u, rec.record_id
            ),
            mb.claim("surface_occlusion_fraction", b.cover_mean, "1", K.INFERRED, u, rec.record_id),
        )
        q = {"surface_biofouling_cover": b.cover_mean, "surface_occlusion_fraction": b.cover_mean}
        out.append(
            _message(
                m,
                bid,
                now,
                claims,
                u,
                mb.entity_embedding(b, (b.cover_mean, b.cover_var), b.presence_p),
                mb.entity_support(b),
                EcologicalPayload(kind="ENTITY", quantities=q, units=dict.fromkeys(q, "1")),
                rec.record_id,
                change=f"{CONTEXT_NOTE}; surface biofouling cover belief on "
                f"{entity_type_of(b)}; affects observability only, not evidence of corrosion",
            )
        )
    if "turbidity" in m.field_ids and m.fields.fields["turbidity"].n_obs > 0:
        out.extend(_turbidity_context(m, now, records))
    m.ledger.store_records(records)
    return out


def _turbidity_context(m: Model2E, now: TimeStamp, records: list[ProvenanceRecord]) -> list[BeliefMessage]:
    g = m.fields.grid
    bid = m.field_ids["turbidity"]
    mean, var = m.fields.moments_at_time("turbidity", now.time_ns)
    mid = g.origin + 0.5 * g.spacing * np.asarray(g.shape)
    out = []
    for qx in (0, 1):
        for qy in (0, 1):
            cells = ((g.centers[:, 0] >= mid[0]) == bool(qx)) & ((g.centers[:, 1] >= mid[1]) == bool(qy))
            tm = float(np.mean(mean[0, cells]))
            tv = float(np.mean(var[0, cells]))
            ua, ue, uc, _ = m.fields.uncertainty_channels("turbidity", cells)
            u = mb.unc(ua, ue, uc, m.fields.predicted_uo("turbidity", now.time_ns, cells))
            rec = m.ledger.record(
                SourceType.CROSS_DOMAIN_CONTEXT,
                (bid,),
                bid,
                now,
                (m.ledger.heads[bid].root,),
                "context.turbidity_observability",
            )
            records.append(rec)
            cc = m.cfg.coupling
            atten = cc.beam_attenuation_clear_per_m + cc.beam_attenuation_per_m_per_ntu * tm
            q = {
                "turbidity_ntu": tm,
                "turbidity_sd_ntu": math.sqrt(tv),
                "beam_attenuation_per_m": atten,
                "visibility_fraction_at_5m": visibility(tm, 5.0, cc),
            }
            units = {
                "turbidity_ntu": "NTU",
                "turbidity_sd_ntu": "NTU",
                "beam_attenuation_per_m": "m-1",
                "visibility_fraction_at_5m": "1",
            }
            claims = tuple(mb.claim(k, v, units[k], K.INFERRED, u, rec.record_id) for k, v in q.items())
            out.append(
                _message(
                    m,
                    bid,
                    now,
                    claims,
                    u,
                    mb.field_embedding(q),
                    mb.field_support(m.fields, cells),
                    EcologicalPayload(kind="FIELD", quantities=q, units=units),
                    rec.record_id,
                    change=f"{CONTEXT_NOTE}; turbidity belief affecting observability of region",
                )
            )
    return out
