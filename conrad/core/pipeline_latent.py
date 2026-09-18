"""Optional learned latent path for the Model 2 Core step (EXPERIMENTAL_CANDIDATE modules only).

The analytic operators own the interpretable state and the four uncertainty channels. Learned
modules, when supplied, only evolve the latent tensors z/u/temporal kept on working nodes.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from conrad.core.belief_graph import WorkingBeliefGraph, WorkingNode
from conrad.core.buo import BeliefUpdateOperator
from conrad.core.rbp import RelationalBeliefPropagation
from conrad.core.tbd import TemporalBeliefDynamics
from conrad.schemas.observation import Evidence, EvidenceValidity


@dataclass
class LatentModules:
    buo: BeliefUpdateOperator | None = None
    rbp: RelationalBeliefPropagation | None = None
    tbd: TemporalBeliefDynamics | None = None

    def eval(self) -> None:
        for m in (self.buo, self.rbp, self.tbd):
            if m is not None:
                m.eval()


def quality_vector(ev: Evidence) -> list[float]:
    """[reliability, aleatoric, ood (0 when unmeasured), validity] -> BUO quality_k (Dq = 4)."""
    validity = {EvidenceValidity.VALID: 1.0, EvidenceValidity.DEGRADED: 0.5, EvidenceValidity.INVALID: 0.0}
    ood = ev.sensor_context.ood_score
    return [ev.reliability, ev.aleatoric_uncertainty, 0.0 if ood is None else ood, validity[ev.validity]]


@torch.no_grad()
def latent_predict(tbd: TemporalBeliefDynamics, node: WorkingNode, delta_t_s: float) -> None:
    pred = tbd(node.z[None], node.u[None], node.temporal[None], torch.tensor([delta_t_s]))
    node.z, node.u, node.temporal = pred.z[0], pred.u[0], pred.temporal[0]


@torch.no_grad()
def latent_correct(buo: BeliefUpdateOperator, node: WorkingNode, evidence: Sequence[Evidence]) -> Tensor:
    """Evidence embeddings feed the operator; the result is a NEW belief latent (never a copy of e_k)."""
    e = torch.tensor([list(ev.embedding) for ev in evidence], dtype=torch.float32)[None]
    q = torch.tensor([quality_vector(ev) for ev in evidence], dtype=torch.float32)[None]
    groups = torch.tensor(
        [[zlib.crc32((ev.independence_group or str(ev.evidence_id)).encode()) % (2**31) for ev in evidence]]
    )
    mask = torch.ones(1, len(evidence), dtype=torch.bool)
    out = buo(node.z[None], node.u[None], node.temporal[None], e, q, mask, groups)
    node.u = out.u[0]
    return out.z[0]


@torch.no_grad()
def latent_propagate(rbp: RelationalBeliefPropagation, graph: WorkingBeliefGraph) -> dict[object, Tensor]:
    batch = graph.batch()
    if batch.edge_index.shape[1] == 0:
        return {}
    out = rbp(batch.z, batch.u, batch.edge_index, batch.edge_type, batch.edge_numeric, batch.node_mask)
    return {bid: out.z[i] for i, bid in enumerate(batch.belief_ids) if bool(out.updated[i])}
