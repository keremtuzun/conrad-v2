"""ch33 'Model 1 EGDC exact implementation': learned scorer over the bounded Decision Claim Graph.

    node  = claim_embedding(256) + uncertainty(64) + provenance(32) + age/context(32) -> 384 -> 256
    enc   = 4 graph-transformer layers, 8 heads, typed-edge attention bias, [MISSION] token
    score = MLP([mission(256), action(128), resource(32), risk(32)], 448 -> 256 -> 128 -> 1)
    claim_support = attention(action_query, claim nodes)

The scorer ONLY ranks legal candidates. The ConstraintEngine is not a module of this network, takes
no gradient and cannot be bypassed by any score. Dimensions come from ``EGDCScorerConfig``.

implementation_status: EXPERIMENTAL_CANDIDATE (untrained architecture + imitation training function)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import Field
from torch import Tensor, nn

from conrad.decision.claims import ClaimGraph
from conrad.decision.consequence import ConsequenceVector
from conrad.decision.context import DecisionContext
from conrad.decision.learned_features import (
    _ACTIONS,
    NUM_EDGE_CODES,
    EGDCBatch,
    collate,
    expand_features,
    featurize,
)
from conrad.decision.policy import _stable_sort, _with_score
from conrad.schemas.base import ConradModel
from conrad.schemas.decision import (
    ActionProposal,
)


class EGDCScorerConfig(ConradModel):
    claim_embedding_dim: int = 256
    uncertainty_dim: int = 64
    provenance_dim: int = 32
    context_dim: int = 32
    input_hidden_dim: int = 384
    model_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    action_dim: int = 128
    resource_dim: int = 32
    risk_dim: int = 32
    score_hidden: tuple[int, int] = (256, 128)
    outcome_dim: int = 6
    max_nodes: int = 128
    dropout: float = 0.10
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    lambda_action: float = 1.0
    lambda_support: float = 0.5
    lambda_outcome: float = 0.25
    lambda_uir: float = 1.0
    seed: int = Field(default=2026201, ge=0)

    @property
    def node_input_dim(self) -> int:
        return self.claim_embedding_dim + self.uncertainty_dim + self.provenance_dim + self.context_dim


class _GraphTransformerLayer(nn.Module):
    def __init__(self, cfg: EGDCScorerConfig) -> None:
        super().__init__()
        if cfg.model_dim % cfg.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.h = cfg.num_heads
        self.dk = cfg.model_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.model_dim, 3 * cfg.model_dim)
        self.out = nn.Linear(cfg.model_dim, cfg.model_dim)
        self.edge_bias = nn.Embedding(NUM_EDGE_CODES, cfg.num_heads)
        self.norm1 = nn.LayerNorm(cfg.model_dim)
        self.norm2 = nn.LayerNorm(cfg.model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.model_dim, 4 * cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.model_dim, cfg.model_dim),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, edges: Tensor, key_mask: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, t, self.h, self.dk).transpose(1, 2)
        k = k.view(b, t, self.h, self.dk).transpose(1, 2)
        v = v.view(b, t, self.h, self.dk).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        logits = logits + self.edge_bias(edges).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        attn = self.drop(torch.softmax(logits, dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.drop(self.out(y))
        return x + self.drop(self.ffn(self.norm2(x)))


@dataclass
class EGDCOutput:
    scores: Tensor  # [B, A], -inf on masked actions
    claim_support: Tensor  # [B, A, N] attention over claim nodes (rows sum to 1 over valid nodes)
    outcome: Tensor  # [B, A, outcome_dim]


class EGDCScorer(nn.Module):
    def __init__(self, cfg: EGDCScorerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.node_in = nn.Sequential(
            nn.Linear(cfg.node_input_dim, cfg.input_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.input_hidden_dim),
            nn.Linear(cfg.input_hidden_dim, cfg.model_dim),
        )
        self.mission_token = nn.Parameter(torch.zeros(1, 1, cfg.model_dim))
        nn.init.normal_(self.mission_token, std=0.02)
        self.layers = nn.ModuleList([_GraphTransformerLayer(cfg) for _ in range(cfg.num_layers)])
        self.final_norm = nn.LayerNorm(cfg.model_dim)
        self.action_embedding = nn.Embedding(len(_ACTIONS), cfg.action_dim)
        self.claims_to_action = nn.Linear(cfg.model_dim, cfg.action_dim)
        self.action_query = nn.Linear(cfg.action_dim, cfg.model_dim)
        self.claim_key = nn.Linear(cfg.model_dim, cfg.model_dim)
        joint = cfg.model_dim + cfg.action_dim + cfg.resource_dim + cfg.risk_dim
        h1, h2 = cfg.score_hidden
        self.scorer = nn.Sequential(
            nn.Linear(joint, h1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, 1),
        )
        self.outcome_head = nn.Sequential(nn.Linear(joint, h2), nn.GELU(), nn.Linear(h2, cfg.outcome_dim))

    def forward(self, batch: EGDCBatch) -> EGDCOutput:
        b = batch.nodes.shape[0]
        x = torch.cat([self.mission_token.expand(b, -1, -1), self.node_in(batch.nodes)], dim=1)
        key_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool), batch.node_mask], dim=1)
        for layer in self.layers:
            x = layer(x, batch.edges, key_mask)
        x = self.final_norm(x)
        mission, claims = x[:, 0], x[:, 1:]
        rel = batch.action_claims.float()
        pooled = (rel @ claims) / rel.sum(-1, keepdim=True).clamp(min=1.0)
        action = self.action_embedding(batch.action_types) + self.claims_to_action(pooled)
        logits = (
            self.action_query(action) @ self.claim_key(claims).transpose(1, 2) / math.sqrt(claims.shape[-1])
        )
        logits = logits.masked_fill(~batch.node_mask[:, None, :], float("-inf"))
        support = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        a = action.shape[1]
        joint = torch.cat(
            [
                mission[:, None, :].expand(-1, a, -1),
                action,
                batch.resource[:, None, :].expand(-1, a, -1),
                batch.risk,
            ],
            dim=-1,
        )
        scores = self.scorer(joint).squeeze(-1).masked_fill(~batch.action_mask, float("-inf"))
        return EGDCOutput(scores=scores, claim_support=support, outcome=self.outcome_head(joint))


def build_scorer(cfg: EGDCScorerConfig) -> EGDCScorer:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        return EGDCScorer(cfg)


def egdc_loss(out: EGDCOutput, batch: EGDCBatch, cfg: EGDCScorerConfig) -> tuple[Tensor, dict[str, float]]:
    """CE on oracle/adjudicated action + support-edge supervision + outcome loss + UIR penalty."""
    if batch.target_action is None:
        raise ValueError("imitation loss needs target_action")
    action = nn.functional.cross_entropy(out.scores, batch.target_action)
    chosen = out.claim_support[torch.arange(out.scores.shape[0]), batch.target_action]  # [B, N]
    uir = (chosen * batch.unsupported_mask.float()).sum(-1).mean()
    support = torch.zeros(())
    if batch.target_support is not None:
        want = batch.target_support[torch.arange(out.scores.shape[0]), batch.target_action]
        support = nn.functional.binary_cross_entropy(chosen.clamp(1e-6, 1 - 1e-6), want.clamp(0, 1))
    outcome = torch.zeros(())
    if batch.target_outcome is not None:
        m = batch.action_mask[..., None].float()
        outcome = (
            ((out.outcome - batch.target_outcome) ** 2 * m).sum() / m.sum().clamp(min=1.0) / cfg.outcome_dim
        )
    total = (
        cfg.lambda_action * action
        + cfg.lambda_support * support
        + cfg.lambda_outcome * outcome
        + cfg.lambda_uir * uir
    )
    parts = {"action": float(action), "support": float(support), "outcome": float(outcome), "uir": float(uir)}
    return total, parts


def train_imitation(
    model: EGDCScorer, batches: Sequence[EGDCBatch], cfg: EGDCScorerConfig, epochs: int
) -> list[dict[str, float]]:
    """Imitation of oracle/adjudicated actions. Labels are supplied as tensors; truth never enters here."""
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    history: list[dict[str, float]] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        model.train()
        for _ in range(epochs):
            sums: dict[str, float] = {}
            for batch in batches:
                optim.zero_grad()
                loss, parts = egdc_loss(model(batch), batch, cfg)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                parts["total"] = float(loss)
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + v / len(batches)
            history.append(sums)
    model.eval()
    return history


class LearnedEGDCPolicy:
    """Second policy behind the ``DecisionPolicy`` interface. Its scores only order the candidates."""

    name = "egdc_learned_scorer"

    def __init__(self, model: EGDCScorer) -> None:
        self.model = model.eval()

    def rank(
        self,
        graph: ClaimGraph,
        candidates: Sequence[ActionProposal],
        consequences: Sequence[ConsequenceVector],
        ctx: DecisionContext,
    ) -> list[ActionProposal]:
        batch = featurize(graph, candidates, consequences, ctx, self.model.cfg)
        with torch.no_grad():
            scores = self.model(batch).scores[0].tolist()
        scored = [_with_score(a, s, c) for a, s, c in zip(candidates, scores, consequences, strict=True)]
        return _stable_sort(scored)


__all__ = [
    "EGDCBatch",
    "EGDCOutput",
    "EGDCScorer",
    "EGDCScorerConfig",
    "LearnedEGDCPolicy",
    "build_scorer",
    "collate",
    "egdc_loss",
    "expand_features",
    "featurize",
    "train_imitation",
]
