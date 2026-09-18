"""Hypothesis registry seeded from ch27. Every hypothesis starts UNTESTED.

Baselines, experiments and kill rules are transcribed from the specification; nothing here is a
result. Status changes only through recorded experiments.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from conrad.schemas.base import ConradModel


class HypothesisStatus(str, Enum):
    UNTESTED = "UNTESTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    KILL_CANDIDATE = "KILL_CANDIDATE"
    INCONCLUSIVE = "INCONCLUSIVE"


class Hypothesis(ConradModel):
    hypothesis_id: str = Field(pattern=r"^H-[A-Z0-9]+-[0-9]{2}$")
    mechanism: str
    priority_tier: str = Field(pattern="^[ABC]$")
    statement: str
    baselines: tuple[str, ...]
    experiments: tuple[str, ...]
    metrics: tuple[str, ...]
    kill_rule: str
    status: HypothesisStatus = HypothesisStatus.UNTESTED


def _h(
    hid: str, mech: str, tier: str, text: str, base: str, exps: str, metrics: str, kill: str
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hid,
        mechanism=mech,
        priority_tier=tier,
        statement=text,
        baselines=tuple(b.strip() for b in base.split(";")),
        experiments=tuple(e.strip() for e in exps.split(";")),
        metrics=tuple(m.strip() for m in metrics.split(";")),
        kill_rule=kill,
    )


SEED_HYPOTHESES: tuple[Hypothesis, ...] = (
    _h(
        "H-CORE-01",
        "PBA",
        "A",
        "Persistent structured belief improves hidden-state reconstruction under partial observation.",
        "latest-only; GRU/LSTM; Transformer history; SSM; RSSM/world model; persistent scene graph",
        "E-CORE-01.1 observation sparsity 100/50/20/5%; 01.2 long temporal gaps; 01.3 re-observation after dormancy; "
        "01.4 entity disappearance/reappearance; 01.5 contradictory history; 01.6 long sequence persistence",
        "state error; calibration; memory; latency; identity consistency; unsupported-confidence rate",
        "If RSSM/scene-graph/recurrent baselines match PBA, narrow the general PBA claim.",
    ),
    _h(
        "H-CORE-02",
        "BUO",
        "A",
        "Separating trust, innovation and contradiction improves belief updating over confidence-weighted fusion.",
        "weighted averaging; Bayesian/Kalman-style fusion; confidence gating; GRU update; attention update; "
        "BUO without contradiction memory",
        "E-BUO-02.1 reliable support; 02.2 unreliable support; 02.3 reliable contradiction; 02.4 unreliable "
        "contradiction; 02.5 repeated independent contradiction; 02.6 contradiction resolved; 02.7 sensor degradation",
        "state error; calibration; contradiction detection; resolution time; confident-wrong rate",
        "If explicit trust/innovation decomposition adds nothing, simplify.",
    ),
    _h(
        "H-CORE-03",
        "RBP",
        "A",
        "Typed relational propagation improves hidden-state inference while provenance separates inference from evidence.",
        "no propagation; generic GNN; GAT; temporal GNN; untyped RBP",
        "E-RBP-03.1 hidden related component; 03.2 unrelated neighbour; 03.3 wrong relationship; 03.4 uncertain "
        "relationship; 03.5 multi-hop; 03.6 missing direct evidence",
        "hidden-state error; relational contamination RC; calibration by inference depth",
        "If generic GNN performs equally well, RBP stays an implementation structure, not a mechanism.",
    ),
    _h(
        "H-CORE-04",
        "TBD",
        "A",
        "Explicit physical-time belief dynamics improve prediction/correction under sparse observations.",
        "hold-last-state; linear extrapolation; GRU; LSTM; Transformer; SSM; RSSM; Neural ODE",
        "vary delta_t; smooth dynamics; abrupt events; interventions; long gaps; incorrect dynamics assumptions",
        "temporal state error; uncertainty growth between observations",
        "If TBD does not beat simpler dynamics, use simpler domain dynamics.",
    ),
    _h(
        "H-2T-01",
        "TCDP",
        "A",
        "Mechanism-conditioned propagation over engineering topology improves hidden structural condition inference.",
        "independent components; static GNN; temporal GNN; RBP without mechanism conditioning",
        "E-2T-01.1 corrosion; 01.2 fatigue topology; 01.3 degraded next to healthy unrelated; 01.4 wrong topology; "
        "01.5 partial coverage; 01.6 unseen infrastructure; 01.7 mixed mechanisms",
        "hidden error; relational contamination RC",
        "If TCDP ~= generic temporal GNN, do not force TCDP into the paper.",
    ),
    _h(
        "H-T2T-01",
        "MCDE",
        "A",
        "Mechanism-coupled topology-aware degradation simulation yields better structural intelligence.",
        "random static defects; independent stochastic degradation; engineering model; engineering + stochasticity; "
        "engineering + topology; MCDE without coupling",
        "train identical 2T models per generator; evaluate on the same held-out real data",
        "TU_2T; real structural prediction/calibration",
        "If more elaborate simulation does not improve transfer, simplify Twin2T.",
    ),
    _h(
        "H-2E-01",
        "CEFD",
        "B",
        "Coupled ecological entity and environmental field beliefs improve temporal ecological inference.",
        "entity only; field only; concatenation; cross-attention; multimodal Transformer; uncoupled dual model; "
        "heterogeneous GNN",
        "environmental change with/without ecological effect; unrelated ecological cause; sparse field; sparse "
        "ecology; temporal gaps; OOD habitat",
        "CB_entity; CB_field; unsupported ecological inference UEI",
        "If simple cross-attention/uncoupled models match CEFD, remove the coupling complexity.",
    ),
    _h(
        "H-T2E-01",
        "MEIFE",
        "B",
        "Multi-scale coupled ecological/environmental simulation provides useful training signal.",
        "static ecology; independent entities; independent fields; one-way field->entity; single-scale coupling",
        "same Model2E, only the generator changes, held-out real evaluation; paired counterfactual driver test",
        "TU_2E; coupling/temporal behaviour",
        "If TU_2E is not positive, simplify the generator.",
    ),
    _h(
        "H-2S-01",
        "UAHSM",
        "A",
        "Knowledge status and uncertainty in hierarchical spatial memory improve safe spatial reasoning.",
        "occupancy map; sparse voxel/octree; flat learned memory; temporal learned memory; neural field; persistent "
        "memory without uncertainty; UAHSM without hierarchy",
        "partial coverage; occlusion; pose uncertainty; sensor degradation; dynamic objects; unseen geometry; long "
        "mission memory",
        "UC unsupported confidence; reconstruction; navigation success; collision rate",
        "If hierarchical UAHSM ~= flat sparse memory, use the simpler representation.",
    ),
    _h(
        "H-T2S-01",
        "OCPWE",
        "A",
        "Controlling observability and ambiguity in world generation improves uncertainty and active perception.",
        "fixed worlds; random procedural worlds; procedural + random trajectories; OCPWE without counterfactual "
        "worlds; OCPWE without trajectory control",
        "train identical UAHSM/MCBR per generator; unseen synthetic worlds; held-out real spatial data; counterfactual "
        "worlds W_A != W_B with O_A ~= O_B",
        "U_O before/after discriminating observation; calibration; next-best-observation behaviour",
        "If OCPWE does not improve calibration or NBO behaviour, use conventional procedural generation.",
    ),
    _h(
        "H-M1-01",
        "EGDC",
        "B",
        "Evidence-grounded deliberation reduces unsupported world-dependent reasoning.",
        "FSM; behavior tree; rule controller; utility controller; classical planner; generic agent; generic agent + "
        "tools; RL; EGDC without claim graph",
        "correct/missing/stale beliefs; contradiction; OOD; cross-domain disagreement; mission pressure; resource "
        "constraint; hardware fault",
        "UIR; mission success; decision latency; unnecessary escalation",
        "If removing the Decision Claim Graph does not worsen UIR, the claim graph is not earning its complexity.",
    ),
    _h(
        "H-ACT-01",
        "MCBR",
        "A",
        "Mission-conditioned belief-property-specific acquisition beats generic coverage/entropy/NBV.",
        "random; fixed inspection; coverage; frontier; geometric NBV; entropy NBV; expected IG; uncertainty NBV; "
        "RL active perception; MCBR without mission conditioning",
        "U_A/U_O/U_C/U_E dominated; competing hypotheses; irrelevant high-entropy region; expensive informative "
        "view; sequential sensing",
        "delta belief error; delta mission error; information/energy; information/time; regret vs twin oracle",
        "If MCBR does not target mission-critical uncertainty better than generic NBV/IG, drop it.",
    ),
    _h(
        "H-COM-01",
        "BAAC",
        "B",
        "Belief-aware adaptive communication preserves mission-relevant information under constrained channels.",
        "send everything; FIFO; fixed priority; fixed compression; value/bit heuristic; semantic communication; "
        "BAAC without receiver state; BAAC without uncertainty; BAAC fixed fidelity",
        "bandwidth 100/50/10/1/0.1/0%; latency; packet loss; blackout windows",
        "information/bits; information/joule; critical latency; receiver sync error; deadline success",
        "If BAAC ~= a good deterministic scheduler, the scheduler may be the better product.",
    ),
    _h(
        "H-XFER-01",
        "Twin transfer",
        "A",
        "Domain-twin pretraining improves real-domain performance after controlled adaptation.",
        "G0 random init; G1 generic pretraining; G3 generic + twin; G4 real-only",
        "G0-G4 with identical real adaptation budget, same architecture, untouched final real test",
        "TU_d = Perf_real(Real+Twin_d) - Perf_real(RealOnly)",
        "TU_d must be > 0 with repeated-seed evidence and no major calibration regression.",
    ),
    _h(
        "H-XFER-02",
        "Structured randomization",
        "C",
        "Physically correlated randomization transfers better than independent arbitrary randomization.",
        "no randomization; independent broad; physically constrained independent",
        "compare randomization schemes on real performance",
        "real performance",
        "If structured randomization does not transfer better, use the simpler scheme.",
    ),
    _h(
        "H-XFER-03",
        "Reality-gap decomposition",
        "C",
        "Separating G_domain, G_sensor, G_robot allows better targeted adaptation.",
        "global randomization only; sensor calibration only; robot ID only; domain adaptation only",
        "RGDA-style targeted combination vs single-gap methods",
        "integrated physical/public-real performance",
        "Treat RGDA as engineering methodology unless experiments show otherwise.",
    ),
    _h(
        "H-NAV-01",
        "Belief-aware navigation",
        "C",
        "Model2S knowledge status and uncertainty improve mission safety/efficiency over occupancy alone.",
        "occupancy only; occupancy + binary unknown; occupancy + scalar uncertainty",
        "hidden obstacles; inferred free space; pose degradation; partial mapping",
        "collision; success; path; time; energy",
        "If knowledge status does not improve safety/efficiency, plan over occupancy.",
    ),
    _h(
        "H-NAV-02",
        "Hybrid control",
        "C",
        "Classical control plus learned residual compensates residual underwater dynamics.",
        "PID; MPC; robust/adaptive; RL",
        "MPC + learned residual vs classical, only with evidence of systematic residual dynamics",
        "tracking error; robustness",
        "Pursue only if physical/sim evidence shows systematic residual dynamics.",
    ),
)

SYSTEM_EXPERIMENTS: dict[str, str] = {
    "X-E01": "Does better uncertainty improve autonomous behaviour (none / scalar / 4-channel)?",
    "X-E02": "Provenance value: none / source IDs / full DAG under conflicting claims",
    "X-E03": "Persistent vs frame vs short-history intelligence on identical missions",
    "X-E04": "Active vs fixed inspection at the same time or energy budget",
    "X-E05": "Communication-aware autonomy under channel collapse",
    "SYS-E01": "Flagship partially observable pipeline mission vs progressively simpler stacks",
    "SYS-E02": "Ablation ladder removing one mechanism at a time",
    "SYS-E03": "Failure cascade and combined degradation",
    "SYS-E04": "OOD mission on an unseen pipeline family",
}


class HypothesisRegistry:
    def __init__(self, hypotheses: tuple[Hypothesis, ...] = SEED_HYPOTHESES) -> None:
        ids = [h.hypothesis_id for h in hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate hypothesis id")
        self._by_id = {h.hypothesis_id: h for h in hypotheses}

    def get(self, hypothesis_id: str) -> Hypothesis:
        return self._by_id[hypothesis_id]

    def all(self) -> tuple[Hypothesis, ...]:
        return tuple(self._by_id.values())

    def by_tier(self, tier: str) -> tuple[Hypothesis, ...]:
        return tuple(h for h in self._by_id.values() if h.priority_tier == tier)
