"""Model 2 Core: ECMER, association, BUO, RBP, TBD, PMBL and uncertainty (ch3-ch9, ch33).

BELIEF PLANE: nothing in this package may import truth-plane code (twins, schemas.truth, sim truth,
evaluation). Learned modules are EXPERIMENTAL_CANDIDATE; the analytic operators are the default
runtime path until a learned module beats them under the registered comparator protocol.
"""

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch3 Model 2 Core",
        "ch4 ECMER",
        "ch5 BUO",
        "ch6 Uncertainty",
        "ch7 TBD",
        "ch8 PMBL",
        "ch9 Graph/RBP",
        "ch28 Tensor and Mask Semantics",
        "ch33 V0.2 exact implementations",
        "ch27 H-CORE-01..04",
    ],
    "configuration_keys": ["model2_core.* (conrad.core.config.CoreConfig)"],
    "assumptions": [
        "ch33 association pair features sum to 1109 at defaults, not the stated 853; all listed parts kept",
        "learned BUO state update uses LN(z + trust * gate * innovation)",
        "AnalyticBUO/AnalyticTBD/AnalyticRBP are the default runtime operators (ENGINEERING_ESTIMATE constants)",
        "property variances persist as '<name>.variance' PropertyClaims (no variance field in the contract)",
        "relational estimates are replaced, not re-fused, to avoid double counting",
        "merge/split re-derive successor state by replaying archived evidence",
    ],
    "baselines": [
        "association: nearest_neighbour_decision",
        "buo: AnalyticBUO (Kalman-style), latest-only, simple averaging",
        "rbp: no propagation, AnalyticRBP, untyped RBP",
        "tbd: HoldLastTBD, LinearExtrapolationTBD, MlpTBD, AnalyticTBD",
        "ecmer: single modality, concat, early, late, cross-attention",
    ],
    "acceptance_tests": ["tests/unit/core", "tests/property/core", "tests/contract/core"],
    "claim_status": "IMPLEMENTED",
}
