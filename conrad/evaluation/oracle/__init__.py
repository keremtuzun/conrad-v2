"""Evaluation oracles. TRUTH PLANE: only evaluation / training code may import this package.

Decision-plane packages (conrad.decision, conrad.active, conrad.communication, conrad.orchestration)
never import it; experiments compute oracle results and pass them as data.
"""

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch16 Oracle supervision / BAAC oracle",
        "ch17 Where labels come from",
        "ch18 Oracle",
    ],
    "configuration_keys": [],
    "assumptions": ["all worlds are SYNTHETIC_ONLY abstractions, not physical models"],
    "baselines": [],
    "acceptance_tests": ["tests/unit/active (leakage checks)"],
    "claim_status": "IMPLEMENTED",
}
