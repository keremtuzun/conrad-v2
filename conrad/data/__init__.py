"""Dataset registry: manifests, lineage-aware splits, adapters, SSL corpus and sim-to-real ledger.

Real and synthetic data converge on the shared ``Observation`` schema here; supervision travels
in a physically separate channel. Nothing in this package downloads data or invents statistics.
"""

IMPLEMENTATION_METADATA = {
    "implementation_status": "EXPERIMENTAL_CANDIDATE",
    "source_sections": [
        "ch23 Training Architecture",
        "ch24 Dataset Production Registry",
        "ch26 Dataset pipeline / Data splits",
        "ch35 Priority 2 DATA-CONRAD-SSL-01",
        "ch35 Priority 4 SIMREAL-LEDGER-01",
        "ch36 Data, labels and provenance requirements",
    ],
    "configuration_keys": ["data.manifest_ids", "data.split_hash"],
    "assumptions": [
        "a manifest's dataset checksum is sha256 over the sorted (path, sha256, byte_length) file inventory",
        "lineage groups are connected components over the chosen split units",
        "public dataset manifests are templates only until a DATA-VERIFY task fills them",
    ],
    "baselines": [],
    "acceptance_tests": [
        "tests/unit/data",
        "tests/property/data",
    ],
    "claim_status": "IMPLEMENTED",
}
