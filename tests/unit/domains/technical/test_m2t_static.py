"""BELIEF PLANE: conrad.domains.technical never touches truth (brief rule 2)."""

from __future__ import annotations

import ast
from pathlib import Path

import conrad.domains.technical as pkg

FORBIDDEN = ("conrad.twins", "conrad.schemas.truth", "conrad.sim", "conrad.evaluation")
TRUTH_NAMES = ("TruthState", "SupervisionLabel", "true_world_entity_id", "ComponentRuntime", "Twin2T")


def _modules() -> list[Path]:
    return sorted(Path(pkg.__file__).parent.rglob("*.py"))


def test_no_truth_imports():
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert not n.startswith(FORBIDDEN), f"{path.name} imports {n}"


def test_no_truth_type_references():
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        idents |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not idents & set(TRUTH_NAMES), f"{path.name} references {idents & set(TRUTH_NAMES)}"


def test_metadata_present():
    meta = pkg.IMPLEMENTATION_METADATA
    assert meta["implementation_status"] == "EXPERIMENTAL_CANDIDATE"
    assert meta["claim_status"] in ("NONE", "IMPLEMENTED", "EVALUATED")
