"""Static guard: the Model2S package never imports truth-plane code nor names truth types / true poses."""

from __future__ import annotations

import ast
import re
from pathlib import Path

PKG = Path(__file__).resolve().parents[4] / "conrad" / "domains" / "spatial"
FORBIDDEN_IMPORTS = (
    "conrad.twins",
    "conrad.schemas.truth",
    "conrad.sim",
    "conrad.evaluation",
    "conrad.training",
)
FORBIDDEN_NAMES = {
    "TruthState",
    "SupervisionLabel",
    "TruthAccess",
    "get_truth",
    "occupancy_truth_at",
    "true_pose",
}
TRUE_POSE_TEXT = re.compile(r"true_(robot_)?pose|true_sensor", re.IGNORECASE)


def _modules() -> list[Path]:
    mods = sorted(PKG.rglob("*.py"))
    assert len(mods) >= 10, "spatial package not found"
    return mods


def test_spatial_package_never_imports_truth_plane():
    bad = []
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{path.name}: relative imports are not used here"
                names = [node.module or ""]
            bad += [
                f"{path.name}: {n}"
                for n in names
                if any(n == f or n.startswith(f + ".") for f in FORBIDDEN_IMPORTS)
            ]
    assert not bad, bad


def test_spatial_package_never_references_truth_types_or_true_poses():
    bad = []
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Name):
                ident = node.id
            elif isinstance(node, ast.Attribute):
                ident = node.attr
            else:
                continue
            if ident in FORBIDDEN_NAMES:
                bad.append(f"{path.name}:{node.lineno} {ident}")
        code_only = "\n".join(line.split("#")[0] for line in text.splitlines())
        if TRUE_POSE_TEXT.search(code_only):
            bad.append(f"{path.name}: mentions a true pose")
    assert not bad, bad
