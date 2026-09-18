"""Static guards: the robotics stack never actuates hardware directly and never reads sim truth."""

import ast
from pathlib import Path

from conrad.robotics.navigation.stack import _ReadOnlyHardware

ROOT = Path(__file__).resolve().parents[3]
ROBOTICS = ROOT / "conrad" / "robotics"
FORBIDDEN_IMPORTS = (
    "conrad.sim",
    "conrad.twins",
    "conrad.schemas.truth",
    "conrad.evaluation",
    "conrad.training",
)
ACTUATION_CALLS = {"send", "set_thruster_commands"}


def _py(root):
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def test_robotics_never_calls_hardware_send():
    offenders = []
    for path in [*_py(ROBOTICS), *_py(ROOT / "conrad" / "evaluation" / "nav_benchmarks")]:
        if path.relative_to(ROOT).as_posix() == "conrad/robotics/hardware/interface.py":
            continue  # defines the abstract method and the spec alias
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ACTUATION_CALLS
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_robotics_imports_no_truth():
    offenders = []
    for path in _py(ROBOTICS):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.relative_to(ROOT)}: {n}" for n in names if n.startswith(FORBIDDEN_IMPORTS)]
    assert offenders == []


def test_stack_hardware_facade_has_no_actuation():
    facade = _ReadOnlyHardware.__new__(_ReadOnlyHardware)
    assert not hasattr(facade, "send") and not hasattr(facade, "set_thruster_commands")


def test_every_package_declares_metadata():
    import importlib

    for name in ("estimation", "navigation", "trajectory", "control", "allocation", "safety"):
        meta = importlib.import_module(f"conrad.robotics.{name}").IMPLEMENTATION_METADATA
        assert meta["claim_status"] in ("NONE", "IMPLEMENTED")
    for name in ("conrad.sim.kernel", "conrad.evaluation.nav_benchmarks"):
        assert "claim_status" in importlib.import_module(name).IMPLEMENTATION_METADATA
