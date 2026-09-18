"""Static architecture guards (INV-ARCH-01/04/07, CC-07, SS-09).

These walk the AST of every module, so a leak fails CI even if no test exercises that code path.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
PKG = ROOT / "conrad"

DEPLOYMENT_PACKAGES = (
    "core",
    "domains",
    "decision",
    "active",
    "communication",
    "robotics",
    "runtime",
    "orchestration",
)
FORBIDDEN_FOR_DEPLOYMENT = (
    "conrad.twins",
    "conrad.schemas.truth",
    "conrad.evaluation",
    "conrad.training",
    "conrad.sim",
)
# runtime may construct the simulated adapter, and only there.
ALLOWED_EXCEPTIONS = {("runtime", "conrad.sim.kernel"), ("runtime", "conrad.sim.scenarios")}
VENDOR_MODULES = (
    "rclpy",
    "rospy",
    "UnityEngine",
    "mlagents",
    "pymavlink",
    "brping",
    "pyrealsense2",
    "PySpin",
    "zmq",
)
CORE_NO_VENDOR = (
    "schemas",
    "core",
    "domains",
    "twins",
    "decision",
    "active",
    "communication",
    "robotics",
    "runtime",
)
LEGACY_MARKERS = (
    "rov_digital_twin",
    "Digital Twin",
    "digital_twin",
    "ChatGPT/conrad",
    "ChatGPT" + chr(92) + "conrad",
)
OBSOLETE_PATTERN = re.compile(r"twin_?1(?![0-9])|model_?2[abc](?![a-z])")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
            found += [f"{node.module}.{a.name}" for a in node.names]
    return found


def _modules(package: str) -> list[Path]:
    base = PKG / package
    return sorted(base.rglob("*.py")) if base.exists() else []


@pytest.mark.parametrize("package", DEPLOYMENT_PACKAGES)
def test_deployment_planes_never_import_truth(package: str) -> None:
    violations = []
    for path in _modules(package):
        for name in _imports(path):
            for bad in FORBIDDEN_FOR_DEPLOYMENT:
                if (name == bad or name.startswith(bad + ".")) and not any(
                    package == pkg and name.startswith(ok) for pkg, ok in ALLOWED_EXCEPTIONS
                ):
                    violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert not violations, "truth-plane leakage:\n" + "\n".join(sorted(set(violations)))


def test_deployment_planes_never_name_truth_types() -> None:
    banned = {"TruthState", "SupervisionLabel", "TruthAccess", "get_truth"}
    violations = []
    for package in DEPLOYMENT_PACKAGES:
        for path in _modules(package):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                ident = (
                    node.id
                    if isinstance(node, ast.Name)
                    else node.attr
                    if isinstance(node, ast.Attribute)
                    else None
                )
                if ident in banned:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} references {ident}")
    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("package", CORE_NO_VENDOR)
def test_no_vendor_sdk_outside_adapters(package: str) -> None:
    violations = [
        f"{path.relative_to(ROOT)} imports {name}"
        for path in _modules(package)
        for name in _imports(path)
        if name.split(".")[0] in VENDOR_MODULES
    ]
    assert not violations, "\n".join(violations)


def test_only_the_gateway_calls_hardware_send() -> None:
    """Command Gateway is the sole caller of RobotHardwareInterface.send / set_thruster_commands."""
    allowed = {
        PKG / "runtime" / "command_gateway.py",
        PKG / "robotics" / "hardware" / "interface.py",
    }
    violations = []
    for path in sorted(PKG.rglob("*.py")):
        if path in allowed or "adapters" in path.parts or (path.parts[-3:-1] == ("sim", "kernel")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("send", "set_thruster_commands")
            ):
                target = ast.unparse(node.func.value)
                if any(tok in target.lower() for tok in ("hw", "hardware", "robot", "rhi")):
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} calls {target}.{node.func.attr}"
                    )
    assert not violations, "actuator path bypasses the Command Gateway:\n" + "\n".join(violations)


def test_no_legacy_repository_reference_anywhere() -> None:
    """Conrad V2 has zero dependency on the legacy repository (prompt s3-s6, INV-ARCH-04)."""
    violations = []
    targets = [*PKG.rglob("*.py"), *(ROOT / "configs").rglob("*.yaml"), ROOT / "pyproject.toml"]
    for path in targets:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in LEGACY_MARKERS:
            if marker in text:
                violations.append(f"{path.relative_to(ROOT)} mentions {marker!r}")
    assert not violations, "\n".join(violations)


def test_no_symlinks_or_submodules() -> None:
    assert not (ROOT / ".gitmodules").exists()
    links = [p for p in PKG.rglob("*") if p.is_symlink()]
    assert not links


def test_no_obsolete_architecture_identifiers() -> None:
    """Twin1 and Model2A/B/C must not exist as modules, classes, functions or variables."""
    violations = []
    for path in sorted(PKG.rglob("*.py")):
        if OBSOLETE_PATTERN.search(path.stem.lower()):
            violations.append(f"module {path.relative_to(ROOT)}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = getattr(node, "name", None) or getattr(node, "id", None)
            if isinstance(name, str) and OBSOLETE_PATTERN.search(name.lower()):
                violations.append(f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)} {name}")
    assert not violations, "\n".join(violations)
