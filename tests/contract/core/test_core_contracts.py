"""Contracts: no truth imports from conrad.core, package metadata, 4-channel uncertainty, CC-05 / CC-06."""

import ast
import importlib
from pathlib import Path

import torch

import conrad.core as core_pkg
from conrad.core.config import tiny_config
from conrad.core.primitives import SetAggregator, masked_loss
from conrad.core.uncertainty import channels_to_uncertainty
from conrad.schemas.uncertainty import Uncertainty

CORE_DIR = Path(core_pkg.__file__).parent
FORBIDDEN = ("conrad.twins", "conrad.schemas.truth", "conrad.sim", "conrad.evaluation", "conrad.training")


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_core_never_imports_truth_plane_or_evaluation():
    offenders = [
        (str(p.relative_to(CORE_DIR)), mod)
        for p in CORE_DIR.rglob("*.py")
        for mod in _imports(p)
        if mod.startswith(FORBIDDEN)
    ]
    assert offenders == []


def test_core_does_not_use_uuid4_or_wall_clock():
    for p in CORE_DIR.rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "uuid4" not in text, p
        assert "time.time(" not in text and "datetime.now" not in text, p


def test_core_has_no_torchvision_or_pretrained_weights():
    for p in CORE_DIR.rglob("*.py"):
        mods = list(_imports(p))
        assert not any(m.startswith(("torchvision", "timm")) for m in mods), p
        assert "pretrained=True" not in p.read_text(encoding="utf-8")


def test_package_metadata_contract():
    for name in ("conrad.core", "conrad.core.ecmer"):
        meta = importlib.import_module(name).IMPLEMENTATION_METADATA
        assert meta["implementation_status"] == "EXPERIMENTAL_CANDIDATE"
        assert meta["claim_status"] in ("NONE", "IMPLEMENTED")
        for key in ("source_sections", "configuration_keys", "assumptions", "baselines", "acceptance_tests"):
            assert meta[key], key


def test_uncertainty_is_always_four_channels_and_rejects_invalid():
    u = channels_to_uncertainty(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert isinstance(u, Uncertainty) and u.as_tuple() == pytest_approx((0.1, 0.2, 0.3, 0.4))
    for bad in ((0.1, 0.2, 0.3), (0.1, -0.2, 0.3, 0.4), (0.1, float("nan"), 0.0, 0.0)):
        try:
            channels_to_uncertainty(bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid channels {bad}")


def pytest_approx(values):
    import pytest

    return pytest.approx(values, abs=1e-6)


def test_cc05_named():
    loss, den = masked_loss(torch.randn(4), torch.zeros(4, dtype=torch.bool))
    assert float(loss) == 0.0 and float(den) == 0.0


def test_cc06_named():
    cfg = tiny_config()
    agg = SetAggregator(cfg.evidence_dim, cfg.heads).eval()
    x = torch.randn(1, 3, cfg.evidence_dim)
    base = agg(x, torch.ones(1, 3, dtype=torch.bool))
    shuffled = torch.cat([x[:, [2, 0, 1]], torch.randn(1, 2, cfg.evidence_dim)], 1)
    assert torch.allclose(base, agg(shuffled, torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)), atol=1e-5)
