"""The experiment harness hands the belief side design data only, and a tiny E003 episode runs."""

from __future__ import annotations

from conrad.evaluation.structural_experiments.common import registry_context
from conrad.evaluation.structural_experiments.e003_tcdp import run_episode_kind
from conrad.evaluation.structural_experiments.scenarios import make_scenario


def test_registry_context_is_design_only():
    ctx = registry_context(make_scenario(5, {"kind": "tier", "tier": 3}))
    keys = {k for c in ctx["asset_registry"]["components"] for k in c}
    assert keys <= {"registry_id", "parent_id", "component_type", "material", "coating", "wall_thickness_m"}
    assert not any(k.startswith("initial_") for k in keys)


def test_e003_episode_smoke(tmp_path):
    acc = run_episode_kind("misleading", 7, {"steps": 3, "burn_in_steps": 1, "n_segments": 3}, tmp_path)
    assert any(k.endswith("hidden_mae_mm") for k in acc)
    assert all(x >= 0 for k, vals in acc.items() if k.endswith("claim_rate_hidden") for x in vals)
