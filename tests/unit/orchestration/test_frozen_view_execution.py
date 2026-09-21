"""The frozen V4 view execution protocol travels with the frozen production planner."""

import pytest
import yaml

from conrad.active.production import (
    FROZEN_PATH,
    PRODUCTION,
    FrozenConfigError,
    load_frozen,
    planner_digest,
    production_view_execution,
)
from conrad.orchestration.mission_config import runtime_config


def test_frozen_file_declares_a_view_execution_protocol_with_a_matching_digest():
    raw = load_frozen()
    assert "view_execution" in raw, "v4 freezes the execution protocol, not only the ranking rule"
    assert planner_digest(raw["view_execution"]) == raw["view_execution_digest"]
    assert raw["view_execution"]["enabled"] is True
    assert raw["view_execution"]["protect_active_view"] is True


def test_production_runtime_adopts_the_frozen_protocol():
    cfg = runtime_config({"planner": PRODUCTION})
    frozen = production_view_execution()
    assert frozen is not None
    assert cfg.view_execution.model_dump(mode="json") == frozen


def test_an_explicit_protocol_in_the_mission_config_wins_over_the_frozen_one():
    """An experiment must still be able to run the incumbent protocol as a control arm."""
    cfg = runtime_config({"planner": PRODUCTION, "view_execution": {"enabled": False}})
    assert cfg.view_execution.enabled is False


def test_a_baseline_planner_does_not_silently_inherit_the_frozen_protocol():
    cfg = runtime_config({"planner": "A-B2_coverage"})
    assert cfg.view_execution.enabled is False


def test_an_edited_view_execution_section_fails_closed(tmp_path):
    raw = yaml.safe_load(FROZEN_PATH.read_text(encoding="utf-8"))
    raw["view_execution"]["protect_active_view"] = False
    f = tmp_path / "mcbr_frozen_v4.yaml"
    f.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(FrozenConfigError, match="view_execution digest"):
        load_frozen(f)
