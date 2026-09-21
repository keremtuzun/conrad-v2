"""Frozen production planner behind the MCBR interface (selected on VALIDATION data, 2026-09-19).

``configs/active/mcbr_frozen_v2.yaml`` (v2; v1 = ``mcbr_frozen.yaml``) names the selected ranking rule and its parameters. The file carries the
SHA-256 of its canonical ``planner`` section; loading refuses a copy whose content no longer matches, so the
production planner cannot drift silently after the final evaluation. The mission runtime uses it by default
(``MissionRuntimeConfig.planner = "PRODUCTION"``).

implementation_status: EXPERIMENTAL_CANDIDATE (frozen selection)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from conrad.active.baselines import make_planners
from conrad.active.config import MCBRConfig
from conrad.active.planner import MCBRPlanner
from conrad.active.rankers import RankerConfig, ranker_planner
from conrad.schemas.ids import IdFactory
from conrad.settings import REPO_ROOT

PRODUCTION = "PRODUCTION"
FROZEN_PATH = REPO_ROOT / "configs" / "active" / "mcbr_frozen_v4.yaml"
"""The frozen production selection. It pointed at ``mcbr_frozen_v2.yaml`` until 2026-09-21 while the audit
trail, the gate I4 run-2 harness and ``artifacts/gates/I4/unity_i4_occluded_rep2_results.json`` all named
``mcbr_frozen_v3.yaml``. No result was affected, because v2 and v3 have byte-identical ``planner`` and
``mission_predictive`` sections (``config_digest 8eca16cc...5d903``), so the planner that flew was the
planner the audit describes. v4 keeps those two sections byte-identical again and adds the ``view_execution``
section selected in ``docs/audits/MCBR_V4.md``, so the pointer now names the file that actually describes
production. v2 and v3 are left on disk untouched as historical evidence."""


class FrozenConfigError(RuntimeError):
    pass


def planner_digest(planner_section: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(planner_section, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_frozen(path: Path = FROZEN_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FrozenConfigError(f"no frozen production planner at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw["planner"]
    digest = planner_digest(section)
    if digest != raw["config_digest"]:
        raise FrozenConfigError(f"{path}: planner section digest {digest} != recorded {raw['config_digest']}")
    if (
        "mission_predictive" in raw
    ):  # v2+: the mission's belief-side predictive model is frozen with the planner
        pd = planner_digest(raw["mission_predictive"])
        if pd != raw.get("mission_predictive_digest"):
            raise FrozenConfigError(
                f"{path}: mission_predictive digest {pd} != recorded {raw.get('mission_predictive_digest')}"
            )
    if "view_execution" in raw:  # v4+: the view execution protocol is frozen with the planner
        vd = planner_digest(raw["view_execution"])
        if vd != raw.get("view_execution_digest"):
            raise FrozenConfigError(
                f"{path}: view_execution digest {vd} != recorded {raw.get('view_execution_digest')}"
            )
    return dict(raw)


def production_view_execution(path: Path = FROZEN_PATH) -> dict[str, Any] | None:
    """The frozen mission view execution protocol, if this frozen file declares one (v4+)."""
    raw = load_frozen(path).get("view_execution")
    return None if raw is None else dict(raw)


def build_planner(section: dict[str, Any], ids: IdFactory, cfg: MCBRConfig, name: str) -> MCBRPlanner:
    """``section`` = {"base": <make_planners name>} or {"ranker": RankerConfig fields, "with_stop": bool};
    an optional ``mcbr`` mapping overrides fields of the shared candidate-generation config."""
    if section.get("mcbr"):
        cfg = cfg.model_copy(update=dict(section["mcbr"]))
    if "ranker" in section:
        return ranker_planner(
            ids, cfg, RankerConfig(**section["ranker"]), name, with_stop=bool(section.get("with_stop", False))
        )
    planners = make_planners(ids, cfg)
    base = str(section["base"])
    if base not in planners:
        raise FrozenConfigError(f"unknown base planner {base!r}")
    p = planners[base]
    p.name = name
    return p


def production_planner(ids: IdFactory, cfg: MCBRConfig, path: Path = FROZEN_PATH) -> MCBRPlanner:
    frozen = load_frozen(path)
    section = frozen["planner"]
    return build_planner(section, ids, cfg, f"{PRODUCTION}[{section['selected']}]")
