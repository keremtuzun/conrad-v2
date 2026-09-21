"""Run one MCBR V4 experiment config without invoking git.

``conrad.evaluation.dispatch`` shells out to git for the registry record; this runner does not, and reads
the checked-out commit from ``.git/HEAD`` exactly as ``conrad.sim.mission.run`` does. It writes the same
experiment JSON into ``artifacts/experiments/<experiment_id>/`` and prints the headline tables.

    python -m uv run python scripts/run_mcbr_v4_experiment.py configs/eval/i4_v4_diagnostic.yaml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def git_commit() -> str:
    """Read-only lookup of the checked-out commit; no git command is executed."""
    head = REPO / ".git" / "HEAD"
    if not head.exists():
        return "UNAVAILABLE_NO_GIT_METADATA"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        target = REPO / ".git" / ref[5:]
        return target.read_text(encoding="utf-8").strip() if target.exists() else "UNAVAILABLE_UNRESOLVED_REF"
    return ref


def main(config_path: str) -> dict[str, Any]:
    from conrad.evaluation.decision_experiments import i4_view_execution as mod

    cfg = yaml.safe_load((REPO / config_path).read_text(encoding="utf-8"))
    seeds = [int(s) for s in cfg.pop("seeds", [])]
    experiment_id = str(cfg.get("experiment_id", mod.EXPERIMENT))
    out_dir = REPO / "artifacts" / "experiments" / experiment_id
    t0 = time.perf_counter()
    result = mod.run(cfg, seeds, out_dir)
    wall = time.perf_counter() - t0
    name = str(cfg.get("out_name", "i4_view_execution.json"))
    full = json.loads((out_dir / name).read_text(encoding="utf-8"))
    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "config": config_path,
                "code_commit": git_commit(),
                "wall_clock_s": wall,
                "partition": full["partition"],
                "partition_digest": full["partition_digest"],
                "n_worlds": full["n_worlds"],
                "arms": full["arms"],
            },
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"{experiment_id}: {full['n_worlds']} worlds, {len(full['arms'])} arms, {wall:.0f} s")
    print(json.dumps(full["read_table"], indent=1))
    print(json.dumps(full["execution_breakdown"], indent=1))
    print(mod.table_markdown(full))
    return dict(result)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/eval/i4_v4_diagnostic.yaml")
