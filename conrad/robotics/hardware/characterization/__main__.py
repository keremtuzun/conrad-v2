"""Merge a characterization handoff into a new RobotConfig version.

    python -m conrad.robotics.hardware.characterization \\
        --base configs/robot/physical_template.yaml --handoff handoff/2026-10/handoff.yaml \\
        --version 0.3.0 --out configs/robot/physical_v0_3_0.yaml --report artifacts/characterization/change_0_3_0.json

Nothing is written unless ingestion, unit conversion and the merge all succeed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from conrad.robotics.hardware.characterization.ingest import load_bundle
from conrad.robotics.hardware.characterization.merge import merge_characterization
from conrad.robotics.hardware.config import load_robot_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m conrad.robotics.hardware.characterization")
    ap.add_argument("--base", required=True)
    ap.add_argument("--handoff", required=True, help=".yaml or .csv handoff file")
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--allow-downgrade", action="store_true")
    args = ap.parse_args(argv)

    base = load_robot_config(args.base)
    merged, report = merge_characterization(
        base, load_bundle(args.handoff), args.version, allow_downgrade=args.allow_downgrade
    )
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; choose a new version file")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(merged.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(
        f"{merged.config_name} {base.config_version} -> {merged.config_version}: {len(report.changes)} changes, "
        f"{len(report.remaining_open)} still OPEN, {len(report.remaining_ungrounded)} not MEASURED/IDENTIFIED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
