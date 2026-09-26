"""Run the prospectively declared Spatial V1.1 native-macOS Unity parity supplement once."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import scripts.validate_spatial_unity_parity as parity

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYER = (
    REPO_ROOT
    / "unity"
    / "ConradUnityV2"
    / "Builds"
    / "macOS-ARM64"
    / "ConradSim.app"
    / "Contents"
    / "MacOS"
    / "ConradSim"
)
PLAYER_SHA256 = "7e42f64199b043204d62737d06aabc6b6ceb1f3c1c699f158a612b938ea87ac6"

parity.PLAYER_SHA256 = PLAYER_SHA256
parity.SEEDS = (2003, 2004)
parity.PROTOCOL_EXTRA = {
    "protocol_id": "SPATIAL-V1.1-MACOS-PARITY-1",
    "platform": "macOS 26.6.2 (25G83) arm64 host",
    "unity_editor": "6000.5.9f1 (b57deb96f08d)",
    "player_format": "Mach-O universal (x86_64, arm64), native arm64 execution",
    "player_path": "unity/ConradUnityV2/Builds/macOS-ARM64/ConradSim.app/Contents/MacOS/ConradSim",
}


def main() -> None:
    if sys.platform != "darwin":
        raise SystemExit("SPATIAL-V1.1-MACOS-PARITY-1 requires macOS")
    os.environ["CONRAD_UNITY_PLAYER"] = str(PLAYER)
    parity.main()


if __name__ == "__main__":
    main()
