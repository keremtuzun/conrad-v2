"""R5 is a prospective, sampling-consistent change on the unused half of v9 development."""

from pathlib import Path

import yaml

from conrad.evaluation import partitions as P
from conrad.evaluation.dispatch import EXPERIMENTS
from conrad.robotics.estimation.ekf import EkfConfig

ROOT = Path(__file__).resolve().parents[3]
R4 = ROOT / "configs/eval/m1_action_spatial_v1_development_r4.yaml"
R5 = ROOT / "configs/eval/m1_action_spatial_v1_development_r5.yaml"


def _config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_r5_uses_only_the_unopened_half_of_v9_development():
    cfg = _config(R5)
    development = P.split(P.I5_V9_DOMAIN, P.Partition.DEVELOPMENT, P.Purpose.DESIGN).world_seeds
    assert cfg["seeds"] == list(development[5:])
    assert not set(cfg["seeds"]) & set(development[:5])
    assert cfg["partition"] == "development"


def test_r5_changes_only_the_declared_speed_envelope_from_r4():
    r4, r5 = _config(R4), _config(R5)
    for cfg in (r4, r5):
        cfg.pop("experiment_id")
        cfg.pop("hypothesis")
        cfg.pop("seeds")
    r5_speed = r5["spatial_v1"]["runtime"].pop("cruise_speed_fraction")
    assert r5_speed == 0.4
    assert r5 == r4


def test_r5_speed_is_below_the_one_fix_interval_innovation_gate_radius():
    cfg = _config(R5)
    ekf = EkfConfig()
    fix_sigma_m = 0.10
    gate_radius_m = (ekf.fix_gate_nis * (ekf.initial_position_sigma_m**2 + fix_sigma_m**2)) ** 0.5
    max_speed_mps = 1.0  # configs/robot/sim_reference.yaml, SYNTHETIC_ONLY
    one_second_travel_m = cfg["spatial_v1"]["runtime"]["cruise_speed_fraction"] * max_speed_mps
    assert one_second_travel_m < gate_radius_m


def test_r4_and_r5_are_registered_for_reproduction():
    assert EXPERIMENTS["M1-ACTION-SPATIAL-V1-DEV-R4"][1] == str(R4.relative_to(ROOT))
    assert EXPERIMENTS["M1-ACTION-SPATIAL-V1-DEV-R5"][1] == str(R5.relative_to(ROOT))
