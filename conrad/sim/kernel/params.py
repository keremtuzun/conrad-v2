"""Kernel parameters. Vehicle values come from RobotConfig via ``Sourced.require`` (raise on OPEN).

Environment / numerical values that RobotConfig does not carry live in :class:`SimKernelConfig`
and are labelled SYNTHETIC_ONLY simulator inputs (never measurements).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import Field

from conrad.schemas.base import ConradModel
from conrad.schemas.robot import RobotConfig

SIM_FRAME_CONVENTION = "SIMULATION_DEFAULT: WORLD right-handed +Z up (depth = water_surface_z_m - z); body +X fwd, +Y left, +Z up"


class SimValidityLevel(str, Enum):
    L0_FUNCTIONAL = "L0"
    L1_APPROXIMATE_PHYSICS = "L1"
    L2_CHARACTERIZED = "L2"
    L3_IDENTIFIED = "L3"
    L4_VALIDATED_ENVELOPE = "L4"


class SimKernelConfig(ConradModel):
    """SYNTHETIC_ONLY environment and numerical settings of the Python kernel."""

    label: str = "SYNTHETIC_ONLY"
    physics_dt_s: float = Field(default=0.01, gt=0)
    water_density_kgm3: float = Field(default=1025.0, gt=0)
    gravity_mps2: float = Field(default=9.80665, gt=0)
    water_surface_z_m: float = Field(
        default=0.0, description="WORLD z of the surface; pressure depth = s - z"
    )
    thrust_noise_fraction: float = Field(default=0.01, ge=0)
    thruster_power_w_per_n15: float = Field(default=6.0, ge=0, description="P = c |T|^1.5 (synthetic)")
    hotel_power_w: float = Field(default=25.0, ge=0)
    imu_gyro_noise_rps: float = Field(default=0.002, ge=0)
    imu_orientation_noise_rad: float = Field(default=0.005, ge=0)
    imu_provides_orientation: bool = True
    report_thruster_faults: bool = True
    collision_restitution: float = Field(default=0.0, ge=0, le=1)


@dataclass(frozen=True)
class ThrusterParams:
    thruster_id: str
    position: np.ndarray
    direction: np.ndarray
    max_forward_n: float
    max_reverse_n: float
    deadzone: float
    tau_s: float
    latency_s: float
    k: float


@dataclass(frozen=True)
class VehicleParams:
    mass: float
    volume: float
    r_g: np.ndarray
    r_b: np.ndarray
    inertia: np.ndarray
    dims: np.ndarray
    d_lin: np.ndarray
    d_quad: np.ndarray
    added_mass: np.ndarray
    thrusters: tuple[ThrusterParams, ...]
    battery_capacity_j: float
    battery_voltage_v: float
    battery_max_current_a: float

    @property
    def allocation_matrix(self) -> np.ndarray:
        cols = [np.concatenate([t.direction, np.cross(t.position, t.direction)]) for t in self.thrusters]
        return np.stack(cols, axis=1)


def _vec(values: tuple[float, ...], n: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"RobotConfig parameter {name!r} must have {n} entries, found {arr.shape}")
    return arr


def vehicle_params_from_config(config: RobotConfig) -> VehicleParams:
    """Extract every physical parameter; an OPEN one raises ``OpenParameterError``."""
    if not config.thrusters:
        raise ValueError("RobotConfig has no thrusters; the simulator cannot actuate the vehicle")
    thrusters = []
    for t in config.thrusters:
        p = f"thrusters[{t.thruster_id}]"
        d = _vec(t.direction_body.require(f"{p}.direction_body"), 3, f"{p}.direction_body")
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            raise ValueError(f"{p}.direction_body has zero norm")
        thrusters.append(
            ThrusterParams(
                thruster_id=t.thruster_id,
                position=_vec(t.position_body_m.require(f"{p}.position_body_m"), 3, f"{p}.position"),
                direction=d / norm,
                max_forward_n=float(t.max_forward_thrust_n.require(f"{p}.max_forward_thrust_n")),
                max_reverse_n=float(t.max_reverse_thrust_n.require(f"{p}.max_reverse_thrust_n")),
                deadzone=float(t.deadzone_command.require(f"{p}.deadzone_command")),
                tau_s=float(t.time_constant_s.require(f"{p}.time_constant_s")),
                latency_s=float(t.latency_s.require(f"{p}.latency_s")),
                k=float(t.thrust_coefficient.require(f"{p}.thrust_coefficient")),
            )
        )
    return VehicleParams(
        mass=float(config.mass_kg.require("mass_kg")),
        volume=float(config.displaced_volume_m3.require("displaced_volume_m3")),
        r_g=_vec(config.center_of_mass_body_m.require("center_of_mass_body_m"), 3, "center_of_mass"),
        r_b=_vec(config.center_of_buoyancy_body_m.require("center_of_buoyancy_body_m"), 3, "cob"),
        inertia=_vec(config.inertia_diag_kgm2.require("inertia_diag_kgm2"), 3, "inertia_diag_kgm2"),
        dims=_vec(config.dimensions_m.require("dimensions_m"), 3, "dimensions_m"),
        d_lin=_vec(config.linear_drag.require("linear_drag"), 6, "linear_drag"),
        d_quad=_vec(config.quadratic_drag.require("quadratic_drag"), 6, "quadratic_drag"),
        added_mass=_vec(config.added_mass_diag.require("added_mass_diag"), 6, "added_mass_diag"),
        thrusters=tuple(thrusters),
        battery_capacity_j=float(config.battery.capacity_j.require("battery.capacity_j")),
        battery_voltage_v=float(config.battery.nominal_voltage_v.require("battery.nominal_voltage_v")),
        battery_max_current_a=float(config.battery.max_current_a.require("battery.max_current_a")),
    )
