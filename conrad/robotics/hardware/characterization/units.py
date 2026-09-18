"""Unit validation and conversion to SI for characterization records.

Only exact, tabulated conversions are applied. A unit that is not in the table is accepted only when
it is spelled exactly like the target parameter's declared units (pass-through, no conversion).
Conversions that need extra facts (mAh -> J needs a voltage) are refused rather than guessed.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

from collections.abc import Sequence

# unit -> (SI unit, scale, offset):  si = value * scale + offset
_TABLE: dict[str, tuple[str, float, float]] = {
    # mass
    "kg": ("kg", 1.0, 0.0),
    "g": ("kg", 1e-3, 0.0),
    "lb": ("kg", 0.45359237, 0.0),
    # length
    "m": ("m", 1.0, 0.0),
    "cm": ("m", 1e-2, 0.0),
    "mm": ("m", 1e-3, 0.0),
    "in": ("m", 0.0254, 0.0),
    # volume
    "m^3": ("m^3", 1.0, 0.0),
    "L": ("m^3", 1e-3, 0.0),
    "mL": ("m^3", 1e-6, 0.0),
    "cm^3": ("m^3", 1e-6, 0.0),
    # inertia
    "kg*m^2": ("kg*m^2", 1.0, 0.0),
    "g*mm^2": ("kg*m^2", 1e-9, 0.0),
    "g*cm^2": ("kg*m^2", 1e-7, 0.0),
    # force / thrust
    "N": ("N", 1.0, 0.0),
    "kgf": ("N", 9.80665, 0.0),
    "lbf": ("N", 4.4482216152605, 0.0),
    # time
    "s": ("s", 1.0, 0.0),
    "ms": ("s", 1e-3, 0.0),
    "us": ("s", 1e-6, 0.0),
    # frequency / rates
    "Hz": ("Hz", 1.0, 0.0),
    "kHz": ("Hz", 1e3, 0.0),
    "m/s": ("m/s", 1.0, 0.0),
    "cm/s": ("m/s", 1e-2, 0.0),
    "knot": ("m/s", 0.514444, 0.0),
    "bps": ("bps", 1.0, 0.0),
    "bit/s": ("bps", 1.0, 0.0),
    "kbps": ("bps", 1e3, 0.0),
    "Mbps": ("bps", 1e6, 0.0),
    # energy / electrical
    "J": ("J", 1.0, 0.0),
    "Wh": ("J", 3600.0, 0.0),
    "kWh": ("J", 3.6e6, 0.0),
    "V": ("V", 1.0, 0.0),
    "mV": ("V", 1e-3, 0.0),
    "A": ("A", 1.0, 0.0),
    "mA": ("A", 1e-3, 0.0),
    "W": ("W", 1.0, 0.0),
    # temperature (safety.thermal_limit uses C in RobotConfig; kept as C)
    "C": ("C", 1.0, 0.0),
    "K": ("C", 1.0, -273.15),
    # angle
    "rad": ("rad", 1.0, 0.0),
    "deg": ("rad", 3.141592653589793 / 180.0, 0.0),
    # dimensionless
    "unitless": ("unitless", 1.0, 0.0),
    "fraction": ("unitless", 1.0, 0.0),
    "%": ("unitless", 0.01, 0.0),
    # memory
    "B": ("B", 1.0, 0.0),
    "MB": ("B", 1e6, 0.0),
    "GB": ("B", 1e9, 0.0),
    "MiB": ("B", 2.0**20, 0.0),
    "GiB": ("B", 2.0**30, 0.0),
}

# RobotConfig spells some SI units differently; these are equivalent spellings, not conversions.
_EQUIVALENT = {
    "kg*m^2": {"kg*m^2", "kg m^2"},
    "unitless": {"unitless", "fraction", "unit"},
    "B": {"B", "bytes"},
    "C": {"C", "degC"},
    "bps": {"bps", "bit/s"},
}

_REFUSED = {"mAh": "mAh -> J needs the pack voltage; submit J or Wh", "Ah": "Ah -> J needs the pack voltage"}


class UnitError(ValueError):
    pass


def si_unit(units: str) -> str | None:
    entry = _TABLE.get(units)
    return None if entry is None else entry[0]


def same_units(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(a in group and b in group for group in _EQUIVALENT.values())


def convert_scalar(value: float, units: str, target_units: str) -> float:
    if units in _REFUSED:
        raise UnitError(_REFUSED[units])
    entry = _TABLE.get(units)
    if entry is None:
        if units == target_units:
            return value
        raise UnitError(f"unit {units!r} is unknown and differs from the target units {target_units!r}")
    si, scale, offset = entry
    if not same_units(si, target_units):
        raise UnitError(f"{units!r} is a {si!r} quantity; the target expects {target_units!r}")
    return value * scale + offset


def convert_value(value: float | Sequence[float], units: str, target_units: str) -> float | tuple[float, ...]:
    """Scalar or vector conversion. Vectors share one unit for every component."""
    if isinstance(value, int | float):
        return convert_scalar(float(value), units, target_units)
    return tuple(convert_scalar(float(v), units, target_units) for v in value)


def convert_sigma(sigma: float | None, units: str, target_units: str) -> float | None:
    """Uncertainties scale but never shift (a K offset must not be added to a 1-sigma)."""
    if sigma is None:
        return None
    entry = _TABLE.get(units)
    if entry is None:
        convert_scalar(0.0, units, target_units)  # same validation, raises if incompatible
        return sigma
    convert_scalar(0.0, units, target_units)
    return abs(sigma * entry[1])
