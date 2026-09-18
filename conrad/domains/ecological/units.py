"""Unit handling for 2E evidence. Units travel with every value; mismatches fail closed.

Compound strings look like ``cover_fraction[1];offset_m[m,m,m]``: named segments, each with one
unit per value, in payload order.

implementation_status: EXPERIMENTAL_CANDIDATE
"""

from __future__ import annotations

import re

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[([^\]]*)\]$")

UNIT_ALIASES: dict[str, frozenset[str]] = {
    "degC": frozenset({"degC", "C", "celsius"}),
    "NTU": frozenset({"NTU", "ntu"}),
    "m s-1": frozenset({"m s-1", "m/s"}),
    "W m-2": frozenset({"W m-2", "W/m2"}),
    "1": frozenset({"1", "unitless", "fraction"}),
    "m": frozenset({"m"}),
}


class UnitError(ValueError):
    """A value arrived with missing, unparseable or incompatible units."""


def same_units(found: str, expected: str) -> bool:
    return found == expected or found in UNIT_ALIASES.get(expected, frozenset())


def require_units(found: str | None, expected: str, what: str) -> None:
    if found is None or not same_units(found, expected):
        raise UnitError(f"{what}: units {found!r} are not compatible with {expected!r}")


def parse_compound(units: str) -> list[tuple[str, tuple[str, ...]]]:
    """``'a[1];b[m,m,m]'`` -> ``[('a', ('1',)), ('b', ('m', 'm', 'm'))]``."""
    out: list[tuple[str, tuple[str, ...]]] = []
    for raw in units.split(";"):
        m = _SEGMENT.match(raw.strip())
        if m is None:
            raise UnitError(f"unparseable compound unit segment {raw!r} in {units!r}")
        parts = tuple(p.strip() for p in m.group(2).split(","))
        if any(not p for p in parts):
            raise UnitError(f"empty unit in segment {raw!r}")
        out.append((m.group(1), parts))
    return out


def split_compound(
    values: tuple[float, ...], units: str
) -> dict[str, tuple[tuple[float, ...], tuple[str, ...]]]:
    """Map each named segment to its values and units. Length must match exactly."""
    segments = parse_compound(units)
    need = sum(len(u) for _, u in segments)
    feature_segments = [name for name, u in segments if name == "feature"]
    if feature_segments and len(values) >= need:
        # 'feature[1]' is a variable-length vector: it absorbs the values the other segments leave.
        fixed = sum(len(u) for name, u in segments if name != "feature")
        width = len(values) - fixed
        segments = [(n, u * width if n == "feature" else u) for n, u in segments]
        need = len(values)
    if need != len(values):
        raise UnitError(f"{len(values)} values but units {units!r} declare {need}")
    out: dict[str, tuple[tuple[float, ...], tuple[str, ...]]] = {}
    i = 0
    for name, u in segments:
        out[name] = (tuple(values[i : i + len(u)]), u)
        i += len(u)
    return out
