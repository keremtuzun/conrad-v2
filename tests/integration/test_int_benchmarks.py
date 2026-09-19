"""INT-001..INT-010 are registered, described and resolvable (ch20 integrated intelligence benchmarks).

Registration test only: the benchmark missions themselves are experiments (`conrad sim run --scenario INT-00x`).
"""

from __future__ import annotations

from conrad.sim.mission import scenarios


def test_all_ten_int_benchmarks_registered_and_described() -> None:
    ids = [f"INT-{i:03d}" for i in range(1, 11)]
    assert set(ids) <= set(scenarios.INT_DESCRIPTIONS)
    for sid in ids:
        world, runtime = scenarios.resolve(sid, {})
        assert isinstance(world, dict) and isinstance(runtime, dict)
