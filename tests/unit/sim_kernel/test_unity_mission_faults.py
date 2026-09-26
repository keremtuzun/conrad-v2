"""Mission-level Unity fault aliases preserve the kernel scenario semantics."""

from types import SimpleNamespace

from conrad.adapters.unity import FaultType
from conrad.sim.mission.options import FaultSpec
from conrad.sim.mission.unity_world import LOW_POWER, UnityMissionWorld


class _Hardware:
    def __init__(self) -> None:
        self.requests = []

    def now_ns(self) -> int:
        return 30_000_000_000

    def inject_fault(self, request):
        self.requests.append(request)
        return SimpleNamespace(accepted=True, reason_codes=())


def test_low_power_maps_to_unity_battery_degradation() -> None:
    world = object.__new__(UnityMissionWorld)
    world.scenario_id = "I5-BATTERY-RESERVE"
    world.hardware = _Hardware()
    world.suite = SimpleNamespace(dynamic_fix_outages=[])
    world._pending_faults = [FaultSpec(t_s=30.0, type=LOW_POWER, magnitude=0.12)]
    world._pending_eco = []
    world.t2e = None

    fired = world.due_faults()

    assert len(fired) == 1
    assert fired[0]["type"] == LOW_POWER
    assert world._pending_faults == []
    assert len(world.hardware.requests) == 1
    request = world.hardware.requests[0]
    assert request.fault_type is FaultType.BATTERY_DEGRADATION
    assert request.magnitude == 0.12
    assert request.start_time_ns == 30_000_000_000
