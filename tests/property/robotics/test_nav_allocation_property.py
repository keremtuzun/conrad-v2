import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from conrad.robotics.allocation import ThrusterAllocator
from conrad.robotics.hardware.config import load_robot_config
from conrad.schemas.ids import IdFactory

ALLOC = ThrusterAllocator(load_robot_config("configs/robot/sim_reference.yaml"), IdFactory(seed=1))
wrench = st.lists(st.floats(-300, 300, allow_nan=False), min_size=6, max_size=6)


@settings(max_examples=60, deadline=None)
@given(wrench)
def test_commands_always_bounded_and_never_worse_than_zero(w):
    tau = np.array(w)
    r = ALLOC.solve(tau)
    assert all(-1.0 <= u <= 1.0 for u in r.commands.values())
    assert np.all(r.thrust_n <= 40.0 + 1e-6) and np.all(r.thrust_n >= -30.0 - 1e-6)
    weights = np.array(ALLOC.config.wrench_weights)
    assert np.linalg.norm(weights * (r.achieved_wrench - tau)) <= np.linalg.norm(weights * tau) + 1e-6


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-5, 5, allow_nan=False), min_size=6, max_size=6))
def test_small_wrenches_are_exact(w):
    tau = np.array(w)
    r = ALLOC.solve(tau)
    assert not r.saturated
    assert np.allclose(r.achieved_wrench, tau, atol=1e-8)
