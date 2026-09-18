# ADR-0002 Frame convention

- Date: 2026-09-18
- Packages: conrad.schemas.frames, conrad.sim, conrad.adapters.unity. Source: ch2 Coordinate frames, ch28 Time/Frames.
- Status: PROPOSED (simulation only). Physical WORLD convention remains OPEN.

## Decision
T_A_from_B maps B coordinates to A; quaternion (w,x,y,z); SI units. Simulation WORLD is right-handed, +Z up, body +X forward/+Y left/+Z up, depth = -z, declared through `FrameConvention(authority=SIMULATION_DEFAULT)`. Unity (left-handed, +Y up) is converted only inside the Unity adapter with unit-axis probe tests.
## Open
Geodetic origin, physical vertical datum and the hardware body-frame convention are OPEN until the hardware owner supplies the frame contract (EXT-HW-03). Nothing in core assumes them.
## Evidence
tests/contract/test_p0_contracts.py::test_frame_contract, test_frame_graph_round_trip; tests/contract/unity.
## Approval
Kerem: PENDING REVIEW; Burak required before any physical lane use.
