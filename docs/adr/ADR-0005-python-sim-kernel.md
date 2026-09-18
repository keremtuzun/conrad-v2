# ADR-0005 Python 6-DOF simulation kernel alongside Unity V2

- Date: 2026-09-18
- Packages: conrad.sim.kernel, conrad.adapters.unity, unity/ConradUnityV2. Source: ch20-21, ch34 (CPU execution must support all tests/replay).
- Status: ACCEPTED

## Problem
Unity remains the C# robot-physics and sensor simulator, but CI, replay and CPU tests must run with no Unity editor, and this build environment cannot execute Unity.
## Decision
Both simulators implement the same RobotHardwareInterface and the same physics equations. `conrad.sim.kernel` is the headless, deterministic reference used by CI, replay, benchmarks and host-HIL. The Unity C# project is delivered as source and reaches the stack only through `conrad.adapters.unity`. The kernel is a second RHI adapter, not a new subsystem and not a competing truth layer: twins still own world truth.
## Validity
Both report simulation validity L1 (approximate physics). Unity execution and cross-validation of kernel vs Unity are BLOCKED_EXTERNAL (EXT-UNITY-01).
## Approval
Kerem: PENDING REVIEW
