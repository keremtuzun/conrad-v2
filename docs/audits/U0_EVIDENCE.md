# Gate U0 evidence (Unity V2.0)

Date: 2026-09-19. Spec: Phase 4 "Unity V2.0 in parallel" / "Unity Gate U0", ch21 development progression V2.0 and
Unity gates U-D1/D2/D5. Machine-readable record: `artifacts/gates/U0/u0_summary.json`; test output:
`artifacts/gates/U0/pytest_output.txt`; build log: `artifacts/gates/U0/unity_build.log`.

## What was run

| Item | Value |
|---|---|
| Unity editor | 6000.5.9f1 (revision b57deb96f08d); project upgraded from 2022.3.20f1 (`ProjectVersion.txt`) |
| Compile | batch mode, 0 `error CS`, 0 `warning CS` (the build log shows a full script recompile) |
| Build | `BuildScript.BuildWindows64Player`, Windows64 Mono, result Succeeded, 0 errors, 0 warnings |
| Build log sha256 | `f28cd7ed36b22ca29d9f403e5f69af6b491ee16243271a345e42197624021c62` |
| Player sha256 | `36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277` (`Builds/Win64/ConradSim.exe`, gitignored) |
| Player mode | `-batchmode` without `-nographics`; graphics device Direct3D11 (AMD Radeon 890M) |
| Transport | length-prefixed JSON over loopback TCP (`TcpBridgeTransport` / `TcpBridgeServer`), lock-step |
| Git HEAD at run time | `6824356bb74f27e5f8fa13ca737bb3ab6144a6e7`. The working tree held this workstream's uncommitted changes. |
| Tests | `python -m uv run pytest tests/unity_live -v -rA`: **23 passed** in 96 s |
| Validity level | **L1_APPROXIMATE_PHYSICS**. Every physical parameter is SYNTHETIC_ONLY (`configs/robot/sim_reference.yaml` and labelled variants). No real-world measurement is involved. |

The scenario water density is 1022.22 kg/m^3 (= m / V of `sim_reference`), a SYNTHETIC_ONLY scenario value
chosen so that the reference configuration is exactly neutral. The analytic references use the parameters the
player loaded: 1-DOF `m_eff v' = F - d1 v - d2 v|v|`, integrated with the player's step order (5 ms,
semi-implicit Euler), and with the thruster latency and first-order lag where thrust is involved.

## Criteria

| U0 criterion | Test(s) | Measured | Declared tolerance | Result |
|---|---|---|---|---|
| Stable physics, 6-DOF rigid body | combined 6-DOF wrench 30 s, then 30 s coast | all finite; max 0.486 m/s, 0.884 rad/s; quaternion norm error 1.1e-16; final 2.1e-4 m/s, 1.3e-3 rad/s | < 2 m/s, < 3 rad/s; at rest < 0.02 | PASS |
| Known command moves robot correctly | allocated pure wrench per DOF, 0.8 s | surge +0.320 m/s, sway +0.230 m/s, heave +0.213 m/s, roll +0.497 rad/s, pitch +0.497 rad/s, yaw +0.770 rad/s; cross-axis 0.0; world displacement in the Conrad sign (+X fwd, +Y left, +Z up) | correct sign; cross-axis < 10 % | PASS |
| Gravity/buoyancy equilibrium | neutral 20 s; +2 % volume; +2 % mass | neutral: depth change 0.0 m, pitch 0.0 deg. Positive: +3.4114 m vs reference +3.4113 m, v = 0.186434 vs analytic 0.186436 m/s. Negative: -3.4078 m vs -3.4085 m, same speed | 1 mm / 0.001 deg; trajectory and terminal velocity within 1 % | PASS |
| Basic drag (terminal velocity) | 20 N surge, 15 s | 0.944136 m/s vs analytic 0.944139 m/s (rel. 2.7e-6); whole velocity history within 2.7e-6 of the reference | 0.5 % (terminal), 1 % (history) | PASS |
| Mass/inertia/CoM/CoB from RobotConfig; OPEN refused | yaw 0.4 N m for Izz 0.16 vs 0.32; CoB +5 mm fwd; CoM 5 mm aft; OPEN mass | yaw-rate history error 7.7e-7 / 1.2e-7 (0.357 vs 0.266 rad/s at 0.5 s); trim 14.0362 deg vs expected 14.0362 deg for both offsets; OPEN mass: Python export raises `OpenParameterError`, player exits with code 3 and logs "Unity refuses OPEN physical parameters: mass_kg" | 1 % of peak rate; 0.5 deg | PASS |
| Sensor frames and timestamps | 2 s at rest + 1 s yaw-left, 20 ms steps | IMU 299 pkts, period 10 ms, latency 10 ms; depth 59 / 50 ms; camera 14 / 200 ms; sonar 5 / 500 ms; all strictly increasing, consecutive sequence, on the physics grid, clock SIM, frame ids equal RobotConfig. IMU at rest (-0.00028, -0.00024, 9.8071) m/s^2; yaw-left gyro z +0.486 rad/s; depth mean 10.0019 m; camera 48x64x3 on Direct3D11, bottom quarter 253 dn (seafloor) vs top 51 dn (water) | 5 sigma/sqrt(n); gyro z > 0.05; bottom > top + 50 | PASS |
| Deterministic seed/reset, replay | same seed twice (reset), same seed in a second player process, other seed; wire record then replay | trajectory difference 0.0 (reset) and 0.0 (new process) over 100 samples; IMU/depth payload digests identical; other seed gives different sensor digests; wire replay of 30 steps identical, including acquisition timestamps | 1e-9 | PASS |
| Frame conversion probes | `FRAME_PROBE` through an engine Transform; sonar and scene placement; heightfield contact | +X->(0,0,1), +Y->(-1,0,0), +Z->(0,1,0); yaw-left forward -> Unity -X, pitch+ forward -> -Y, roll+ up -> +X; round trip 4.8e-8 m, 1.7e-8 (quaternion). Wall 5.78 m vs 5.75 m expected; pole on the left at bearing 0.367 vs 0.353 rad, 4.84 m vs 4.81 m, not mirrored; vehicle rests on heightfield at -11.875 m vs -11.875 m expected | 1e-6 (engine float32); 1 bin (0.3125 m) or 2 bins; 3 cm | PASS |
| RobotHardwareInterface through the real CommandGateway | `CommandMode.SIMULATED`, `ExecutionLane.SIMULATION` | command accepted (adapter `unity_v2`); duplicate, wrong-digest and unauthorized refused (`DUPLICATE_COMMAND_ID`, `WRONG_ROBOT_CONFIG_DIGEST`, `MISSING_SAFETY_AUTHORIZATION`); vehicle moved 1.75 m forward in 3 s; thruster commands echoed; battery 0.99981; health OK; IMU/depth/camera/sonar served | functional | PASS |

## Honest notes

- The first full live run had one failure, in the test and not in the simulator. The scene test put the "pole"
  at bearing 0.675 rad, outside the sonar fan used in the test (+-0.5 rad). The pole was moved inside the fan
  and the bearing is now asserted to lie in the fan. The run recorded above is a fresh full run after that fix.
- Two simulator defects found while writing U0 were fixed in C#. (1) A RESET with the seed already in use did not
  restart the sensor, thruster or current random streams, so resets were not deterministic. Every reset now
  rebuilds the seeded subsystems. (2) The first sensor sample after a reset came one physics step after reset
  instead of on the period grid. Samples now fall at k * period.
- Agreement with the references is at the 1e-5..1e-7 level because the references integrate the same equations
  with the same step order. This proves that Unity applies the configured model as specified. It says **nothing** about
  real-vehicle fidelity. Validity stays L1.
- Not covered by U0 and not claimed: currents, noise statistics, fault behaviour in the player, power-model
  accuracy, free-running (non-lock-step) mode against the real player, Twin 2S geometry wiring, and targets
  other than Windows64 Mono.
- `-nographics` is not supported with a camera: the camera sensor refuses to start, and the player exits with code 3.
  No raycast depth-image fallback was needed on this host, so none is implemented.
