# Spatial V1 development performance

This is a software profile, not gate evidence or physical calibration. Reproduce
with `python -m uv run python scripts/profile_spatial_mission.py
artifacts/software_completion/<new-directory> --duration-s 24`. The script
refuses to overwrite a run, records one JSON row per completed mission, and
keeps raw `cProfile` statistics beside the run bundles.

The first 24-second comparison used seed 2026201, `pipeline_with_supports`,
0.1-second control period, and the same legacy versus spatial mission setup
except for the opt-in structural truth, sensor, and Model2T backends. Another
clean-checkout `pytest -q` was running concurrently, so wall times are
contended development observations rather than an unloaded performance claim.
Python tracing and profiling were enabled for both runs.

| Backend | Prepare wall s | Run wall s | Finish wall s | Peak traced Python MiB | Bundle MiB | Observations | Evidence | Belief revisions | Commands |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy | 2.44 | 36.24 | 1.67 | 21.75 | 4.83 | 65 | 41 | 309 | 240 |
| Spatial V1 | 2.15 | 34.88 | 1.85 | 20.85 | 4.75 | 64 | 40 | 302 | 240 |

The spatial profile called `SpatialMissionModel2T.receive_context` 81 times
(1.12 s cumulative) and `update_beliefs` 11 times (0.65 s cumulative).
`capsule_visibility_certificate` and `pose_visible_capsule_supports` each ran
24 times (about 0.02 s cumulative each). This mission did not invoke an MCBR
planning cycle, so it does not measure spatial candidate scoring. A longer
closed-loop inspection with actual planning requests is required before
performance acceptance. No limit or pass threshold is inferred from these two
runs.

A 120-second spatial development mission on the same seed, run concurrently
with the clean-checkout regression, generated 4 actual MCBR plans (48
candidates each), 262 observations, 142 Evidence rows, 1,730 belief revisions,
and 1,200 commands. The bundle was 23.55 MiB and peak traced Python allocation
was 84.36 MiB. Profiled wall time was 200.20 s. The four planner calls took
16.26 s cumulatively under `cProfile` plus `tracemalloc`; 192 surface
predictive information calculations took 6.68 s. These times include profiler
overhead and competing CPU load.

The four plans were flown, but the target earned **zero** structural
observations. An off-axis flight pose reproduced the cause: the old support
producer returned immediately when the centre ray missed the cylinder, even
when the pipe was inside the payload field of view. The producer now considers
the near surface for such poses only when the continuous visibility certificate
admits the resulting resolution rectangles. A unit and mission sensor
regression pin this case. The 120-second profile is a **failed pre-repair
development run**, not evidence of closed-loop spatial mission success. A
post-repair mission and unloaded planner timing are still required.

After the planner's local aim-region repair, a 180-second seed-2026201
development profile ran without a competing test suite. Its 1,800 commands,
637 observations, 457 Evidence rows, and 2,680 belief revisions produced a
39.32 MiB bundle and 125.75 MiB peak traced Python allocation. Profiled
mission wall time was 287.56 s, followed by 8.85 s to finish the bundle.
Seven MCBR plans were made and all seven views were fully flown. Planner
calls took 43.63 s cumulative under `cProfile` and `tracemalloc`; surface
predictive `expected_information` took 23.00 s and spatial support production
8.97 s cumulatively. The resulting target remained `UNKNOWN` because the
selected views did not cover the required angular crown. This profile does
not demonstrate mission completion or a performance PASS; it identifies the
predictive computation as the largest measured spatial planning cost.

A later 80-second profile used the completed healthy mission configuration
and `scripts/profile_spatial_bundle.py` at branch `6ddff15` (the run itself
preceded that commit, with the same runtime implementation). Without
allocation tracing, 800 mission ticks took 42.81 process CPU seconds and
49.78 wall seconds while pytest also ran. Three actual MCBR plans took 5.42 s
total; 80 spatial sensor support/response calls took 1.40 s, 51 spatial
Model2T ingests 0.030 s, and 51 updates 1.01 s. There were 1,096 belief
revisions occupying 3,474,536 JSON payload bytes; the bundle was 17,347,658
bytes. An otherwise matched allocation-traced run peaked at 40,482,797
traced Python bytes but inflated wall time to 159.52 s and planner time to
26.73 s. The untraced result replaces those traced timings for development
cost assessment. Unity step overhead and scaling beyond this 80-second window
are still unmeasured.

The later detectability-v2 healthy closed-loop mission ran 4,800 kernel
steps over 480 simulated seconds on this Windows host while the ordinary
clean-checkout suite also ran. Its steps used 480.55 process CPU seconds and
553.05 wall seconds. The 480 sensor support/response calls took 13.39 s;
387 spatial Model2T ingests took 0.14 s and 387 updates took 25.83 s; 15
actual MCBR plans took 25.83 s. It persisted 8,001 belief revisions with
24,767,104 payload bytes, and the replay bundle used 119,730,814 bytes.
These are development cost measurements under host contention, not formal
latency bounds.

The matched 40-step development probe at seed 401 used the same stored
spatial world/runtime options for kernel and Unity. The kernel's measured
step wall time was 1.899 s total (47.5 ms mean, 398.7 ms p95), and Unity's
was 1.554 s total (38.8 ms mean, 122.2 ms p95). Unity startup/prepare took
3.689 s versus 0.773 s for kernel. The probe is short and includes a
different backend startup cost; the lower Unity step mean in this sample is
not a general throughput guarantee. It shows no catastrophic Unity step
overhead in this controlled development slice.
