# I5 LOW_POWER post-formal software correction

Status: **SOFTWARE CORRECTION DEVELOPMENT PASS; FORMAL V1.1 REMAINS FAIL**

The sealed Spatial V1.1 Unity formal cycle on worlds 8701000 and 8701001
remains an immutable 7/10 FAIL. It was not rerun, supplemented with unused
final worlds, or reinterpreted after this correction.

## Root cause

The Python mission kernel defines `LOW_POWER(magnitude)` as an instantaneous
state event that consumes enough energy to leave at most `magnitude` battery.
The Unity mission bridge incorrectly translated that event to
`BATTERY_DEGRADATION`, which only scaled capacity. Unity accepted the event but
reported about 0.96 battery at mission end, so the battery-reserve warrant was
never reached.

The correction adds a distinct wire/Unity `LOW_POWER` type and a Unity
`PowerModel.SetRemainingFraction` operation. It clamps the requested fraction,
increases `EnergyUsedJ` monotonically, and never restores already-used energy.
`BATTERY_DEGRADATION` retains its independent capacity-scaling semantics.

## Verification

- Python/C# contract, mission-fault, and bridge tests: 47 passed.
- Unity 6000.5.9f1 build: succeeded with zero warnings and zero errors.
- native macOS player SHA-256:
  `c52be937690a5d3d5be03f4acc3fc06842dd17ce3f53000e34f6daa528ab71e9`.
- declared development partition: `i5_unity_v3`, world 8701100 only.
- scenario/arm: `I5-BATTERY-RESERVE` / `egdc_structured`.
- first post-fault battery: `0.11995199236341447`.
- warrant reached: true.
- chosen at onset: `RETURN_TO_SAFE_STATE:`.
- action correctness: true.
- decision violations: 0.
- development verification artifact SHA-256:
  `24ad18a1748ddc92076d09f20eef2eda87995aafdb768bd5084b54ff91efab27`.

No final or validation partition was opened for this post-result correction.
Per the frozen I5 stop rule, this repair does not create a new formal verdict.
Any future formal replication would require a separately versioned,
prospectively declared cycle on entirely fresh worlds.
