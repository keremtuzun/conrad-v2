# Gate I6: multi-domain intelligence (surrogate PASS, formal Unity run prepared)

Date: 2026-09-19. Spec: ch25 "Integration Gate I6 - Multi-domain Intelligence". Registry criteria (`conrad/evaluation/gates.py`):

1. one mission produces 2S, 2T and 2E beliefs;
2. Model1 reasons across all three via the Belief Bus;
3. children remain authoritative within domains.

## Status

| Evidence | Status | File |
|---|---|---|
| Surrogate (python kernel, ADR-0008) | **PASS**, 3 of 3 criteria on 3 of 3 held-out worlds | `artifacts/gates/I6/evidence_surrogate.json`, `artifacts/experiments/I6-MULTIDOMAIN-E001/i6_e001.json` |
| Formal (Unity) | NOT_RUN. Needs a rebuilt player. The harness is ready. | `tests/unity_live/test_i6_unity.py` |
| Official (`conrad gates status`) | BLOCKED_UPSTREAM (I5, which is blocked by I4) | |

A surrogate PASS never promotes the gate.

## Design

### Scenario (`conrad/sim/mission/scenarios.py`, appended)

`I6-MULTIDOMAIN-TURBID` and `I6-MULTIDOMAIN-CLEAR` run on the same world, with Twin2S, Twin2T and Twin2E, for 120 s.

- **TURBID.** A Twin2E turbidity spike (+25 NTU, 30 m radius) fires at 15 s, during the lane survey.
- **CLEAR.** The same world without the spike. This is the paired control: only the 2E belief differs between the arms, so any change in Model1's choice comes from 2E.

### Model1 across the three domains (`conrad/orchestration/multidomain.py`, new)

The feature is off by default (`MissionRuntimeConfig.multidomain.enabled = False`), so every other gate's mission is unchanged. When it is on:

1. **Imaging-conditions requirement.**
   - For each critical TECHNICAL inspection requirement, the runtime adds one ECOLOGICAL requirement at the same inspection station.
   - Its primary belief is Model2E's turbidity field belief (properties `turbidity.mean`, `turbidity.sd`). Its context domain is SPATIAL.
   - Its consequence (0.2) is below `consequence_matters_above` (0.3), so it never triggers sensing actions on its own.
   - Its effect: every EGDC claim graph now holds grounded 2E claims next to the 2T claims and the 2S coverage context. All three come from Belief Bus snapshots, through `query_model2s`, `query_model2t` and `query_model2e`.
2. **Sensing-conditions gate.**
   - EGDC may send an information request on a critical component to MCBR. Before that happens, the gate reads the 2E turbidity claims in that decision's own claim graph.
   - It computes beam transmission `exp(-(c0 + c1 * T_up) * r)`, where:
     - `T_up = mean + 2 sd` is the upper credible turbidity;
     - `c0` and `c1` are Model2E's optical constants;
     - `r = 1.0 m` is the closest feasible inspection range.
   - If the transmission is below 0.5, the request is **DEFERRED_LOW_VISIBILITY**:
     - it is not sent to MCBR;
     - it is marked not executed, so it does not use up EGDC's attempt budget;
     - it is written as an `ACTION_REJECTED` event;
     - it is written as a DECISION provenance record whose parent is the EGDC decision.
   - Each record cites the TECHNICAL claim on the target belief, the SPATIAL coverage claims, and the ECOLOGICAL turbidity claims, with their belief IDs.
   - All thresholds are ENGINEERING_ESTIMATE values.
3. **The gate writes no beliefs.** It reads only the EGDC decision record.

**Why the rule is not inside EGDC.** `conrad/decision/*` and `deliberation.py` belong to the I5 workstream, so I did not edit them. The gate reads only the decision record, so it can move into EGDC's constraint or candidate stage unchanged. Until then, the deferral is an orchestration rule applied to Model1's claim graph, not an EGDC candidate action. **Gap for the I5 owner.**

### Orchestration hooks (minimal, additive)

| File | Change |
|---|---|
| `conrad/orchestration/mission_config.py` | new field `multidomain: MultiDomainSettings` (defaults off) |
| `conrad/orchestration/multidomain_settings.py` | new, settings only (avoids an import cycle) |
| `conrad/orchestration/routing.py` | attribute `sensing_gate = None`; 3 lines at the top of `_inspect` that consult it |
| `conrad/orchestration/mission.py` | `_enable_multidomain()`, called only when enabled |
| `conrad/orchestration/__init__.py` | one metadata assumption line |

The Belief Bus itself (`belief_bus.py`) is **not** changed.

### Unity support for Twin2E

On the kernel, Twin2E drives two sensor paths (`conrad/sim/mission/sensing.py`):

- structural-reading quality: `turbidity` and `biofouling_cover` from `observability_modifiers`;
- the environmental probe and the ecological survey.

The Unity path now keeps both, rendered in Python at the Unity TRUE pose by the same `MissionSensorSuite`. It also declares `environmental_probe` and `ecological_survey` when Twin2E is present. Unity itself renders two more things, converted in `conrad/sim/unity/eco_scene.py` (new):

- **`optics_grid` scene primitive.**
  - Beam attenuation `c = c_clear + c_ntu * turbidity`, using Twin2E's own `ObservationConfig` constants, on a 1 m grid covering the world bounds plus 4 m.
  - The RGB camera ray-casts each pixel and blends it toward the water colour with transmission `exp(-integral of c)`, integrated by the midpoint rule.
  - The grid is refreshed through an incremental `CONFIGURE_SCENE` (`replace=false`) every 1 s of sim time, and immediately after an ecological event.
  - Without a grid, the camera keeps the old uniform proxy with byte-identical output, so I1 to I4 are unaffected.
- **`fouling_cover` on box and capsule primitives.** The mean Twin2E cover over the structure's surface samples, shown as a colour on the visual only. Colliders are unchanged, as on the kernel.
- **Not attenuated:** the range imager and the sonar. This keeps parity with the kernel's Twin2S geometric channel.
- **Currents:** a constant `current_mps` is now passed to Unity's existing `environment.current` (`constant`). The I6 scenarios do not use currents. This path is untested on Unity.
- **Still refused on the Unity path:** scheduled faults.

C# changes:

- `EnvironmentInteraction/EcologyScene.cs` (new, with a `.meta`): `OpticsField`, with trilinear interpolation clamped at the faces, identical to the Python `trilinear`, and `EcologySceneBuilder`.
- `SceneGeometry.cs`: the `optics_grid` case; incremental optics replacement; `ApplyFouling` on boxes and capsules.
- `ImagingSensors.cs`: per-pixel transmission in `CameraSensor`. It adds `optics_grid`, `centre_transmission` and `mean_transmission` to the camera context, and only when a grid is loaded.
- `ExperimentRuntime.cs`: passes `OpticsField.Current` to the camera.

**Compile check.** I did not run the Unity Editor. The runtime sources (without Editor/ and Transport/) compile with Roslyn from the .NET 9 SDK against the player's own `Managed/` Unity and BCL assemblies: 0 errors, 0 warnings. That check is not a Unity build.

### Conversion error (measured, development worlds 7810000-7810002, no player)

| Quantity | t = 0 | Just after the spike (15 s) | Spike + 60 s |
|---|---|---|---|
| attenuation, max abs error (1/m), 400 random points | 2.8e-17 | 4.5e-7 to 5.2e-7 | 1.4e-4 to 2.6e-4 |
| 4 m ray transmission, max abs error, 100 random rays | 0.0 | 1.6e-8 to 1.8e-8 | 3.3e-6 to 4.4e-6 |

- The grid error is tiny because the Twin2E turbidity field is itself a coarse trilinear grid, smooth at the 1 m scale. The remaining error is advection detail, plus the 1e-6 rounding in the wire format.
- The grid is 29 x 21 x 16 cells, and one optics update is 58,650 bytes of JSON. It takes about 0.5 s to build in Python.
- **Biofouling is the approximate part.** Each structure has one cover value, but Twin2E cover varies inside a structure: max absolute error 0.11 to 0.33, mean 0.002 to 0.025. Fouled structures per world: 2 to 5 of 10.

The camera's use of the grid is checked only in the formal test (`test_twin2e_reaches_the_unity_camera`).

## Worlds and protocol

- **Declared first.** `configs/eval/i6_multidomain.yaml` was written before any final run:
  - final worlds: unity_gate final_test 7800014, 7800015, 7800016 (the next free worlds after I4's 7800002-7800013);
  - the partition file was not edited.
- **Development.** All design work used unity_gate development worlds 7810000-7810005 only.
- **Changes made on development worlds, before any final run** (both are recorded in the config header):
  - **Upper credible turbidity.** The first gate used the mean alone. On 7810000 it returned PROCEED, but the true transmission at the target was 0.32. The published 2E field belief is a whole-grid mean: 11 to 15 NTU there, against about 27 NTU true at the target. The gate now uses mean + 2 sd.
  - **Authority check.** It was first "bus head == the child's latest revision". It failed on every development run with 5 TECHNICAL heads, because Model2T commits CONTEXT revisions that are never published, so the bus lags the child. The rule now forbids only a bus head that is not a revision its own child committed, and it reports the lag.
- **Decision rule** (in the config): a criterion passes iff it passes on all 3 worlds, with both arms per world.

## Surrogate results (I6-MULTIDOMAIN-E001, final worlds, one run)

| World | Arm | DIRECT revisions 2S / 2T / 2E | Decisions citing all three | Gate verdicts | Truth agreement | MCBR plans |
|---|---|---|---|---|---|---|
| 7800014 | TURBID | 655 / 21 / 96 | 60 of 60 | 43 DEFERRED | 1.0 | 0 |
| 7800014 | CLEAR | 718 / 69 / 97 | 60 of 60 | 4 PROCEED | 1.0 | 4 |
| 7800015 | TURBID | 410 / 17 / 96 | 60 of 60 | 44 DEFERRED | 1.0 | 0 |
| 7800015 | CLEAR | 555 / 68 / 96 | 60 of 60 | 5 PROCEED | 1.0 | 4 |
| 7800016 | TURBID | 494 / 12 / 96 | 60 of 60 | 43 DEFERRED | 1.0 | 0 |
| 7800016 | CLEAR | 690 / 80 / 98 | 60 of 60 | 4 PROCEED | 1.0 | 4 |

**First deferral in each TURBID run:**

| World | Time | 2E mean belief | Believed transmission | True transmission | Claims cited (2T / 2S / 2E) |
|---|---|---|---|---|---|
| 7800014 | 36.0 s | 13.1 NTU | 0.436 | 0.324 | 1 / 14 / 2 |
| 7800015 | 34.0 s | 16.5 NTU | 0.370 | 0.324 | 1 / 10 / 2 |
| 7800016 | 36.0 s | 25.9 NTU | 0.257 | 0.324 | 1 / 10 / 2 |

- **Turbidity at the target.** True: 25.5 to 27.0 NTU in TURBID, 1.99 to 2.00 NTU in CLEAR.
- **What the deferral changes.** In CLEAR the inspection goal started at 34.1 to 36.1 s. In TURBID it never started.

**Authority, every run:**

- 0 bus rejections;
- 0 bus heads that are not a revision their child committed; 5 heads lagging the 2T child;
- every revision root written by its own domain package (1,345 to 1,875 revisions checked);
- 0 DIRECT revisions with evidence from another domain's sensors;
- 1,996 to 2,742 cross-domain deliveries, all CROSS_DOMAIN_CONTEXT;
- 338 to 496 TECHNICAL CONTEXT revisions, none consuming evidence;
- foreign-write probe: 104 to 194 attempts per run, all rejected, heads unchanged.

Evidence recorded with `record_gate_evidence.py I6 --surrogate`: 3 of 3 criteria PASS.

## Formal Unity I6: what is ready, and how to record it

1. **Rebuild the player.** Do this after the I4 agent has released it; I have not built it.

   ```
   "C:/Program Files/Unity/Hub/Editor/6000.5.9f1/Editor/Unity.exe" -batchmode -nographics -quit \
     -projectPath unity/ConradUnityV2 -executeMethod Conrad.UnityV2.Editor.BuildScript.BuildWindows64Player \
     -logFile unity/ConradUnityV2/Logs/i6_build.log
   ```

   - Check the log for 0 `error CS`.
   - `tests/unity_live/test_i6_unity.py` skips itself until the built `ConradUnityV2.dll` contains the `optics_grid` literal, so a stale player is never launched on a final world.
2. **Record.** One command:

   ```
   python -m uv run python scripts/record_unity_gate_evidence.py I6
   ```

   - It flies 6 sequential missions: 3 worlds x TURBID/CLEAR, about 2 minutes of sim time each.
   - It writes `artifacts/gates/I6/{evidence_formal.json, unity_measured.json, unity_i6_results.json, unity_pytest.*}`.
   - Criterion names match `gates.py`; the recorder checks this.
   - Every criterion also needs three passing nodes: replay, the leakage scan, and `test_twin2e_reaches_the_unity_camera`. That test checks that optics updates were sent, that the camera frames carry the grid, and that TURBID transmission is below CLEAR on each world.
3. **Scope of a formal PASS.** Even with a formal PASS, I6 stays BLOCKED_UPSTREAM until I5 passes.

## Honest notes

- **The old player was launched once, by mistake.** I ran `pytest tests/unity_live/test_i6_unity.py` to check collection, before adding the stale-player guard.
  - The existing player (without Twin2E support) started on world 7800014 for about one minute.
  - The run stopped inside `UnityMissionWorld.build`: no `conrad.sqlite`, no events and no mission tick exist. The player log ends at "bridge up".
  - No measurement was produced or read, and the stray run directory was deleted.
  - The guard was then added.
  - The formal run on 7800014 was never started. Only the world was generated.
- **The 2E field belief is grid-level.**
  - The bus carries one turbidity belief for the whole grid. In TURBID its mean ran up to about 16 NTU below the true turbidity at the target (11.1 against 26.9 NTU on 7800014).
  - The gate is correct here only because it uses mean + 2 sd.
  - A station-local 2E belief on the bus would be better: Model2E's `run_query` supports regions, but the bus serves only heads. This is a gap for the 2E and bus owners.
- **The bus lags Model2T.** Model2T's CONTEXT revisions (the 2E visibility context) are committed but not republished. Model1 therefore sees 2T heads without the latest context uncertainty: 5 heads per run.
- **Deferral means holding.** Twin2E turbidity settles over hours (the settling-time prior is 3,600 to 86,400 s), so within a 120 s mission a deferred inspection is never re-planned. The mission holds and EGDC keeps re-requesting, and each request is deferred.
  - A closer standoff would not help in the twin: structural-reading quality depends on turbidity, not on range (`MissionSensorSuite._quality`).
- **Validity.** All parameters are SYNTHETIC_ONLY. All gate thresholds are ENGINEERING_ESTIMATE values.
