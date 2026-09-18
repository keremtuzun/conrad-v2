# Digital Twin salvage report

- Legacy repository: `C:\Users\Kerem\OneDrive\Documents\ChatGPT\conrad` (branch `codex/rov-digital-twin`, HEAD `6bf01ee6ef189145fbc57b65aaed2de46ccea45b`). Audited read-only on 2026-09-18; `git status --porcelain` showed 0 lines before and after.
- Full audit: [DIGITAL_TWIN_LEGACY_AUDIT.md](DIGITAL_TWIN_LEGACY_AUDIT.md). Machine-readable ledger: `artifacts/migration/digital-twin-salvage-ledger.json` (31 entries).

## Components ported

**None.** Every Conrad V2 module was implemented from the specification. Not one line of legacy code, configuration, data or assets was copied.

| Classification | Count |
|---|---|
| REUSE_WITHOUT_SEMANTIC_CHANGE | 0 |
| PORT_AND_ADAPT | 0 |
| REIMPLEMENT_FROM_CONCEPT | 3 |
| REFERENCE_ONLY | 7 |
| OBSOLETE | 15 |
| REJECT_ARCHITECTURAL_CONFLICT | 6 |

## Why nothing qualified for porting

- **Hydrodynamics:** no coefficient has a source. For example, added mass (5,8,10), linear drag (18,22,28) and 50 N thrusters come with no citation, measurement or datasheet. That fails salvage condition 7 (fake physical constants).
- **Unity:** no art assets exist, only primitive shapes and flat-colour materials. The Unity RL agent drives thrusters directly and observes simulator truth.
- **Control, estimation, frames, persistence:** no legacy implementation exists to salvage.
- **Legacy Model 2 datasets (`data/model2/`):** observation `confidence` is computed from the true severity. **They must never be used for V2 training or evaluation.**

## Concepts reimplemented independently from the specification

| Legacy concept | V2 location | Note |
|---|---|---|
| Residuals carry validity flags and sensor sources; missing inputs are never filled in | `conrad/schemas/robot.py` (`Sourced`, `SourceKind.OPEN`), `conrad/schemas/observation.py` (`QualityContext`) | V2 derives this from spec INV-ARCH-08. No legacy code was used. |
| Graph degradation simulator | `conrad/twins/twin2t` (MCDE) | Rebuilt with a strict split between truth and supervision. |
| Parameter identification that leaves unidentifiable values null | `conrad/robotics/hardware/identification` | Identification data is kept separate from validation data. |

## Major rejections

- Legacy Twin1 Unity project
- RL agent and ONNX policies: direct actuator path, truth inputs
- UDP bridge: publishes true pose/velocity; substring command filter
- ROS 2 bridge: untyped JSON, truth-derived telemetry
- `rov_dt` decision stack: single confidence scalar; a second gateway
- Legacy Model 1: image classifier and threshold ladder; AGPL-3.0 optional dependency

## Independence

`tests/leakage/test_static_boundaries.py::test_no_legacy_repository_reference_anywhere` and `scripts/legacy_independence_check.py` run the V2 checks with the legacy repository renamed away.
