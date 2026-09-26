# OS-FM Architecture Freeze

Status: CONTROLLED_ARCHITECTURE_FREEZE
Revision: OSFM-P4-CONTEXT-R02

This freeze reconciles the current Conrad/OceanSense implementation before promotable P4.8 OS-FM research
pretraining. It preserves validated I4, P4.7B, and P4.7C evidence while defining the target architecture that P4.8
must not accidentally narrow.

## Product Boundary

OceanSense is a host-independent underwater asset intelligence software and computing layer. It is not an ROV, AUV,
thruster controller, battery system, low-level safety controller, navigation stack vendor, communications stack, or
sensor manufacturer.

Third-party hosts own chassis, propulsion, batteries, failsafes, collision avoidance, communications, navigation,
actuation, and platform-specific safety. OceanSense consumes supported observations through host and sensor adapters,
then returns structured evidence, beliefs, uncertainty, InformationNeeds, ObservationPlans, and high-level
InspectionIntents. The host remains final authority for execution.

## Frozen Conceptual Loop

OBSERVE -> REPRESENT -> UNDERSTAND -> BELIEVE -> REMEMBER -> QUESTION -> CHOOSE EVIDENCE -> INSPECT -> UPDATE.

The ownership split is:

| Layer | Role | Status |
|---|---|---|
| Canonical observations and adapters | Convert host/sensor data into versioned, provenance-bearing observations | NOW |
| OS-FM | Perceive and represent heterogeneous observations and context | NOW |
| Task heads | Produce learned evidence for downstream reasoning | P5 |
| Model2T / Model2E / Model2S | Maintain technical, ecological/environmental, and spatial beliefs | NOW/P6 |
| Model2O | Optional operational/process belief family when real data exists | LATER |
| Cross-domain belief fusion | Relate domain beliefs without inventing causality | P7 |
| Persistent Asset Memory | Remember evidence, beliefs, history, repairs, coverage, and unresolved questions outside OS-FM | P6-P10 |
| Semantic Asset Twin | Visualize current OceanSense belief, not simulator truth | P6-P10 |
| Inspection Intelligence | Produce InformationNeeds, ObservationPlans, and high-level intents | P8-P10 |
| Third-party host | Executes supported safe actions | EXTERNAL HOST |

## OS-FM Inputs

The P4 foundation model keeps the existing D_F=384 modality contract:

- RGB/video tokens.
- Sonar/acoustic-image tokens.
- Range/depth tokens.
- 3D geometry tokens.
- Auxiliary context tokens.

Auxiliary context tokens are sparse typed D_F=384 tokens. The current implemented families are environmental,
platform/navigation, and sensor/acquisition context. Stable future hook families are asset/engineering context,
operational/process context, document/knowledge context, mission/query context, and external context.

Each context observation must preserve field name, units, timestamp, provenance, availability, validity, optional
uncertainty, optional quality, optional frame/clock/calibration references, and source observation identity. Missing,
stale, failed, unsupported, degraded, or invalid values are not silently zero-filled as measured values.

## Exact Evidence Path

OS-FM may consume normalized context for representation learning, but exact physical values remain available through a
parallel Model2 direct-evidence path. Learned context conditioning is not a replacement for typed physical evidence.

Synthetic truth may train controlled objectives, but simulator truth never enters runtime perception or belief as a
real observation. The allowed path is:

Twin truth -> sensor or measurement simulation -> canonical observation -> OS-FM / Model2.

## Uncertainty And Hypotheses

OceanSense preserves four uncertainty channels:

- U_A aleatoric.
- U_E epistemic.
- U_C contradiction.
- U_O observational.

The system must support UNKNOWN, OUT_OF_DISTRIBUTION, INSUFFICIENT_EVIDENCE, and abnormal-but-unidentified states.
Competing hypotheses and counter-evidence are preserved; contradiction is not averaged into fake consensus.

## Phase Reconciliation

| Capability | Current status | Phase | Pre-P4.8 required? | Interface required now? | Implementation required now? | Validation needed | External blocker |
|---|---|---|---|---|---|---|---|
| Public-real DATA-OSFM SubPipe corpus | Implemented and readiness-validated in P4.7B/P4.7C | P4 | yes | yes | yes | readiness artifact | none |
| Context encoder for environment/platform/sensor context | Implemented | P4 | yes | yes | yes | unit, joint smoke, readiness | none |
| Auxiliary context token bus | Implemented as typed contract and D_F=384 fusion path | P4 | yes | yes | contract only beyond current families | unit and schema tests | none |
| Context-free and partial-context operation | Implemented | P4 | yes | yes | yes | fusion/joint regression | none |
| Train-only context normalization | Implemented | P4 | yes | yes | yes | leakage tests | none |
| Direct Model2 evidence path | Implemented for context measurements | P4/P6 | yes | yes | current exact values/provenance | dual-path tests | none |
| Asset/engineering context adapter | Hooked by family contract | P6-P10 | no | yes | no | later public/partner data validation | real asset schemas/data |
| Operational/process context / Model2O | Hooked by family contract | P6-P10 optional | no | yes | no | only with real process data | customer/host data |
| Document/knowledge context | Hooked by family contract | P6-P10 | no | yes | no | retrieval/provenance validation | source documents |
| Mission/query token | Hooked by family contract | P8-P10 | no | yes | no | planning ablations | none |
| Task heads | Existing downstream concept | P5 | no | output contract yes | no | transfer, low-label, OOD | datasets/labels |
| Model2T/E/S retrofit to OS-FM evidence | Existing Model2 families, integration future | P6 | no | yes | no | belief regression | none |
| Cross-domain belief fusion | Existing multidomain concepts, not promoted as causal | P7 | no | yes | no | contradiction/hypothesis tests | none |
| Persistent Asset Memory | Existing persistence primitives, product layer future | P6-P10 | no | yes | no | replay, migration, recovery | none |
| InformationNeed and ObservationPlan | Implemented contracts | P8-P10 | no | yes | current contracts exist | decision benchmarks | none |
| Host capability negotiation / InspectionIntent | Boundary defined; full SDK later | P9 | no | yes | no | host integration tests | host partner/API |
| Edge/cloud hierarchy and profiling | Policy defined | P11-P12 | no | no | no | measured workload profiling | hardware/toolchain |
| FPGA / OS-EDGE | Future product research | P14-P16 | no | no | no | simulator/synthesis/prototype | hardware/toolchain |
| Field pilots | Future validation | P18 | no | no | no | operational evidence | partners/assets |

## Architecture Change Control

Major architecture additions after this freeze require:

1. identified missing capability;
2. explanation of why the frozen interfaces cannot support it;
3. proposed interface changes;
4. retraining impact;
5. migration impact;
6. validation plan;
7. ablation or evidence requirement.

P4.8 must not be downgraded to fit small local hardware. Local RTX 3050 Ti is preferred for CUDA debugging and small
runs, but formal P4.8 compute thresholds remain formal.
