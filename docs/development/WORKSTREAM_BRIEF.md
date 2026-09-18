# Conrad V2 workstream brief (binding for every contributor and coding agent)

Repository: `C:\Users\Kerem\Kerem\conrad-v2` (NEW clean-room repo). Specification text:
`docs/specification/source/ready_new_agent_v2.txt` (14k lines; chapter headings start with `# N.`).

## Absolute rules

1. **Twins = truth world. Model 2 = belief world. Model 1 = decision world.** Twin1 does not exist.
2. **Zero truth leakage.** Packages `conrad.core`, `conrad.domains`, `conrad.decision`, `conrad.active`,
   `conrad.communication`, `conrad.robotics`, `conrad.runtime` must NEVER import `conrad.twins`,
   `conrad.schemas.truth`, `conrad.sim.kernel` truth APIs, or anything under `conrad.evaluation.oracle`.
   A static test enforces this. Truth is allowed only in `conrad.twins`, `conrad.sim`, `conrad.training`,
   `conrad.evaluation`, `conrad.data` and tests.
3. A simulator hidden entity ID is never an inference feature or a belief ID. Belief IDs are minted by
   the belief plane. `registry_entity_id` may be set only from a known asset registry supplied as mission
   context (not from truth).
4. **Evidence is not belief.** Never copy an evidence embedding into a belief embedding.
5. Uncertainty is always the 4-channel `Uncertainty` (UA, UE, UC, UO). Never a single confidence scalar.
   It must respond to inputs (no constants): corruption -> UA up, OOD -> UE up, credible disagreement -> UC up,
   low coverage -> UO up.
6. Knowledge status is property-level: OBSERVED / INFERRED / PREDICTED / UNKNOWN (+ MIXED summary).
   Relational results are INFERRED. Temporal results are PREDICTED. UNKNOWN is valid and is NOT "occupied".
7. Every spatial value names its frame; every time value is integer ns in a clock domain; physical
   elapsed time `delta_t` (seconds) drives temporal dynamics, never a sequence index.
8. Neural outputs never command hardware. Only `conrad.runtime.command_gateway` may call
   `RobotHardwareInterface.send`. MCBR outputs an `ObservationPlan`; navigation moves the robot.
9. Do not import the legacy repository, reference its paths, or copy from it. Do not import vendor SDKs,
   ROS or Unity libraries from core packages.
10. Never fabricate physical measurements, real-world data, metrics or results. Anything unmeasured is
    `SourceKind.OPEN` or explicitly labelled `SYNTHETIC_ONLY` / `ENGINEERING_ESTIMATE`.
11. Learned mechanisms are `EXPERIMENTAL_CANDIDATE`: configurable, with baselines and ablations. Dimensions
    and thresholds come from config (defaults from spec ch33: De=Dz=256, Du=64, Dt=64, Dh=128, Dr=64, Dm=128,
    heads 8, dropout 0.10, GELU, LayerNorm, AdamW). Never hard-code them inside modules.
12. No fake completion: no no-op production adapters, no hard-coded success, no swallowed exceptions.

## Contracts

Use the public schemas in `conrad/schemas/` exactly as they are. Do NOT edit files under
`conrad/schemas/`, `conrad/persistence/`, `conrad/runtime/`, `conrad/settings.py`, `pyproject.toml`.
If a contract is missing something you need, keep a domain-local model inside your own package and
list the gap at the end of your report.

All models subclass `conrad.schemas.base.VersionedModel`/`ConradModel` (frozen, extra=forbid).
IDs come from an injected `conrad.schemas.ids.IdFactory` (deterministic when seeded) - never call
`uuid.uuid4()`; never use wall-clock time inside deterministic code (time is injected).
All randomness uses an injected/seeded `numpy.random.Generator` or `torch.Generator`.

## Package metadata

Each package `__init__.py` must define:

```python
IMPLEMENTATION_METADATA = {
    "implementation_status": "FROZEN_CONTRACT" | "EXPERIMENTAL_CANDIDATE" | "OPEN_BLOCKED",
    "source_sections": [...], "configuration_keys": [...], "assumptions": [...],
    "baselines": [...], "acceptance_tests": [...], "claim_status": "NONE" | "IMPLEMENTED" | "EVALUATED" | "VALIDATED",
}
```

`claim_status` may be at most `IMPLEMENTED` unless an experiment you actually executed supports more.

## Tooling (Windows host, bash shell)

Run everything from the repo root with `python -m uv run ...` (uv is not on PATH):

```
python -m uv run ruff check <your paths>
python -m uv run ruff format <your paths>
python -m uv run mypy <your package paths>
python -m uv run pytest <your test paths> -q
```

Python 3.11, torch 2.6 CPU only (no GPU on this machine). Keep tests fast (whole package suite under
~60 s on CPU); use tiny dimensions in tests via config. mypy is strict (`disallow_untyped_defs`) for
`conrad/`; tests may be untyped. Line length 110. Do not add dependencies.
Do not run `git` commands; the integrator commits. Do not create files outside your assigned paths.

## Reporting

Finish with a concise report: files created, public API (import paths + signatures) the integrator
should call, test command + result counts, every assumption you made that is not in the spec, every
OPEN item you left OPEN, and any contract gaps. Report real numbers only from runs you executed.
