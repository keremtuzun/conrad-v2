# ECMER encoders

- **Code:** `conrad/core/ecmer` (`encoders.py`, `fusion.py`, `ssl.py`, `training.py`, `service.py`, `baselines.py`)
- **Config:** `model2_core.ecmer.*` (`EcmerConfig`, `conrad/core/config.py`)

## Architecture

Separate `ResNet18Encoder`s for RGB (3 channels) and sonar (1 channel), `VoxelPointEncoder` for points,
`ScalarSensorEncoder` for scalars, `PoseTimeContextEncoder`. Fusion: an [EVENT] token plus 4 pre-norm transformer
blocks. Quality head outputs reliability, aleatoric log-variance, OOD score and usability. `EcmerEncoder.encode`
turns observations into `Evidence` with DIRECT_OBSERVATION provenance. Missing modalities contribute masked tokens.

## Training

SSL objective `ReprObjective` (view contrast, temporal, cross-modal, masked reconstruction) and stages E1..E4 in
`training.py`. No corpus is registered; the real SSL partition is empty (EXT-DATA-01).

## Executed

Unit tests only (`tests/unit/core/test_core_ecmer.py`). No experiment, no trained checkpoint.

## Results

None.

## Known failure modes

Unknown: not evaluated. Baselines (single modality, concat, early, late, cross-attention) exist but were not
compared.

## Claim status

IMPLEMENTED. Nothing measured.
