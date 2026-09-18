# Learned CEFD

- **Code:** `conrad/domains/ecological/learned/` (`LearnedCEFD`, `train_learned_cefd`)
- **Config:** `LearnedCEFDConfig` (d_model 256, 3 entity layers, 8 heads, U-Net channels 64/128/256)

## Architecture

Entity stream: heterogeneous graph transformer with relation-type attention bias. Field stream: 3-level 3D U-Net.
Coupling: gated cross-attention from each entity to field tokens at its position and 6 neighbours; entity-to-field
messages splatted into the nearest voxel. Loss `cefd_loss`: masked Gaussian NLL (entity and field), masked gradient
loss, BCE of the gate against a `coupled` label.

## Training data

`synthetic_batch` (SYNTHETIC_ONLY) in unit tests only.

## Executed

`tests/unit/domains/ecological/test_eco_encoder_learned.py` (forward/backward, 15 training steps, padded entities do
not couple). Not used by any experiment.

## Results

None beyond the unit test. 2E-E001..E003 measured the analytic CEFD only; 2E-E003 found coupling benefit about 0
and confident thermal-stress claims on all healthy entities in a confounded world (analytic CEFD coupling
KILL_CANDIDATE).

## Known failure modes

Unknown for the learned module.

## Claim status

IMPLEMENTED. Nothing measured.
