# EGDC learned scorer

- **Code:** `conrad/decision/learned.py`, features `learned_features.py`; policy `LearnedEGDCPolicy`
- **Config:** `EGDCScorerConfig`

## Architecture

Claim-node features (claim 256 + uncertainty 64 + provenance 32 + context 32) projected to 256, 4 graph
transformer layers with 8 heads and typed-edge attention bias, a learned [MISSION] token, an action score MLP, a
claim-support attention readout and an outcome head. `LearnedEGDCPolicy` only ranks; the deterministic
ConstraintEngine still decides. Loss `egdc_loss`: action CE + support BCE + masked outcome MSE + a UIR penalty on
support mass placed on unsupported claims. Training function `train_imitation`.

## Training data

Toy batches in unit tests only. The module docstring calls it an untrained architecture plus an imitation training
function.

## Executed

`tests/unit/decision/test_egdc_learned.py` (forward/backward with masks; imitation training reduces loss; the policy
only ranks). Not used in M1-UIR-E001.

## Results

None. M1-UIR-E001 measured `StructuredReasoningPolicy` and the naive baseline only.

## Known failure modes

Unknown: not evaluated.

## Claim status

IMPLEMENTED. Nothing measured.
