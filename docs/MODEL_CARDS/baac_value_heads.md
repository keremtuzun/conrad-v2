# BAAC value heads

- **Code:** `conrad/communication/learned.py` (`build_heads`, `train_heads`)
- **Config:** `BAACHeadsConfig`

## Architecture

Unit and receiver features (64 each) projected to 256 by separate MLPs, 2 cross-attention blocks (up to 256
receiver tokens), one linear output split into novelty, value, information loss per fidelity level (5), log bits,
latency and deadline-violation heads. Loss `baac_loss`: Huber value + BCE novelty + BCE deadline + MSE information
loss, masked by `unit_mask`. `train_heads` uses AdamW without gradient clipping.

## Training data

Toy batches in the unit test only.

## Executed

`tests/unit/communication/test_baac.py::test_learned_value_heads_smoke`. No runtime hook; not used in COM-BAAC-E001.

## Results

None. COM-BAAC-E001 measured the analytic scheduler.

## Known failure modes

Unknown: not evaluated.

## Claim status

IMPLEMENTED. Nothing measured.
