# MCBR learned ranker

- **Code:** `conrad/active/learned.py` (`build_ranker`, `train_ranker`)
- **Config:** `MCBRRankerConfig`

## Architecture

Candidate features (belief, uncertainty, hypothesis, viewpoint, observability, cost, mission) projected to 256; 3
blocks of candidate self-attention plus cross-attention to context tokens. Heads: visibility, value,
discrimination, cost/risk; score = value minus cost/risk. Loss `mcbr_loss`: Huber value + pairwise logistic rank +
Brier visibility, masked by `candidate_mask` (feasible candidates only).

## Training data

Toy batches in the unit test only.

## Executed

`tests/unit/active/test_mcbr.py::test_learned_ranker_forward_backward_and_training`. No runtime hook; not used in
ACTIVE-MCBR-E001.

## Results

None. ACTIVE-MCBR-E001 measured the analytic MCBR, which ranked 8th of 11 on mission error reduction (0.359).

## Known failure modes

Unknown: not evaluated.

## Claim status

IMPLEMENTED. Nothing measured.
