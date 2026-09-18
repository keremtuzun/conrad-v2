# Learned association scorer

- **Code:** `conrad/core/association_scorer.py` (`AssociationScorer`), wrapper `conrad/core/association.py`
- **Config:** `model2_core.association.*`

## Architecture

Pair MLP over evidence, belief, pose, time and relation features (1109 inputs at defaults) with hidden layers
512, 256, 128, plus a NO_MATCH head. Output: logits over K candidates + NO_MATCH. Loss `association_loss`: masked
cross entropy, focal loss on NO_MATCH rows, ranking hinge.

## Training data

Abstract Core sandbox (`conrad/evaluation/core_experiments/sandbox_world.py`), synthetic, seed-separated train and
test episodes.

## Executed

CORE-ASSOC-E001 (`assoc_e001.py`, `configs/eval/core_assoc_e001.yaml`).

## Results

Accuracy 0.669 ± 0.182 vs nearest neighbour 0.955 ± 0.016. False association 0.0008 vs 0.045.

## Known failure modes

Over-conservative: it rarely associates wrongly but misses many true associations.

## Claim status

IMPLEMENTED. Baseline wins; results record verdict KILL_CANDIDATE. Runtime uses nearest neighbour.
