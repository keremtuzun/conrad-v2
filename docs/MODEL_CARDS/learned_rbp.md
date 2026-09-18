# Learned RBP

- **Code:** `conrad/core/rbp.py` (`RelationalBeliefPropagation`)
- **Config:** `model2_core.rbp.*` (`typed`, `layers`=3, `adaptive_stop`=False)

## Architecture

Relation embedding (16 types + numeric features), 3 attention message layers with a relation bias and a gate,
segment softmax over incoming edges. Updates the latent `z` only. Loss: state loss + `contamination_weight` x
contamination loss on misleading targets.

## Training data

Abstract Core sandbox with true and misleading edges, synthetic.

## Executed

CORE-RBP-E001 (`rbp_e001.py`), typed and untyped variants.

## Results

Learned RBP error about 0.20 with contamination about 0.47; typing gave no advantage. (Analytic RBP: true edges
RMSE 0.031 vs 0.231 without propagation, but 0.58 of decoys confidently wrong on misleading edges.)

## Known failure modes

Contamination through misleading edges.

## Claim status

IMPLEMENTED. Results record verdict KILL_CANDIDATE. Runtime uses `AnalyticRBP`; the results record says RBP
should be off by default in the runtime.
