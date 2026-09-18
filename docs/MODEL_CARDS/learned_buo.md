# Learned BUO

- **Code:** `conrad/core/buo.py` (`BeliefUpdateOperator`)
- **Config:** `model2_core.buo.*`

## Architecture

Evidence items (De + 4 quality) pooled by a set aggregator, residual MLP trunk, trust, innovation and gate heads;
update `LayerNorm(z + trust * gate * innovation)`; uncertainty update through `UncertaintyHeads` (four channels).
Ablation flags for trust, gate, contradiction.

## Training data

Abstract Core sandbox, synthetic.

## Executed

CORE-BUO-E001 (`buo_e001.py`), loss MSE + innovation L2.

## Results

Learned BUO was worse than the analytic operator in every condition (errors 0.076 to 0.184). The analytic BUO
matched averaging on reliable support (RMSE 0.021) and beat it on unreliable contradiction (0.020 vs 0.030).

## Known failure modes

Worse than analytic everywhere in this sandbox. (The analytic operator, not this module, lost to latest-only on
true change: 0.103 vs 0.049.)

## Claim status

IMPLEMENTED. Results record verdict KILL_CANDIDATE. Runtime uses `AnalyticBUO`.
