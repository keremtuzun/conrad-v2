# Learned TCDP

- **Code:** `conrad/domains/technical/learned_tcdp.py` (`LearnedTCDP`, `train_learned_tcdp`)
- **Config:** `LearnedTCDPConfig` (d_z 256, d_r 64, d_mech 128, 3 layers, 6 mechanism kinds)

## Architecture

Node encoder, mechanism embedding (CORROSION, FATIGUE, CRACK, LOAD_PATH, REPAIR_INTERVENTION, GENERIC), 3
`TCDPLayer`s with edge messages, edge gate weighted by reliability, and updates only to mutable latent
partitions. Directly observed nodes are never updated. Head outputs mean and log-variance for corrosion and crack.
Loss `tcdp_loss`: masked heteroscedastic Gaussian NLL plus a contamination penalty on misleading nodes.

## Training data

Toy synthetic graphs in unit tests only.

## Executed

`tests/unit/domains/technical/test_m2t_learned.py` (forward/backward, training reduces loss on 4 batches over 25
epochs, bad partitions rejected). Not used by any experiment; not on the Model2T runtime path.

## Results

None beyond the unit test. 2T-E003 measured the analytic TCDP only.

## Known failure modes

Unknown: not evaluated.

## Claim status

IMPLEMENTED. Nothing measured.
