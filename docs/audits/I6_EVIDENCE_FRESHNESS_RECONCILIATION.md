# I6 evidence freshness reconciliation

Date: 2026-09-26
Verdict: **HISTORICAL FORMAL UNITY PASS REMAINS APPLICABLE**

## Question

Spatial V1 and the V1.1 candidate change technical structural truth,
pose-derived structural support, the optional `spatial_v1` Model2T backend,
and its visibility-certificate replay identity. This audit asks whether those
changes invalidate the formal I6 multi-domain result.

## Execution-path comparison

I6 is defined by `configs/eval/i6_multidomain.yaml` and the two
`I6-MULTIDOMAIN-*` scenarios. Their runtime enables only the multi-domain
orchestration gate. It does not select `twin2t_truth_model: spatial_v1`, does
not select `model2t_backend: spatial_v1`, and does not configure the spatial
structural sensor. The default technical child used by the formal I6 run is
therefore unchanged by the Spatial V1/V1.1 opt-in path.

The three I6 criteria are also architecture-independent at this boundary:

1. one mission produces evidence-backed 2S, 2T, and 2E beliefs;
2. Model1 cites committed heads from all three domains through the Belief Bus;
3. each child remains authoritative for its own domain.

The formal record additionally checks deterministic replay, runtime truth
leakage, and Twin2E optics reaching the Unity camera. None of those contracts
depends on the capsule visibility certificate changed in V1.1.

## Retained evidence

- Formal Unity evidence commit:
  `b7859b2b1326691404de30e7b8cf5a4e5a51cf42`
- Formal evidence SHA-256:
  `1e71ce953aa1f33e8b6e09847382a9b96cd4634a62dd7e1e3700e67deed21109`
- Formal per-run result SHA-256:
  `3f61e1fd1f82f4094f322300cab16fda5cd57bdd835814b02131db7029f5bae9`
- Frozen I6 declaration SHA-256:
  `a82f77ab47bc6b29ca296a09f2d4956936a38d51a102a7d3ee32caf04ffd94af`
- Formal result: all three criteria passed on all three declared worlds in
  both CLEAR and TURBID arms, with exact replay and zero runtime truth leaks.

On the current V1.1 candidate source, the retained-artifact and live
multi-domain integration suites pass 10/10:
`tests/integration/test_i6_multi_domain.py` and
`tests/integration/test_multidomain.py`.

## Decision

A fresh I6 final or formal run is not justified. Spatial V1/V1.1 is opt-in and
not active in the I6 scenario, and the changed visibility/localization code is
outside the behavior I6 measures. Reusing the historical formal verdict is a
scope-preserving freshness decision, not a relabelled or regenerated result.

I6 remains **FORMAL UNITY PASS (3/3)**. Any gate-table dependency status must
still be computed separately from the current upstream I4/I5 states. If a
future I6 protocol explicitly enables the spatial truth/Model2T backend, that
will be a new experiment requiring fresh development and final worlds.
