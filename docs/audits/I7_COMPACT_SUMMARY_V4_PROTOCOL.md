# I7 compact critical summary v4 protocol

Status: **VALIDATION DECLARED; FINAL UNOPENED**. Historical I7 surrogate
FAIL 3/4 and formal NOT_RUN remain immutable.

## Frozen candidate

The candidate is `baac-critical-summary-v1` with
`compact_critical_summary: true`. It is the 184-bit semantic critical frame
implemented at source commit `0ff37b04ca54e941be7261ae297f9d7ea520a54f`.
The required baselines remain raw, FIFO, and fixed priority. Value-per-bit is
reported but is not in the specification comparison set. The retention oracle,
mission duration, link model, bandwidth levels, outage construction, and
thresholds are unchanged.

Development used only mission development seeds 5100000, 5100001, 5100002,
5100003, 5100007, and 5100008. The candidate achieved a strict per-seed win
against every required baseline in all 42 declared cells: 12 at 1%/0.1%, 18 at
100%/50%/10%, and 12 at outage 100%/10%. Duplicate receiver contributions were
zero. A same-code legacy-summary outage control matched the candidate's bits,
latencies, full-delta retention, and sync state in every cell; the new frame did
not perturb scheduling outside its semantic credit.

## Frozen partition

`configs/eval/partitions_i7_v4.yaml` has canonical SHA-256
`5216bd9be0de40ff3eb91fe3cc8588bd78688267dadc696fdbf808028312fef1`.
Validation seeds are 8400000-8400004. Final seeds are 8400100-8400104 and must
remain unread until validation passes. The loader rejects collision, digest
drift, and selection access to final seeds.

## Validation decision rule

Run `COM-I7-V4-VAL-BW` and `COM-I7-V4-VAL-OUTAGE` once. Select the candidate
only if all of the following hold without dropping a seed or level:

1. Every declared mission completes and every arm receives the identical
   shadow offer stream; duplicate contributions are zero.
2. In every outage shadow run, a critical finding is created while the link is
   down, held through the outage, delivered first after reconnection, and the
   receiver obtains at least the finding revision. There are no resync requests.
3. BAAC retention is strictly greater than raw, FIFO, and fixed priority for
   every validation seed at every nonzero bandwidth and both outage levels.
4. Critical latency and receiver sync are reported for every arm and level.
   Exact latest-revision equality is diagnostic; the operative outage rule is
   delivery of at least the revision found during the outage because later
   revisions can be younger than link latency at mission end.

Any failure stops the cycle. No final seed may then be opened.

## Surrogate-final rule

Only after a validation PASS, commit and push the selected frozen candidate,
then run `COM-I7-E007` and `COM-I7-E008` once on the final split. The same four
criteria apply. All four must pass for `I7 SURROGATE = PASS`. Formal Unity may
run only after that PASS, using a freshly declared formal partition and a
verified current player; otherwise formal remains NOT_RUN.
