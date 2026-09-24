# I5 nominal CONTINUE final repair cycle: development stop

Date: 2026-09-24. Branch: `kerem/i5-final-repair`, based on
`fae8b24c0ee3ca16cc27123a42a544eea2b9999d`. Evidence class: synthetic development diagnosis.

## Historical baseline

The completed Unity formal result remains **I5 FORMAL = FAIL, 9/10**. All 42 flights on spent worlds
`7710300` and `7710301` completed. The sole failed criterion was integrated action exercise:
`I5-NOMINAL` reached 0/2 CONTINUE warrants, so correct-given-warrant was undefined. The other six
Unity scenarios scored 1.0 correct-given-warrant; action-matrix CONTINUE was 100/100. The predeclared
zero-warrant failure rule stands. The successful older surrogate E008 had 2/10 ordinary nominal
warrants and 9/10 READABLE nominal warrants, indicating low ordinary warrant incidence even in the
kernel. None of those old final worlds was rerun.

## Forensic development

Fresh v7 partition: development `8300000..8300009`, validation `8300100..8300109`, final_test
`8300200..8300209`; canonical digest
`bda2a696da495ce23403cb54c31aa31bfacc48a0fd425be332a57b24046be153`.
Only the first two prespecified development seeds were used. They are now development-touched; the
remaining v7 worlds and both sealed splits remain untouched. The instrumented time series and the
complete causal trace are in `I5_NOMINAL_WARRANT_ANALYSIS.md` and
`artifacts/experiments/I5-NOMINAL-FORENSICS/`.

| Backend and seed | Target readings | Final coverage | Warrant | Main blocker |
|---|---:|---:|---:|---|
| kernel 8300000 | 25 | 50/72 | no | sparse averaged reading footprints |
| Unity 8300000 | 15 | 47/72 | no | sparse averaged reading footprints |
| kernel 8300001 | 72 | 38/64 | no | repeated support; second identical planned view |
| Unity 8300001 | 72 | 45/64 | no | repeated support; second identical planned view |

Kernel versus Unity first diverged in trajectory and first target reading time/point on 8300000
(13.25 s versus 14.30 s). Both associated structural evidence and credited Model2T cells, then
stalled below 0.8. This is a physical path difference, not a Unity observation drop or an evaluator
error. On 8300001, both paths accepted the **same pose twice** at 36 s and 64 s; the second goal
lasted to mission end. The kernel ranker scored the second view above distinct feasible poses despite
zero observational gain and redundancy 1 in its candidate table. The frozen player hash was
`36c5c9f13481406382a8e9ef8fc0ea7cdf055c43bb12fc8fd545b07c199cd277`.

READABLE is not an ordinary nominal sample. It creates 8 x 8 tile observations and forces the rest
region pristine; ordinary NOMINAL gives one averaged scalar for that region and samples its state.
On 8300000, READABLE reached 66/72 cells and a 64 s warrant in kernel, but Unity reached 72/72
without a warrant: `OBSERVED INTACT` at 54 s had a pending report and became `DEGRADED` at 56 s.
Coverage, condition value and report state must all be measured separately.

## Repair candidate and rejection

One general, belief-side candidate refused an identical previously accepted pose and modality for
`EXTEND_COVERAGE`. A first diagnostic patch on V4 alone missed the production planner and was
corrected before interpreting its mission result. The production-path candidate passed a targeted
planner test and made 8300001 fly four distinct views. It still ended at 39/64 cells rather than
38/64, with no warrant. Seed 8300000 stayed at 50/72, with no warrant. No other I5 scenario was
run under this candidate, because it already failed the necessary nominal-reachability screen.
The candidate code was reverted. It was never frozen or promoted.

The deeper issue is a contract mismatch: the predictor values cellwise visibility, but the ordinary
sensor delivers a single scalar associated with the mean visible point. Distinct viewpoints can
produce almost the same credited footprint. Assigning every visible cell to an averaged scalar
would pretend to rule out a local defect that the scalar cannot localize. No independent payload
specification here establishes the per-tile measurement contract. The READABLE override is therefore
not a justified production repair. Lowering completeness, extending only I5-NOMINAL, forcing INTACT,
changing the warrant, or restoring NOT APPLICABLE were rejected.

## Selection, power and stop

The prospective selection rule was: a general mechanism must materially raise legitimate nominal
warrant incidence on development, retain correct-given-warrant at least 0.9, preserve zero nominal
over-escalations, zero hard violations, UIR 0, traceability 1.0, all other I5 scenario floors,
competitive baselines and safety. No candidate passed the first requirement. Therefore validation
was **not opened**, no fresh surrogate final or Unity formal partition was declared, and no full
I5 matrix was run. This is a development stop, not an evaluation PASS or a new formal result.

As a power illustration only, using the old E008 point incidence p = 2/10 = 0.2, two independent
worlds have P(at least one warrant) = 0.36 and P(zero) = 0.64. Fourteen worlds would give
P(at least one) = 0.956; 22 worlds would give P(at least two) = 0.952. Those estimates are too
uncertain and the candidate did not improve incidence, so neither N is a frozen formal design.

Shared I4/I6/I7 runtime and Model2T code are unchanged after candidate withdrawal. Historical I4
FAIL 4/5 and I7 surrogate FAIL 3/4, formal NOT_RUN remain as recorded. No gate evidence or verdict
file was modified. **I5 FORMAL = FAIL.** Stop this cycle.
