# I4 and I5 combined closure: development record

Date: 2026-09-24. Base `8998201` on `kerem/i5-final-repair`; branch
`kerem/i4-i5-full-closure`. This is a new development cycle. The historical I4
formal FAIL 4/5 and I5 formal FAIL 9/10 remain immutable. No new formal result
exists.

The original checkout was dirty with unrelated artifacts and was preserved. The
legacy repository was inspected read only at `6bf01ee` with a clean status. At
recovery there was no active Python or Unity process. An interrupted broad pytest
run from the preceding I5 task had reached 81% but did not finish; it is not a
verification result. The latest I5 forensic milestone was committed as `8998201`
and its targeted tests passed 31/31. This branch starts from it.

## I4 development hypothesis, declared before candidate missions

The existing 40-world current-code I4 development traces were joined by plan ID
and action ID into `artifacts/experiments/I4-I5-CLOSURE/i4_development_action_ledger.json`.
Across 70 accepted V4 views, 24 repeated a previously commanded pose within
0.1 m with the same modality. All 24 had *zero predicted mission gain*, yet
four contained a direct target revision in the repository; 13 were flown and
11 ended at the mission clock. Their metered view intervals consumed 33,820 J. This is an energy
attribution, not an estimate of energy that a different policy would save: the
robot still has a mission after declining a view.

The frozen Bayesian EIG ratio ranker can assign a positive generic entropy
score to one of those views even though the analytic mission gain for the open
need is zero. Actual direct revisions in four such intervals show that zero
predicted mission gain is not a safe proxy for zero information. The candidate
tested here applies a general positive-mission-gain
feasibility rule *before ranking*. It refuses any feasible view with predicted
mission gain <= 0 and records `NO_PREDICTED_MISSION_GAIN`. Calibration checks,
which require a fresh direct revision rather than uncertainty reduction, keep
their existing behavior. The feature is opt-in on V4; production I5 and other
gates remain on their prior planner configuration. A unit test checks that an
all-repeated candidate set returns an explicit no-feasible plan.

Screen this candidate against V4 on the already-development seeds
`8100000..8100009`, then across the remaining development seeds only if the
screen preserves safety and shows a new defect read or a positive paired
information/kJ difference. Require no worse collisions, no hidden-state loss,
and no loss of the matched simple-view comparison before selection. This rule
is declared before candidate missions. The I4 validation range
`8100100..8100139`, final pool `8100200..8100319`, and OOD range
`8100400..8100419` stay closed.

### Screen result and withdrawal

All ten candidate development missions completed. Compared pairwise with the
same ten incumbent development bundles, hidden-state improvement was exactly
unchanged (0.489703 both), target read stayed 6/10, and mean information/kJ
stayed 0.027490. Candidate mission energy rose by 1,898.7 J per world and
travel by 2.04 m per world; redundant observations rose from 1.2 to 1.3 per
world. Collisions remained zero. The extra energy was concentrated in worlds
where neither arm read the defect, so the information/kJ ratio was unchanged.
The predeclared screen condition failed. No remaining development seed was run
with this candidate. The opt-in planner code and test were reverted; the
action ledger, paired summary and local development bundles are retained as
diagnostic evidence. The large bundles are ignored by Git, not deleted.

The outcome reinforces the causal distinction: refusing a view with zero
*predicted* mission gain does not stop the mission's other power draw or
guarantee a better view. The
historical I4 formal FAIL remains unchanged.

## I5 track boundary

The v7 I5 development forensics at `8998201` show sparse spatial support from
one averaged nominal reading per visible rest surface. A duplicate-view filter
changed paths but did not improve warrant reachability and was reverted. The
READABLE control has 8 by 8 tile readings and a separate pristine-rest truth
override; copying it wholesale would alter the scenario. This cycle will
examine the sensor observation contract separately. The v7 validation and
final worlds remain unopened. No I5 criterion, warrant, threshold, baseline or
historical evidence is changed.

### I5 observation-contract limitation

The structural payload's ordinary `region_tiles=None` renders one T0 scalar
from the entire visible rest surface and locates it at that visible surface's
mean point. Association stores a measured point and position uncertainty in
`Evidence.spatial_support`, but no measured footprint extent. Model2T then
credits cells within a declared 50 degree / 1 m footprint. READABLE sets 8 by
8 tiles, but every tile still reads the same Twin2T rest-region state;
`pristine_rest=True` also changes that state. These facts explain why simply
increasing tile count is not evidence that ordinary nominal can rule out a
local defect.

The identifiability problem is independent of seed choice. A surface uniformly
at corrosion depth *d* and a surface with half its area at zero and half at
*2d* have the same average but different component worst cases. An averaged
scalar and a single mean point cannot distinguish them. The current Twin2T
rest surface has one state for the entire region, so it cannot even represent
the second world as a local counterexample outside the explicit defect patch.
To claim OBSERVED INTACT from more credited cells, the system needs a measured
spatial resolution/support contract and a truth model with local states that
can test missed defects. Chapter 26 Phase 8 names vehicle/sensor constraints
as external boundary input; the current repository has no calibrated payload
resolution for this contract. This is a technical blocker to promoting a
coverage-only I5 repair, not a proposal to weaken the warrant.

The committed I5 trace script writes `truth_initial_evaluation_only`, but the
earlier committed JSON traces lack that key. Both nominal kernel development
seeds were reproduced with that script and a `repro` tag. Observation counts,
final coverage and scored outcomes matched their earlier runs; the new files
also contain the initial truth values. The source edit that added the field to
the script before the earlier commit cannot be dated from the old JSON alone.
The discrepancy is now explicit rather than silently attributed to the old
artifact. The truth data remain evaluation-only and never enter runtime.

The nominal rest-region truth differs across those two development worlds.
Seed 8300000 starts with a 2.637 mm crack on the rest region, above the
declared 2.5 mm DEGRADED band, although the named target patch is set to zero.
Seed 8300001 starts with a 0.264 mm crack and 0.891 mm corrosion on the rest
region. The first world has a real local reason not to warrant CONTINUE even
if coverage closes. The second is a better test of coverage reachability but
still ends at only 38/64 credited cells. This is a development truth-side
diagnosis, not a seed-selection rule or a change to I5-NOMINAL.

## Closure decision for this cycle

The user requested either genuine formal PASS for both gates or a demonstrated
fundamental limitation that requires an authoritative architecture or sensor
specification change. The second condition is met for I5: the current ordinary
observation is one scalar and one mean point for a potentially heterogeneous
surface, while the required component claim is a worst case over that surface.
The observation map is many-to-one. No belief update can infer which of two
surfaces with the same average contains a local defect without additional
spatially resolved data or an assumption that every part is homogeneous. The
current truth implementation makes that homogeneity assumption for the rest
region, and READABLE repeats its shared state in multiple tile readings.
Promoting either assumption into an OBSERVED INTACT claim would not test the
intended hidden-local-defect problem.

A valid next architecture needs independent payload spatial-resolution and
support semantics, a surface-local truth state that can generate heterogeneous
defects, and evidence carrying measured support into Model2T. The authoritative
Chapter 26 work package assigns feasible sensor constraints to the physical
sensor boundary; none is provided for this payload in the repository. Choosing
tile size or credited extent from the failed gate would be post-result tuning.
This cycle therefore stops at **B**. It does not declare a repair, open I4/I5
validation or final worlds, or make another Unity formal attempt. The fresh I5
v7 seeds 8300000 and 8300001 are development-touched; its validation and final
splits remain sealed. The I4 8100000..8100009 candidate screen used only the
existing development split; I4 validation, final and OOD remain sealed.

No formal criterion, threshold, baseline, metric, confidence rule, scenario
semantics, historical evidence or gate verdict changed. I4 remains **FORMAL
FAIL 4/5** and I5 remains **FORMAL FAIL 9/10**. The source change that remains
is the partition loader's correction to count allocated world seeds rather
than another file's exclusion metadata as a collision. I4/I5/I7 production
runtime paths are unchanged after candidate withdrawal.
