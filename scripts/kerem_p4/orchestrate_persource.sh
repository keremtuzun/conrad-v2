#!/usr/bin/env bash
# Round 4, unattended on one L4 (Kerem trainer): make every sonar source pass on its own.
#   readiness -> ceiling check (pixels / untrained / round-3 best.pt per source)
#   -> 3 pilots warm-started from round 3's best.pt, PILOT_STEPS each:
#        P1 single-source sonar batches; P2 + sonar views; P3 + temporal pairs (+-3 frames)
#   -> pick the pilot with the best post-warm-up min(pooled family ranks, every sonar source's own rank)
#      whose meaning score >= 0.3; long run only if it beats round 3's weakest source by >= MIN_GAIN
#   -> long run (LONG_STEPS) warm-started from the chosen pilot's best.pt
#   -> final checks + ceiling check + evidence bundle -> power off after FINAL_WAIT_MIN minutes.
# Env: VM_START_EPOCH (required), R3_BEST (round-3 best.pt), HOURLY_TRY, PILOT_STEPS, LONG_STEPS, MAX_TRY.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/conrad-v2
L=~/p4logs; mkdir -p $L
CFG=configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml
RD=artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness_kerem_run.json
VM_START_EPOCH=${VM_START_EPOCH:?}; HOURLY_TRY=${HOURLY_TRY:-44}; MAX_TRY=${MAX_TRY:-2500}
R3_BEST=${R3_BEST:-artifacts/runs/v11-p4-kerem-sonar5-a1-20261004T165839Z/checkpoints/best.pt}
PILOT_STEPS=${PILOT_STEPS:-16000}; LONG_STEPS=${LONG_STEPS:-120000}; MIN_GAIN=${MIN_GAIN:-5}; FINAL_WAIT_MIN=${FINAL_WAIT_MIN:-60}
COMMON="KEREM_WEIGHT_DECAY=0.05 KEREM_FAMILY_DEPTH=6 KEREM_CLEAN_ANCHOR=1 KEREM_RANK_WEIGHT=0 KEREM_NN_FLOOR=0.3 KEREM_SONAR_ANCHOR=0 KEREM_SELECT=per_source KEREM_WARMUP=500 KEREM_LR=5e-5 KEREM_EXTRA_CAMERA=seaclear,uvvid KEREM_EXTRA_SONAR=catalunya,china_offshore,aquascan,uatd"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/orchestrate.log; }
spent(){ python3 -c "import time;print(round((time.time()-$VM_START_EPOCH)/3600*$HOURLY_TRY,1))"; }

[ "$(git status --porcelain --untracked-files=no | wc -l)" = "0" ] || { log "ABORT dirty tracked tree"; exit 2; }
[ -f "$R3_BEST" ] || { log "ABORT round-3 best.pt missing: $R3_BEST"; exit 2; }
uv run conrad train osfm-v11-10p-readiness --output $RD > $L/readiness.out 2>&1 || true
BIDS=$(python3 -c "import json;r=json.load(open('$RD'));print(r['decision']+'|'+','.join(sorted(str(i['id']) for i in r['items'] if i['status']=='BLOCKER')))")
case "$BIDS" in "READY FOR KEREM|") OVR="";; "NOT READY FOR KEREM|2,4,8") OVR="--kerem-override-unreviewed-trainer";; *) log "ABORT readiness $BIDS"; exit 3;; esac
log "commit $(git rev-parse HEAD) readiness $BIDS r3_best $(sha256sum $R3_BEST | cut -c1-16)"

env $COMMON uv run python scripts/kerem_p4/ceiling_check.py $L/ceiling_before.json round3_best=$R3_BEST > $L/ceiling_before.out 2>&1
log "ceiling check exit $? spent $(spent)"; cat $L/ceiling_before.out >> $L/orchestrate.log

declare -A PENV=(
  [P1]="KEREM_SONAR_BATCH=single"
  [P2]="KEREM_SONAR_BATCH=single KEREM_SONAR_VIEWS=sonar"
  [P3]="KEREM_SONAR_BATCH=single KEREM_SONAR_VIEWS=sonar KEREM_TEMPORAL_PAIRS=3"
)
for tag in P1 P2 P3; do
  PID=pilot${PILOT_STEPS}-$tag-persource-nonpromotable-$(date -u +%Y%m%dT%H%M%SZ)
  log "pilot $tag start $PID (${PENV[$tag]})"
  env $COMMON ${PENV[$tag]} KEREM_INIT_FROM=$R3_BEST uv run conrad train osfm-v11-10p-launch --config $CFG --readiness $RD --run-id $PID --max-steps $PILOT_STEPS $OVR > $L/$PID.out 2>&1
  log "pilot $tag exit $? spent $(spent) $(tail -1 artifacts/runs/$PID/metrics.jsonl 2>/dev/null | python3 -c "import json,sys;r=json.loads(sys.stdin.read());print({k:round(v,2) for k,v in r.items() if 'source_min' in k or k.endswith('effective_rank') or 'nn_temporal_hit_min' in k})" 2>/dev/null)"
  echo $PID > $L/pilot_$tag
done

CHOICE=$(python3 - <<'PY'
import json, os
base = json.load(open(os.path.expanduser("~/p4logs/ceiling_before.json")))
r3 = min(v["round3_best"] for k, v in base.items() if k.startswith("imaging_sonar/") and not k.endswith("/pooled"))
best = None
for tag in ("P1", "P2", "P3"):
    try:
        rid = open(os.path.expanduser(f"~/p4logs/pilot_{tag}")).read().strip()
        rows = [json.loads(l) for l in open(f"artifacts/runs/{rid}/metrics.jsonl")]
    except Exception:
        continue
    ok = [r for r in rows if r["step"] > 500 and r.get("val/nn_temporal_hit_min", 0) >= 0.3 and "val/selection_rank" in r]
    if not ok or not os.path.isfile(f"artifacts/runs/{rid}/checkpoints/best.pt"):
        continue
    score = max(r["val/selection_rank"] for r in ok)
    if best is None or score > best[0]:
        best = (score, tag, rid, rows[-1].get("steps_per_s", 3.0))
print("" if best is None else f"{best[1]}|{best[2]}|{best[0]:.1f}|{r3:.1f}|{best[3]}")
PY
)
FINAL_RID=""
if [ -z "$CHOICE" ]; then
  log "no usable pilot"
else
  IFS='|' read -r TAG PRID SCORE R3MIN RATE <<< "$CHOICE"
  log "choice $TAG ($PRID) selection rank $SCORE vs round-3 weakest source $R3MIN"
  FINAL_RID=$PRID
  if python3 -c "import sys;sys.exit(0 if $SCORE >= $R3MIN + $MIN_GAIN else 1)"; then
    RID=v11-p4-kerem-persource-a1-$(date -u +%Y%m%dT%H%M%SZ); echo $RID > $L/run_id; FINAL_RID=$RID
    T0=$(date -u +%s); SPENT=$(spent)
    (env $COMMON ${PENV[$TAG]} KEREM_INIT_FROM=artifacts/runs/$PRID/checkpoints/best.pt setsid nohup uv run conrad train osfm-v11-10p-launch --config $CFG --readiness $RD --run-id $RID --max-steps $LONG_STEPS $OVR > $L/$RID.out 2>&1 < /dev/null &)
    sleep 90; TPID=$(pgrep -n -x conrad || true)
    if [ -n "$TPID" ]; then
      log "long run LAUNCHED $RID pid=$TPID steps=$LONG_STEPS from $TAG best.pt spent_before=$SPENT"
      (HOURLY_TRY=$HOURLY_TRY SPENT_BEFORE_TRY=$SPENT MAX_HOURS=30 MAX_TRY=$MAX_TRY RANK_STOP=0 NN_COLLAPSE=0.15 \
        setsid nohup python3 scripts/kerem_p4/watchdog.py artifacts/runs/$RID $TPID $LONG_STEPS $T0 > $L/watchdog.out 2>&1 < /dev/null &)
      while kill -0 $TPID 2>/dev/null; do sleep 60; done
      sleep 30
      log "long run ended spent $(spent) $(tail -1 artifacts/runs/$RID/metrics.jsonl 2>/dev/null | cut -c1-300)"
    else
      log "long run failed to start"; tail -20 $L/$RID.out >> $L/orchestrate.log; FINAL_RID=$PRID
    fi
  else
    log "no long run: best pilot does not beat round 3's weakest source by $MIN_GAIN"
  fi
fi

if [ -n "$FINAL_RID" ]; then
  O=~/final_$FINAL_RID; mkdir -p $O
  R=artifacts/runs/$FINAL_RID
  env $COMMON uv run python scripts/kerem_p4/final_checks.py $R $O > $O/final_checks.out 2>&1; log "final checks exit $?"
  env $COMMON uv run python scripts/kerem_p4/ceiling_check.py $O/ceiling_after.json round3_best=$R3_BEST final_best=$R/checkpoints/best.pt > $O/ceiling_after.out 2>&1; log "ceiling after exit $?"
  sha256sum $R/checkpoints/*.pt $R3_BEST artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt > $O/sha256.txt
  cp -r $R/metrics.jsonl $R/events.jsonl $R/reports $R/config.resolved.yaml $R/git.json $R/manifests.json $R/environment.json $R/checkpoints/*.meta.json $O/ 2>/dev/null
  cp $L/orchestrate.log $L/ceiling_before.json $L/ceiling_before.out $L/watchdog-$FINAL_RID.jsonl $L/$FINAL_RID.out $L/readiness.out $L/gpu.csv $RD $O/ 2>/dev/null
  for t in P1 P2 P3; do P=$(cat $L/pilot_$t 2>/dev/null); [ -n "$P" ] && mkdir -p $O/pilots/$P && cp -r artifacts/runs/$P/metrics.jsonl artifacts/runs/$P/reports $O/pilots/$P/ 2>/dev/null; done
  tar czf ~/final_bundle.tgz -C ~ final_$FINAL_RID && log "bundle $(stat -c %s ~/final_bundle.tgz) bytes"
fi
log "ALL_DONE spent $(spent); powering off in $FINAL_WAIT_MIN minutes"
sleep $((FINAL_WAIT_MIN*60)); sudo shutdown -h now
