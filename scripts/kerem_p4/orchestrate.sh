#!/usr/bin/env bash
# Unattended V1.1 P4 run on one L4 (Kerem trainer):
#   readiness -> pilot A (pixel-patch sonar) and pilot B (frozen P4.8 sonar anchor), 6000 steps each
#   -> pick the pilot with the best post-warm-up weakest-family held-out rank whose meaning score >= 0.3
#   -> long run sized to the remaining 72 h (one automatic retry at half LR on semantic collapse)
#   -> final checks + evidence bundle -> power off after FINAL_WAIT_MIN minutes.
# Env: VM_START_EPOCH (required), VM_HARD_STOP_H, HOURLY_TRY, KEREM_EXTRA_SONAR, KEREM_EXTRA_CAMERA, PILOT_STEPS.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/conrad-v2
L=~/p4logs; mkdir -p $L
CFG=configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml
RD=artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness_kerem_run.json
VM_START_EPOCH=${VM_START_EPOCH:?}; VM_HARD_STOP_H=${VM_HARD_STOP_H:-78}; HOURLY_TRY=${HOURLY_TRY:-43}
PILOT_STEPS=${PILOT_STEPS:-6000}; FINAL_WAIT_MIN=${FINAL_WAIT_MIN:-60}
COMMON="KEREM_WEIGHT_DECAY=0.05 KEREM_FAMILY_DEPTH=6 KEREM_CLEAN_ANCHOR=1 KEREM_RANK_WEIGHT=0 KEREM_NN_FLOOR=0.3 KEREM_EXTRA_CAMERA=${KEREM_EXTRA_CAMERA:-seaclear,uvvid} KEREM_EXTRA_SONAR=${KEREM_EXTRA_SONAR:-catalunya,china_offshore,aquascan,uatd}"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/orchestrate.log; }

[ "$(git status --porcelain --untracked-files=no | wc -l)" = "0" ] || { log "ABORT dirty tracked tree"; exit 2; }
uv run conrad train osfm-v11-10p-readiness --output $RD > $L/readiness.out 2>&1 || true
BIDS=$(python3 -c "import json;r=json.load(open('$RD'));print(r['decision']+'|'+','.join(sorted(str(i['id']) for i in r['items'] if i['status']=='BLOCKER')))")
case "$BIDS" in "READY FOR KEREM|") OVR="";; "NOT READY FOR KEREM|2,4,8") OVR="--kerem-override-unreviewed-trainer";; *) log "ABORT readiness $BIDS"; exit 3;; esac
log "commit $(git rev-parse HEAD) readiness $BIDS"

run_pilot(){ # $1 tag, $2 env
  local PID=pilot${PILOT_STEPS}-$1-nonpromotable-$(date -u +%Y%m%dT%H%M%SZ)
  log "pilot $1 start $PID ($2)"
  env $COMMON KEREM_LR=1e-4 $2 uv run conrad train osfm-v11-10p-launch --config $CFG --readiness $RD --run-id $PID --max-steps $PILOT_STEPS $OVR > $L/$PID.out 2>&1
  log "pilot $1 exit $? $(tail -1 artifacts/runs/$PID/metrics.jsonl 2>/dev/null | cut -c1-500)"
  echo $PID > $L/pilot_$1
}
run_pilot A "KEREM_SONAR_ANCHOR=0"
run_pilot B "KEREM_SONAR_ANCHOR=1"
CHOICE=$(python3 - <<'PY'
import json, os
best = None
for tag, env in (("A", "KEREM_SONAR_ANCHOR=0"), ("B", "KEREM_SONAR_ANCHOR=1")):
    try:
        rid = open(os.path.expanduser(f"~/p4logs/pilot_{tag}")).read().strip()
        rows = [json.loads(l) for l in open(f"artifacts/runs/{rid}/metrics.jsonl")]
    except Exception:
        continue
    ok = [r for r in rows if r["step"] > 2000 and r.get("val/nn_temporal_hit_min", 0) >= 0.3]
    if not ok:
        continue
    score = max(r["val/representation_effective_rank"] for r in ok)
    cand = (score, tag, env, rows[-1].get("steps_per_s", 3.0))
    if best is None or cand > best:
        best = cand
print("" if best is None else f"{best[1]}|{best[2]}|{best[3]}|{best[0]}")
PY
)
[ -n "$CHOICE" ] || { log "ABORT no usable pilot"; exit 4; }
TAG=${CHOICE%%|*}; REST=${CHOICE#*|}; ENVSEL=${REST%%|*}; REST=${REST#*|}; RATE=${REST%%|*}; SCORE=${REST#*|}
log "choice pilot $TAG ($ENVSEL) best weakest-family rank $SCORE rate $RATE"

FIRST_T0=$(date -u +%s); LR=5e-5; FINAL_RID=""
for attempt in 1 2; do
  NOW=$(date -u +%s)
  STEPS=$(python3 -c "
used_run=($NOW-$FIRST_T0)/3600; used_vm=($NOW-$VM_START_EPOCH)/3600
budget=min(72.0-used_run, $VM_HARD_STOP_H-used_vm-1.5)-1.5
print(max(0, min(int($RATE*0.93*budget*3600), 1500000)//5000*5000))")
  [ "$STEPS" -ge 50000 ] || { log "attempt $attempt skipped: only $STEPS steps fit"; break; }
  RID=v11-p4-kerem-sonar5-a$attempt-$(date -u +%Y%m%dT%H%M%SZ); echo $RID > $L/run_id; echo $STEPS > $L/run_steps; FINAL_RID=$RID
  T0=$(date -u +%s)
  (env $COMMON $ENVSEL KEREM_LR=$LR setsid nohup uv run conrad train osfm-v11-10p-launch --config $CFG --readiness $RD --run-id $RID --max-steps $STEPS $OVR > $L/$RID.out 2>&1 < /dev/null &)
  sleep 60; TPID=$(pgrep -n -x conrad || true)
  [ -n "$TPID" ] || { log "attempt $attempt failed to start"; tail -20 $L/$RID.out >> $L/orchestrate.log; break; }
  SPENT=$(python3 -c "print(round(($T0-$VM_START_EPOCH)/3600*$HOURLY_TRY,2))")
  REMAIN_H=$(python3 -c "print(round(72.0-($T0-$FIRST_T0)/3600,3))")
  log "attempt $attempt LAUNCHED $RID pid=$TPID steps=$STEPS lr=$LR spent_before_try=$SPENT"
  (HOURLY_TRY=$HOURLY_TRY SPENT_BEFORE_TRY=$SPENT MAX_HOURS=$REMAIN_H MAX_TRY=8500 RANK_STOP=0 NN_COLLAPSE=0.15 \
    setsid nohup python3 scripts/kerem_p4/watchdog.py artifacts/runs/$RID $TPID $STEPS $T0 > $L/watchdog.out 2>&1 < /dev/null &)
  while kill -0 $TPID 2>/dev/null; do sleep 60; done
  sleep 30
  REASON=$(python3 -c "
import json,os
rows=[json.loads(l) for l in open(os.path.expanduser('~/p4logs/watchdog-$RID.jsonl'))]
print(next((r.get('reason') for r in reversed(rows) if r.get('action')=='TERMINATE'), 'none'))" 2>/dev/null || echo unknown)
  log "attempt $attempt ended reason=$REASON $(tail -1 artifacts/runs/$RID/metrics.jsonl 2>/dev/null | cut -c1-300)"
  [ "$REASON" = "semantic_collapse" ] || break
  LR=$(python3 -c "print($LR/2)")
done

if [ -n "$FINAL_RID" ]; then
  O=~/final_$FINAL_RID; mkdir -p $O
  env $COMMON uv run python scripts/kerem_p4/final_checks.py artifacts/runs/$FINAL_RID $O > $O/final_checks.out 2>&1; log "final checks exit $?"
  R=artifacts/runs/$FINAL_RID
  sha256sum $R/checkpoints/*.pt artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt > $O/sha256.txt
  cp -r $R/metrics.jsonl $R/events.jsonl $R/reports $R/config.resolved.yaml $R/git.json $R/manifests.json $R/environment.json $R/checkpoints/*.meta.json $O/ 2>/dev/null
  cp $L/orchestrate.log $L/watchdog-$FINAL_RID.jsonl $L/$FINAL_RID.out $L/readiness.out $L/gpu.csv $RD $O/ 2>/dev/null
  for t in A B; do P=$(cat $L/pilot_$t 2>/dev/null); [ -n "$P" ] && mkdir -p $O/pilots/$P && cp -r artifacts/runs/$P/metrics.jsonl artifacts/runs/$P/reports $O/pilots/$P/ 2>/dev/null; done
  tar czf ~/final_bundle.tgz -C ~ final_$FINAL_RID && log "bundle $(stat -c %s ~/final_bundle.tgz) bytes"
fi
log "ALL_DONE; powering off in $FINAL_WAIT_MIN minutes"
sleep $((FINAL_WAIT_MIN*60)); sudo shutdown -h now
