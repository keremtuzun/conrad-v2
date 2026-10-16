#!/usr/bin/env bash
# Round 4, stage 2: SubPipe per-stream batches. Replaces the remainder of orchestrate_persource.sh.
#   pilot S (P2 settings + SubPipe batches from one sonar stream at a time, SubPipe weight 2), PILOT_STEPS,
#   warm-started from INIT -> long run LONG_STEPS from S's best.pt if S beats INIT's selection rank by
#   MIN_GAIN, else from INIT with the P2 settings -> final checks + ceiling check + bundle -> power off.
# Env: VM_START_EPOCH, INIT (checkpoint), INIT_RUN (its run dir, for the baseline), HOURLY_TRY, MAX_TRY.
set -uo pipefail
export PATH=$HOME/.local/bin:$PATH
cd ~/conrad-v2
L=~/p4logs
CFG=configs/train/osfm/v11_p4_10p_829_semantic_candidate.yaml
RD=artifacts/gates/V1.1/10P_KEREM_HANDOFF/readiness_kerem_run.json
VM_START_EPOCH=${VM_START_EPOCH:?}; INIT=${INIT:?}; INIT_RUN=${INIT_RUN:?}; HOURLY_TRY=${HOURLY_TRY:-52}; MAX_TRY=${MAX_TRY:-2500}
R3_BEST=artifacts/runs/v11-p4-kerem-sonar5-a1-20261004T165839Z/checkpoints/best.pt
PILOT_STEPS=${PILOT_STEPS:-8000}; LONG_STEPS=${LONG_STEPS:-100000}; MIN_GAIN=${MIN_GAIN:-3}; FINAL_WAIT_MIN=${FINAL_WAIT_MIN:-60}
COMMON="KEREM_SELECT_EXCLUDE=aquascan KEREM_WEIGHT_DECAY=0.05 KEREM_FAMILY_DEPTH=6 KEREM_CLEAN_ANCHOR=1 KEREM_RANK_WEIGHT=0 KEREM_NN_FLOOR=0.3 KEREM_SONAR_ANCHOR=0 KEREM_SELECT=per_source KEREM_WARMUP=500 KEREM_LR=5e-5 KEREM_EXTRA_CAMERA=seaclear,uvvid KEREM_EXTRA_SONAR=catalunya,china_offshore,aquascan,uatd KEREM_SONAR_BATCH=single KEREM_SONAR_VIEWS=sonar"
STAGE2=${STAGE_ENV:-"KEREM_STREAM_BATCH_SOURCES=subpipe KEREM_SONAR_WEIGHTS=subpipe:2"}
BASE_ENV=${BASE_ENV:-}; TAG=${STAGE_TAG:-S}; LONG_PREFIX=${LONG_PREFIX:-v11-p4-kerem-persource-a2}
OVR="--kerem-override-unreviewed-trainer"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $L/orchestrate.log; }
spent(){ python3 -c "import time;print(round((time.time()-$VM_START_EPOCH)/3600*$HOURLY_TRY,1))"; }
best_sel(){ python3 -c "
import json
rows=[json.loads(l) for l in open('$1/metrics.jsonl')]
def sel(r):
    ranks=[r['val/imaging_sonar/effective_rank'], r['val/rgb_camera/effective_rank']]
    ranks+=[v for k,v in r.items() if k.startswith('val/imaging_sonar/') and k.count('/')==3 and k.endswith('effective_rank') and '/aquascan/' not in k]
    return min(ranks)
ok=[sel(r) for r in rows if r.get('val/nn_temporal_hit_min',0)>=0.3 and r['step']>500]
print(round(max(ok),1) if ok else 0)"; }

[ "$(git status --porcelain --untracked-files=no | wc -l)" = "0" ] || { log "ABORT dirty tracked tree"; exit 2; }
BASE=$(best_sel $INIT_RUN)
log "stage 2 commit $(git rev-parse --short HEAD) init $INIT baseline selection $BASE spent $(spent)"
PID=pilot${PILOT_STEPS}-$TAG-subpipe-nonpromotable-$(date -u +%Y%m%dT%H%M%SZ); echo $PID > $L/pilot_$TAG
log "pilot $TAG start $PID ($STAGE2)"
env $COMMON $STAGE2 KEREM_INIT_FROM=$INIT uv run conrad train osfm-v11-10p-launch --config $CFG --readiness $RD --run-id $PID --max-steps $PILOT_STEPS $OVR > $L/$PID.out 2>&1; RC=$?
SSEL=$(best_sel artifacts/runs/$PID)
log "pilot $TAG exit $RC best selection $SSEL spent $(spent) $(tail -1 artifacts/runs/$PID/metrics.jsonl 2>/dev/null | python3 -c "import json,sys;r=json.loads(sys.stdin.read());print({k.split('/')[2]:round(v) for k,v in r.items() if k.startswith('val/imaging_sonar/') and k.count('/')==3 and k.endswith('effective_rank')}, 'nn', round(r['val/nn_temporal_hit_min'],2))" 2>/dev/null)"
if python3 -c "import sys;sys.exit(0 if $SSEL >= $BASE + $MIN_GAIN else 1)" && [ -f artifacts/runs/$PID/checkpoints/best.pt ]; then
  LENV="$STAGE2"; LINIT=artifacts/runs/$PID/checkpoints/best.pt; log "long run uses pilot $TAG settings"
else
  LENV="$BASE_ENV"; LINIT=$INIT; log "pilot $TAG did not beat the baseline by $MIN_GAIN; long run continues from INIT with: ${BASE_ENV:-P2 settings}"
fi
RID=$LONG_PREFIX-$(date -u +%Y%m%dT%H%M%SZ); echo $RID > $L/run_id; echo $LONG_STEPS > $L/run_steps
T0=$(date -u +%s); SPENT=$(spent)
(env $COMMON $LENV KEREM_INIT_FROM=$LINIT setsid nohup uv run conrad train osfm-v11-10p-launch --config $CFG --readiness $RD --run-id $RID --max-steps $LONG_STEPS $OVR > $L/$RID.out 2>&1 < /dev/null &)
sleep 90; TPID=$(pgrep -n -x conrad || true)
FINAL_RID=$RID
if [ -n "$TPID" ]; then
  log "long run LAUNCHED $RID pid=$TPID steps=$LONG_STEPS init=$LINIT spent_before=$SPENT"
  (HOURLY_TRY=$HOURLY_TRY SPENT_BEFORE_TRY=$SPENT MAX_HOURS=30 MAX_TRY=$MAX_TRY RANK_STOP=0 NN_COLLAPSE=0.15 \
    setsid nohup python3 scripts/kerem_p4/watchdog.py artifacts/runs/$RID $TPID $LONG_STEPS $T0 > $L/watchdog.out 2>&1 < /dev/null &)
  while kill -0 $TPID 2>/dev/null; do sleep 60; done
  sleep 30
  log "long run ended spent $(spent) $(tail -1 artifacts/runs/$RID/metrics.jsonl 2>/dev/null | cut -c1-300)"
else
  log "long run failed to start"; tail -20 $L/$RID.out >> $L/orchestrate.log; FINAL_RID=$PID
fi

O=~/final_$FINAL_RID; mkdir -p $O
R=artifacts/runs/$FINAL_RID
env $COMMON uv run python scripts/kerem_p4/final_checks.py $R $O > $O/final_checks.out 2>&1; log "final checks exit $?"
env $COMMON uv run python scripts/kerem_p4/ceiling_check.py $O/ceiling_after.json round3_best=$R3_BEST final_best=$R/checkpoints/best.pt > $O/ceiling_after.out 2>&1; log "ceiling after exit $?"
sha256sum $R/checkpoints/*.pt $INIT $R3_BEST artifacts/runs/train-osfm_u1_sonar_research-1790575109765916596/checkpoints/osfm_u1_sonar_research_full.pt > $O/sha256.txt
cp -r $R/metrics.jsonl $R/events.jsonl $R/reports $R/config.resolved.yaml $R/git.json $R/manifests.json $R/environment.json $R/checkpoints/*.meta.json $O/ 2>/dev/null
cp $L/orchestrate.log $L/ceiling_before.json $L/ceiling_before.out $L/watchdog-*.jsonl $L/$FINAL_RID.out $L/readiness.out $L/gpu.csv $L/subpipe_diag.json $RD $O/ 2>/dev/null
for t in P1 P2 P3 S T; do P=$(cat $L/pilot_$t 2>/dev/null); [ -n "$P" ] && mkdir -p $O/pilots/$P && cp -r artifacts/runs/$P/metrics.jsonl artifacts/runs/$P/reports $O/pilots/$P/ 2>/dev/null; done
for SR in artifacts/runs/v11-p4-kerem-persource-a*; do [ "$SR" = "$R" ] || { mkdir -p $O/stopped/$(basename $SR) && cp -r $SR/metrics.jsonl $SR/reports $O/stopped/$(basename $SR)/ 2>/dev/null; }; done
tar czf ~/final_bundle.tgz -C ~ final_$FINAL_RID && log "bundle $(stat -c %s ~/final_bundle.tgz) bytes"
log "ALL_DONE spent $(spent); powering off in $FINAL_WAIT_MIN minutes"
sleep $((FINAL_WAIT_MIN*60)); sudo shutdown -h now
