#!/bin/bash
# Train one π0.5 Conveyor variant at the largest global batch that fits (2048 -> 1024 -> 512), 30 epochs.
# Usage: train_run.sh <config> <exp_name>
set -u
source /scratch/Projects/CFP-05/CFP05-CF-002/chronos_pi05/env.sh
CFG=$1; EXP=$2; WINDOWS=${3:-27992}; EPOCHS=${EPOCHS:-30}
cd $W/openpi; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
mkdir -p $W/runs/$CFG
for B in ${BATCHES:-2048 1024 512}; do  # BATCHES="256" for small datasets (BiCAT)
  STEPS=$(( (EPOCHS * WINDOWS + B - 1) / B ))
  LR=$(python3 -c "print(f'{5e-5 * ($B / 256) ** 0.5:.3e}')")
  WARM=$(( STEPS / 10 )); VAL=$(( STEPS / 10 )); SAVE=${SAVE_EVERY:-$(( STEPS / 4 ))}  # SAVE_EVERY=STEPS -> final checkpoint only (~42 GB each)
  LOG=$W/logs/train_${CFG}_${EXP}_b$B.log
  echo "$(date +%T) $CFG batch $B steps $STEPS lr $LR warmup $WARM -> $LOG"
  cat > $W/runs/$CFG/${EXP}_meta.json <<JSON
{"config": "$CFG", "exp_name": "$EXP", "batch_size": $B, "num_train_steps": $STEPS, "peak_lr": $LR, "warmup_steps": $WARM,
 "epochs": $EPOCHS, "train_windows": $WINDOWS, "subset": "${SUBSET:-$W/data/conveyor_subsets/d500_v50.json}", "seed": 0,
 "openpi_sha": "$(git -C $W/openpi rev-parse HEAD)", "chronos_policy_sha": "$(git -C $W/chronos_policy rev-parse HEAD)",
 "host": "$(hostname)", "pbs_job": "${PBS_JOBID:-}", "start": "$(date -Is)"}
JSON
  .venv/bin/python scripts/train.py $CFG --exp-name=$EXP --overwrite --batch-size=$B --num-workers=64 \
    --num-train-steps=$STEPS --lr-schedule.warmup-steps=$WARM --lr-schedule.peak-lr=$LR \
    --lr-schedule.decay-steps=$STEPS --lr-schedule.decay-lr=$(python3 -c "print(f'{$LR/10:.3e}')") \
    --save-interval=$SAVE --keep-period=$STEPS --val-interval=$VAL --log-interval=10 > $LOG 2>&1
  rc=$?
  if grep -qE "EDQUOT|ENOSPC|No space left" $LOG; then echo "$(date +%T) DISK FULL (not OOM), stopping"; exit 3; fi
  if grep -qiE "RESOURCE_EXHAUSTED|out of memory" $LOG; then echo "$(date +%T) OOM at batch $B, retrying smaller"; continue; fi
  echo "$(date +%T) exit $rc"; exit $rc
done
