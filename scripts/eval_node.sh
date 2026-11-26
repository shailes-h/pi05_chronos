#!/bin/bash
# Full Conveyor eval (6 conditions x 3 paper scene batches x 50 episodes) of one or more π0.5 checkpoints on one node.
# Usage: eval_node.sh <out_root> <gpus_per_model> <label>:<config>:<ckpt_dir> [<label>:<config>:<ckpt_dir> ...]
# Each model gets <gpus_per_model> policy servers (one per GPU); its 18 jobs are spread over them round-robin.
set -u
source /scratch/Projects/CFP-05/CFP05-CF-002/chronos_pi05/env.sh
OUT=$1; G=$2; shift 2
CONDS="nominal sensor_id sensor_ood inference_0_100ms inference_0_200ms combined_sensor_ood_inference_0_200ms"
BATCHES="600301:701:709 600351:809:811 600851:1801:1811"   # paper scene batches (first scene, sensor seed, inference seed)
mkdir -p $OUT/logs; gpu=0; port=8100; jobs=$OUT/jobs_$(echo "$@" | md5sum | cut -c1-8).txt; : > $jobs; pids=(); wpids=()
for spec in "$@"; do
  IFS=: read label cfg dir <<< "$spec"; ports=()
  for i in $(seq 1 $G); do
    CUDA_VISIBLE_DEVICES=$((gpu % 8)) XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.5} nohup $W/openpi/.venv/bin/python $W/openpi/scripts/serve_policy.py \
      --port $port policy:checkpoint --policy.config=$cfg --policy.dir=$dir > $OUT/logs/server_${label}_$port.log 2>&1 &
    pids+=($!); ports+=($port); gpu=$((gpu+1)); port=$((port+1))
  done
  k=0
  for b in $BATCHES; do IFS=: read first ss is <<< "$b"; for c in $CONDS; do
    p=${ports[$((k % G))]}; k=$((k+1)); o=$OUT/$label/scenes$first/$c.json
    [ -f $o ] || echo "cd $W/chronos_policy && MUJOCO_EGL_DEVICE_ID=$(( (k - 1) % 8 )) $W/envs/chronos/bin/python -m pi05_chronos.eval_conveyor --port $p --condition $c --first-episode $first --sensor-seed $ss --inference-seed $is --episodes ${EPISODES:-50} --label $label:$dir --output $o > $OUT/logs/${label}_${first}_$c.log 2>&1" >> $jobs
  done; done
done
echo "$(date +%T) waiting for servers"; for p in $(seq 8100 $((port-1))); do
  until (echo > /dev/tcp/127.0.0.1/$p) 2>/dev/null; do sleep 5; done; done
for p in $(seq 8100 $((port-1))); do (cd $W/chronos_policy && $W/envs/chronos/bin/python -m pi05_chronos.eval_conveyor --warmup --port $p --output /dev/null) & wpids+=($!); done; wait ${wpids[@]}
echo "$(date +%T) servers up; running $(wc -l < $jobs) eval jobs"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
xargs -P 64 -I{} bash -c "{}" < $jobs
echo "$(date +%T) done: $(find $OUT -name '*.json' | wc -l) result files"
kill ${pids[@]} 2>/dev/null
