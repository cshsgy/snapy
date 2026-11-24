#!/bin/bash
# S3 (Xi 05:49Z): cubed6_moist, 6 blocks in one process, implicit on (scheme 9), dt changing every step; before = 730b3500 eager, after = V3 eager and cuda-graph; 3 reps interleaved. Uses the same SPEED_* env as run_all.sh.
W=/home/sam/.local/share/slack-agents/ziwei/workspace; S=$W/aam/speed; R=${SPEED_RUNS:-$S/runs}; mkdir -p $R; L=$S/launch.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
GPU_STEPS=${GPU_STEPS:-100}; GPU_WARM=5
echo "S3_START $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$CUDA_VISIBLE_DEVICES"
CSI="--deck $S/cubed6_moist_implicit.yaml"
for rep in 1 2 3; do
  OMP_NUM_THREADS=8 $L before 1 cuda $R/s3_cuda_eager_varying.before.r$rep --warmup $GPU_WARM --steps $GPU_STEPS --threads 8 $CSI --dt-mode varying 2>&1 | tail -1
  OMP_NUM_THREADS=8 $L after  1 cuda $R/s3_cuda_eager_varying.after.r$rep  --warmup $GPU_WARM --steps $GPU_STEPS --threads 8 $CSI --execution eager --dt-mode varying 2>&1 | tail -1
  OMP_NUM_THREADS=8 $L after  1 cuda $R/s3_cuda_graph_varying.after.r$rep  --warmup $GPU_WARM --steps $GPU_STEPS --threads 8 $CSI --execution cuda-graph --dt-mode varying 2>&1 | tail -1
done
# also the fixed-dt graph cell for the recapture comparison
for rep in 1 2 3; do OMP_NUM_THREADS=8 $L after 1 cuda $R/s3_cuda_graph_fixed.after.r$rep --warmup $GPU_WARM --steps $GPU_STEPS --threads 8 $CSI --execution cuda-graph --dt-mode fixed 2>&1 | tail -1; done
echo "S3_DONE $(date -u +%FT%TZ)"
