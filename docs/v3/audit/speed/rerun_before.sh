#!/bin/bash
# after RUN_ALL_DONE: the before-tree CUDA eager cells (730b3500 rejected the execution key on the first pass), then the strict-refusal record
W=/home/sam/.local/share/slack-agents/ziwei/workspace; S=$W/aam/speed; R=$S/runs; L=$S/launch.sh
export CUDA_VISIBLE_DEVICES=1
until grep -q "^RUN_ALL_DONE" $S/run_all.log; do sleep 20; done
echo "RERUN_START $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
rm -f $R/c1_cuda_fp64_eager.before.r*.rank0.* $R/c1_cuda_fp32_eager.before.r*.rank0.* $R/c1d_cuda_fp64_eager.before.r*.rank0.* $R/c2_cuda_eager.before.r*.rank0.* $R/c3_cuda_eager.before.r*.rank0.*
STR="--deck straka_hllc --init straka --nb2 1"; STRY="--deck $W/dev/snapy-speed-after/examples/straka.yaml --init straka --nb2 1"
for rep in 1 2 3; do
  OMP_NUM_THREADS=8 $L before 1 cuda $R/c1_cuda_fp64_eager.before.r$rep --warmup 5 --steps 100 --threads 8 $STR --precision float64 2>&1 | tail -1
  OMP_NUM_THREADS=8 $L before 1 cuda $R/c1_cuda_fp32_eager.before.r$rep --warmup 5 --steps 100 --threads 8 $STR --precision float32 2>&1 | tail -1
  OMP_NUM_THREADS=8 $L before 1 cuda $R/c1d_cuda_fp64_eager.before.r$rep --warmup 5 --steps 100 --threads 8 $STRY 2>&1 | tail -1
  OMP_NUM_THREADS=8 $L before 1 cuda $R/c2_cuda_eager.before.r$rep --warmup 5 --steps 100 --threads 8 --deck $S/cart3d_pert_64.yaml 2>&1 | tail -1
  OMP_NUM_THREADS=8 $L before 1 cuda $R/c3_cuda_eager.before.r$rep --warmup 5 --steps 100 --threads 8 --deck $S/cubed6_moist.yaml 2>&1 | tail -1
done
echo "== strict refusal record (after, cubed6, execution-strict: true)"
OMP_NUM_THREADS=8 $L after 1 cuda $R/c3_strict_refusal --warmup 1 --steps 1 --threads 8 --deck $S/cubed6_moist_strict.yaml --execution cuda-graph 2>&1 | tail -1
grep -h "RuntimeError" $R/c3_strict_refusal.rank0.log | head -2
echo "RERUN_DONE $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
