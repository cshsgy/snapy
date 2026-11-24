#!/bin/bash
# V3 speed audit: every cell 3 times, before/after interleaved, CPU and CUDA cells one at a time.
W=/home/sam/.local/share/slack-agents/ziwei/workspace; S=$W/aam/speed; R=${SPEED_RUNS:-$S/runs}; mkdir -p $R
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
L=$S/launch.sh
CPU_STEPS=${CPU_STEPS:-30}; CPU_WARM=3; GPU_STEPS=${GPU_STEPS:-100}; GPU_WARM=5
echo "RUN_ALL_START $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader | head -2
cell() {  # cell NAME TREE NPROC DEVICE THREADS args...
  name=$1; tree=$2; np=$3; dev=$4; thr=$5; shift 5
  for rep in 1 2 3; do :; done
}
run_cell() {  # run_cell NAME TREE NPROC DEV THREADS REP args...
  name=$1; tree=$2; np=$3; dev=$4; thr=$5; rep=$6; shift 6
  if [ $dev = cpu ]; then st=$CPU_STEPS; wm=$CPU_WARM; else st=$GPU_STEPS; wm=$GPU_WARM; fi
  OMP_NUM_THREADS=$thr $L $tree $np $dev $R/$name.$tree.r$rep --warmup $wm --steps $st --threads $thr "$@" 2>&1 | tail -1
}
both() {  # both NAME NPROC DEV THREADS args... (before and after, 3 reps interleaved)
  name=$1; np=$2; dev=$3; thr=$4; shift 4
  for rep in 1 2 3; do for tree in before after; do run_cell $name $tree $np $dev $thr $rep "$@"; done; done
}
after_only() {  # after_only NAME NPROC DEV THREADS args...
  name=$1; np=$2; dev=$3; thr=$4; shift 4
  for rep in 1 2 3; do run_cell $name after $np $dev $thr $rep "$@"; done
}
STR="--deck straka_hllc --init straka --nb2 1"
STRY="--deck ${SPEED_SRC_AFTER:-$W/dev/snapy-speed-after}/examples/straka.yaml --init straka --nb2 1"
C3="--deck $S/cart3d_pert_64.yaml"; C3N4="--deck $S/cart3d_pert_64_nb4.yaml"
CS1="--deck $S/cubed6_moist.yaml"; CS6="--deck $S/cubed6_moist_6ranks.yaml"
# S4 cells (Xi 06:33Z): V3 eager within 5% of main eager on c1, c1d, c3 and S3, CUDA and CPU; plus the graph cells that S1/S2 use
echo "== S4 c1: straka_hllc FP64 =="
both c1_cpu_fp64 1 cpu 8 $STR --precision float64
both c1_cuda_fp64_eager 1 cuda 8 $STR --precision float64 --execution eager
after_only c1_cuda_fp64_graph 1 cuda 8 $STR --precision float64 --execution cuda-graph
echo "== S4 c1d: shipped examples/straka.yaml =="
both c1d_cpu_fp64 1 cpu 8 $STRY
both c1d_cuda_fp64_eager 1 cuda 8 $STRY --execution eager
echo "== S4 c3: cubed6_moist =="
both c3_cpu_6rank 6 cpu 4 $CS6
both c3_cuda_eager 1 cuda 8 $CS1 --execution eager
echo "== S4 S3: cubed6_moist_implicit, dt varying =="
CSI="--deck $S/cubed6_moist_implicit.yaml --dt-mode varying"
both s3_cuda_eager_varying 1 cuda 8 $CSI --execution eager
after_only s3_cuda_graph_varying 1 cuda 8 $CSI --execution cuda-graph
echo "RUN_S4_DONE $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
