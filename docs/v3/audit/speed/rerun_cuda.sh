#!/bin/bash
# all CUDA cells again, before/after adjacent per repetition, on the least-busy GPU (Sam's training jobs hold all 8 GPUs, ~60 GB each)
W=/home/sam/.local/share/slack-agents/ziwei/workspace; S=$W/aam/speed; R=$S/runs; L=$S/launch.sh
pick_gpu() { best=; bestu=1000; for g in 0 1 2 3 4 5 6 7; do u=0; for i in 1 2 3 4 5; do u=$((u + $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g))); sleep 1; done; [ $u -lt $bestu ] && bestu=$u && best=$g; done; echo $best; }
export CUDA_VISIBLE_DEVICES=${GPU:-$(pick_gpu)}
echo "RERUN_CUDA_START $(date -u +%FT%TZ) gpu=$CUDA_VISIBLE_DEVICES load=$(cut -d' ' -f1-3 /proc/loadavg)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
mkdir -p $S/runs_pass1; mv $R/*cuda* $R/c4_* $S/runs_pass1/ 2>/dev/null
util() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i $CUDA_VISIBLE_DEVICES | tr -d ' '; }
rc() { local name=$1 tree=$2 rep=$3; shift 3; printf "gpu_util_before=%s " "$(util)"; OMP_NUM_THREADS=8 $L $tree 1 cuda $R/$name.$tree.r$rep --warmup 5 --steps 100 --threads 8 "$@" 2>&1 | tail -1; }
STR="--deck straka_hllc --init straka --nb2 1"; STRY="--deck $W/dev/snapy-speed-after/examples/straka.yaml --init straka --nb2 1"
C3="--deck $S/cart3d_pert_64.yaml"; CS1="--deck $S/cubed6_moist.yaml"
pair() { local name=$1; shift; for rep in 1 2 3; do rc $name before $rep "$@"; rc $name after $rep "$@"; done; }
eg() { local name=$1; shift; for rep in 1 2 3; do rc ${name}_eager after $rep "$@"; rc ${name}_graph after $rep "$@" --execution cuda-graph; done; }
echo "== pairs before/after (eager)"
pair c1_cuda_fp64_eager $STR --precision float64
pair c1_cuda_fp32_eager $STR --precision float32
pair c1d_cuda_fp64_eager $STRY
pair c2_cuda_eager $C3
pair c3_cuda_eager $CS1
echo "== eager vs graph (after)"
eg c1_cuda_fp64 $STR --precision float64
eg c1_cuda_fp32 $STR --precision float32
eg c1d_cuda_fp64 $STRY
eg c2_cuda $C3
eg c3_cuda $CS1
echo "== case 4"
for rep in 1 2 3; do rc c4_graph_fixed after $rep $STR --precision float64 --execution cuda-graph --dt-mode fixed; rc c4_graph_varying after $rep $STR --precision float64 --execution cuda-graph --dt-mode varying; rc c4_eager_varying after $rep $STR --precision float64 --dt-mode varying; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "RERUN_CUDA_DONE $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
