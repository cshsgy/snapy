#!/bin/bash
# X1 golden gate driver: the golden part of the X1 gate scripts, with the local paths as arguments.
# usage: run_gate.sh DEVICE BASE_PREFIX STEP_PREFIX [MUTANT_PREFIX]
#   DEVICE        cpu or cuda
#   *_PREFIX      install prefix of each snapy build (pip install -t PREFIX, see build_snapy.sh)
#   PYTHON        env, python with torch/pyharp/kintera (default: python)
#   OUT           env, output directory (default: ./x1_golden_<DEVICE>)
# Runs from OUT: golden_base_a, golden_base_b, golden_step, golden_step_perturb (1-ulp control),
# golden_mutant (if MUTANT_PREFIX), then the compares. Each line "rc=N" is the exit code.
set -u
DEV=$1; BASE=$2; STEP=$3; MUT=${4:-}
PY=${PYTHON:-python}
H=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/x1_golden_$DEV}
mkdir -p "$OUT"; cd "$OUT"
[ "$DEV" = cpu ] && export CUDA_VISIBLE_DEVICES=
G="$PY $H/golden_x1.py"
log() { echo "== $(date -u +%H:%M:%S) $*"; }
log golden base_a; PYTHONPATH=$BASE $G run --device $DEV --out golden_base_a; echo "rc=$?"
log golden base_b; PYTHONPATH=$BASE $G run --device $DEV --out golden_base_b; echo "rc=$?"
log golden step;   PYTHONPATH=$STEP $G run --device $DEV --out golden_step; echo "rc=$?"
log golden step_perturb; PYTHONPATH=$STEP $G run --device $DEV --out golden_step_perturb --perturb; echo "rc=$?"
log compare base_a base_b; $G compare golden_base_a golden_base_b; echo "rc=$?"
log compare base_a step; $G compare golden_base_a golden_step; echo "rc=$?"
log compare base_b step; $G compare golden_base_b golden_step; echo "rc=$?"
log NEGATIVE compare base_a step_perturb; $G compare golden_base_a golden_step_perturb; echo "rc=$?"
if [ -n "$MUT" ]; then
  log golden mutant; PYTHONPATH=$MUT $G run --device $DEV --out golden_mutant; echo "rc=$?"
  log MUTANT compare base_a mutant; $G compare golden_base_a golden_mutant; echo "rc=$?"
fi
log done
