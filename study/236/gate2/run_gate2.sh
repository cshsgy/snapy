#!/bin/bash
# Gate 2 arm A, CPU only. One process per card. Does not start uranus.
set -u
export CUDA_VISIBLE_DEVICES=
ROOT=/tmp/snapy-236-g2
BIN=$(find "$ROOT/build-cpu" -type f -name 'gate2_arm_a.release' -perm -u+x | head -1)
if [ -z "$BIN" ]; then
  echo "missing gate2_arm_a.release" >&2
  exit 1
fi
OUT=$ROOT/study/236/gate2/raw
LOG=$ROOT/study/236/gate2/logs
mkdir -p "$OUT" "$LOG"
CARD=$ROOT/tests/test_flux_positivity_carry.yaml
CARDU=$ROOT/tests/test_flux_positivity_carry_vapor_u0.yaml
export LD_LIBRARY_PATH="/tmp/kintera-py/kintera/lib:${LD_LIBRARY_PATH:-}"
echo "BIN $BIN"
echo "kintera cmake 2.5.13"
rc=0
run() {
  local eos=$1 card=$2 case=$3
  echo "RUN $eos $case"
  if ! "$BIN" --card "$card" --eos "$eos" --case "$case" --out "$OUT" \
      >"$LOG/${eos}__${case}.log" 2>&1; then
    echo "RUN_FAIL $eos $case"
    rc=1
  fi
}
for eos in ideal-moist moist-mixture; do
  for c in adv-lmars adv-hllc settling x2 donors mixed; do
    run "$eos" "$CARD" "$c"
  done
  for c in vapor-u0-lmars vapor-u0-hllc; do
    run "$eos" "$CARDU" "$c"
  done
done
exit $rc
