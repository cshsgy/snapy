#!/bin/bash
# launch.sh TREE(before|after) NPROC DEVICE OUTPREFIX [bench args...]
# one rank per process; the nb/pz gate's env scheme (RANK/WORLD_SIZE/..., gloo on CPU, ucx on CUDA)
W=/home/sam/.local/share/slack-agents/ziwei/workspace; PY=$W/.venv/bin/python
TREE=$1; NPROC=$2; DEV=$3; OUT=$4; shift 4
# packages/sources overridable per run (closing dry run): SPEED_PKG_BEFORE/AFTER, SPEED_SRC_BEFORE/AFTER, SPEED_KINTERA_PKG
case $TREE in
  before) PKG=${SPEED_PKG_BEFORE:-$W/aam/w1x/pkg_snapy_sp_before_730b3500}; SRC=${SPEED_SRC_BEFORE:-$W/dev/snapy-speed-before};;
  after)  PKG=${SPEED_PKG_AFTER:-$W/aam/w1x/pkg_snapy_w3fin_d3178f20};     SRC=${SPEED_SRC_AFTER:-$W/dev/snapy-speed-after};;
esac
export PYTHONPATH=$PKG:$W/aam/w1x/pkg_pyharp_main:${SPEED_KINTERA_PKG:-$W/aam/w1x/pkg_kintera_main}
export DEVICE=$DEV OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
PIDS=()
if [ "$NPROC" -gt 1 ]; then
  PORT=$($PY -c "import socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1])")
  BACKEND=gloo; [ "$DEV" = cuda ] && BACKEND=ucx
  for r in $(seq 0 $((NPROC-1))); do
    RANK=$r WORLD_SIZE=$NPROC PROCESS_RANK=$r PROCESS_WORLD_SIZE=$NPROC LOCAL_RANK=$r LOCAL_WORLD_SIZE=$NPROC \
    MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT BACKEND=$BACKEND DEVICE_ID=0 \
    $PY $W/aam/speed/bench.py --tree $SRC --device $DEV --out $OUT.rank$r.json "$@" > $OUT.rank$r.log 2>&1 < /dev/null &
    PIDS+=($!)
  done
else
  DEVICE_ID=0 $PY $W/aam/speed/bench.py --tree $SRC --device $DEV --out $OUT.rank0.json "$@" > $OUT.rank0.log 2>&1 < /dev/null &
  PIDS+=($!)
fi
RC=0; for p in "${PIDS[@]}"; do wait $p || RC=$?; done
echo "LAUNCH_RC=$RC $OUT"; grep -h "^rank 0" $OUT.rank0.log | tail -1
exit $RC
