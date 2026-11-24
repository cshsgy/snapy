#!/bin/bash
# W2 sign-off (#274): head 9ecee6f5 vs main 6007eaa9 on CPU and CUDA (GPU 4): #348 harness, #349 dump set FP64/FP32, base-vs-base controls, serial ctest by name. Every gate echoes its command.
W=/home/sam/.local/share/slack-agents/ziwei/workspace; PY=$W/.venv/bin/python; KP=$W/aam/w1x/pkg_kintera_main; HP=$W/aam/w1x/pkg_pyharp_main
BASE=$W/aam/w1x/pkg_snapy_w2main_6007eaa9; HEAD=$W/aam/w1x/pkg_snapy_w2tip_9ecee6f5
R=$W/aam/w2so/runs; mkdir -p $R; H348=$W/aam/h348/tools/harness348/h348.py; D349=$W/aam/dumpset349/tools/dumpset349
export CUDA_VISIBLE_DEVICES=4; cd /tmp
run() { echo "\$ $*"; "$@"; }
for leg in cpu cuda; do
  for side in base head baseb; do
    PKG=$BASE; [ $side = head ] && PKG=$HEAD
    rm -rf $R/h348_${side}_$leg; echo "== $(date -u +%H:%M:%S) h348 $side $leg"; echo "\$ PYTHONPATH=pkg_snapy_$([ $side = head ] && echo w2tip_9ecee6f5 || echo w2main_6007eaa9) python h348.py --device $leg --out h348_${side}_$leg"; PYTHONPATH=$PKG:$HP:$KP $PY $H348 --device $leg --out $R/h348_${side}_$leg > $R/h348_${side}_$leg.log 2>&1; echo "H348_${side}_${leg}_RC=$? files=$(ls $R/h348_${side}_$leg 2>/dev/null | wc -l)"; grep -iE "refus|error|Traceback" $R/h348_${side}_$leg.log | sort | uniq -c | head -4 | cut -c1-160
    for dt in float64 float32; do tag=d349; [ $dt = float32 ] && tag=d349f32; rm -rf $R/${tag}_${side}_$leg; echo "== $(date -u +%H:%M:%S) dumpset349 $dt $side $leg"; PYTHONPATH=$PKG:$HP:$KP $PY $D349/dumpset349.py --device $leg --dtype $dt --out $R/${tag}_${side}_$leg > $R/${tag}_${side}_$leg.log 2>&1; echo "${tag}_${side}_${leg}_RC=$? bins=$(ls $R/${tag}_${side}_$leg 2>/dev/null | grep -c 'bin$')"; done
  done
  echo "== compare #348 $leg: \$ python h348.py --compare h348_base_$leg h348_head_$leg"; $PY $H348 --compare $R/h348_base_$leg $R/h348_head_$leg 2>&1 | tail -22; echo "H348_CMP_${leg}_RC=${PIPESTATUS[0]}"
  echo "== control #348 $leg: base vs baseb"; $PY $H348 --compare $R/h348_base_$leg $R/h348_baseb_$leg 2>&1 | tail -2; echo "H348_CMPBB_${leg}_RC=${PIPESTATUS[0]}"
  for tag in d349 d349f32; do echo "== compare #349 $tag $leg: \$ python compare.py ${tag}_base_$leg ${tag}_head_$leg"; $PY $D349/compare.py $R/${tag}_base_$leg $R/${tag}_head_$leg 2>&1 | tail -30; echo "${tag}_CMP_${leg}_RC=${PIPESTATUS[0]}"; echo "== control #349 $tag $leg"; $PY $D349/compare.py $R/${tag}_base_$leg $R/${tag}_baseb_$leg 2>&1 | tail -2; echo "${tag}_CMPBB_${leg}_RC=${PIPESTATUS[0]}"; done
done
echo "W2SO_GATES_DONE $(date -u +%H:%M:%S)"
HK=$HP:$KP; RE="axial|aam|closure|global_phase|coriolis|register|diffusion|radial|cubed|sphere|seam|exchange|implicit|vic|straka|shallow|redo|mesh|conservation|budget|options|gnomonic|momentum|hydro|split|combine|restart|output|coordinate|eos|riemann|reconstruct|scalar|fp32|sedimentation|forcing|capture|stage_executor|positivity|user|jit|netcdf"
ctest_tree() { t=$1; d=$2; echo "== $(date -u +%H:%M:%S) ctest-serial $t: \$ CUDA_VISIBLE_DEVICES=4 PET_MASTER_PORT=29730 ctest -j 1 --timeout 900 -R '<subset>' ($(cd $W/$d/build-x1 && ctest -N -R "$RE" | tail -1))"; (cd $W/$d/build-x1 && CUDA_VISIBLE_DEVICES=4 PET_MASTER_PORT=29730 PYTHONPATH=$W/aam/w1x/pkg_snapy_$t:$HK ctest -j 1 --timeout 900 -R "$RE" --output-on-failure > $W/aam/w2so/ctests_${t}.log 2>&1; echo "CTESTS_${t}_RC=$?"); grep -E "Test +#[0-9]+:" $W/aam/w2so/ctests_${t}.log | sed -E 's/^ *[0-9]+\/[0-9]+ +Test +#[0-9]+: +//; s/ +\.+ */ /' | awk '{print $1, $2}' | sort > $W/aam/w2so/ctests_${t}.byname; tail -2 $W/aam/w2so/ctests_${t}.log; }
ctest_tree w2main_6007eaa9 dev/snapy-w3-w2main
ctest_tree w2tip_9ecee6f5 dev/snapy-w3-w2tip
echo "== byname diff main -> head"; diff $W/aam/w2so/ctests_w2main_6007eaa9.byname $W/aam/w2so/ctests_w2tip_9ecee6f5.byname; echo "rc=$?"
echo "W2SO_POST_DONE $(date -u +%H:%M:%S)"
