#!/bin/bash
# Closing dry run (plan f5968106, round 3): dryrun.sh HEAD_SHA NAME
#  builds HEAD on kintera 2.6.12 (pkg_0cfea06) and 2.6.13 (pkg_bcc3419); correctness legs base deeabfee@2.6.12 vs HEAD@2.6.12;
#  V3-only leg HEAD@2.6.12 vs HEAD@2.6.13; speed table before=730b3500@2.6.12 after=HEAD@2.6.12. Markers DRYRUN_<NAME>_*.
set -u
W=/home/sam/.local/share/slack-agents/ziwei/workspace; VENV=$W/.venv; PY=$VENV/bin/python; SP=$VENV/lib/python3.12/site-packages; export CUDACXX=/usr/local/cuda/bin/nvcc
SHA=${1:?head sha}; NAME=${2:?name}; O=$W/aam/final; R=$W/aam/w2r/runs/dryrun_$NAME; mkdir -p "$R"
K12=$W/aam/rows/pkg_0cfea06; K13=$W/aam/rows/pkg_bcc3419; HP=$W/aam/w1x/pkg_pyharp_main
BASE=$W/aam/w1x/pkg_snapy_deeabfee_k2612; BASESRC=$W/dev/snapy-deeabfee_k2612; SPEEDBASE=$W/aam/w1x/pkg_snapy_730b3500_k2612; SPEEDBASESRC=$W/dev/snapy-730b3500_k2612
D349=$W/aam/dumpset349/tools/dumpset349; H348=$W/aam/w2so/h348_face.py; NBPZ=$W/aam/nbpz/tools/gate_nbpz/gate_nbpz.py; XG=$W/aam/x1g/tools/v3/x1_golden/run_gate.sh
RE=$(grep -m1 'RE="' $W/aam/w2so/gates.sh | sed 's/.*RE="//; s/".*//')
echo "DRYRUN_${NAME}_START $(date -u +%FT%TZ) head=$SHA base=deeabfee@2.6.12 speedbase=730b3500@2.6.12 load=$(cut -d' ' -f1-3 /proc/loadavg)"
for p in $BASE $SPEEDBASE; do [ -f $p/snapy/__init__.py ] || echo "DRYRUN_${NAME}_MISSING $p"; done
build() { TAG=$1; KP=$2; D=$W/dev/snapy-$NAME; [ -d $D ] || git -C $W/dev/snapy worktree add -q --detach $D $SHA || { echo "DRYRUN_${NAME}_WORKTREE_FAILED"; exit 1; }
  cd $D && git checkout -q --detach $SHA; B=build-$TAG; HK=$HP:$KP
  # the move (f7034194) raised find_package(Kintera 2.6.13): for the same-kintera comparison the 2.6.12 build lowers that
  # one floor line for the configure only (build policy, not code); restored after the build and declared in the post
  if [ $TAG = k2612 ]; then sed -i 's/find_package(Kintera 2\.6\.13 REQUIRED)/find_package(Kintera 2.6.12 REQUIRED)/' CMakeLists.txt; echo "DRYRUN_${NAME}_${TAG}_FLOOR_LOWERED $(grep -n 'find_package(Kintera' CMakeLists.txt)"; fi
  PYTHONPATH=$HK /usr/bin/cmake -B $B -DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DNETCDF=OFF -DPNETCDF=OFF -DUCX=ON -DUCX_INCLUDE_DIR=$SP/commux/include -DPython3_EXECUTABLE=$PY "-DCMAKE_PREFIX_PATH=$SP/torch/share/cmake;$KP/kintera;$HP/pyharp;$SP/pydisort" > $O/cmake_${NAME}_$TAG.log 2>&1; echo "DRYRUN_${NAME}_${TAG}_CMAKE_EXIT=$?"
  PYTHONPATH=$HK /usr/bin/cmake --build $B --parallel 24 > $O/build_${NAME}_$TAG.log 2>&1; echo "DRYRUN_${NAME}_${TAG}_BUILD_EXIT=$? $(date -u +%FT%TZ) warnings=$(grep -c 'warning:' $O/build_${NAME}_$TAG.log)"; grep -m3 "error" $O/build_${NAME}_$TAG.log | cut -c1-200
  rm -f "$D/build"; ln -sfn $B build; [ -f $B/lib/libnetcdf.so ] || echo '' | g++ -shared -fPIC -o $B/lib/libnetcdf.so -x c++ -
  PKG=$W/aam/w1x/pkg_snapy_${NAME}_$TAG; rm -rf "${PKG:?}"; mkdir -p $PKG; PYTHONPATH=$HK $PY -m pip install --no-build-isolation --no-deps --target $PKG . > $O/pip_${NAME}_$TAG.log 2>&1; echo "DRYRUN_${NAME}_${TAG}_PIP_EXIT=$?"
  mkdir -p $PKG/snapy/lib; cp $B/lib/libsnap*.so $B/lib/libnetcdf.so $PKG/snapy/lib/
  (cd /tmp && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PKG:$HK $PY -c "import snapy,kintera; print('DRYRUN_${NAME}_${TAG}_IMPORT_OK kintera', kintera.__version__)" 2>&1 | tail -1)
  cd $D && git checkout -q -- CMakeLists.txt && echo "DRYRUN_${NAME}_${TAG}_TREE_RESTORED $(git status --short | grep -v '^??' | wc -l) dirty"; }
build k2612 $K12; build k2613 $K13
STEP=$W/aam/w1x/pkg_snapy_${NAME}_k2612; STEP13=$W/aam/w1x/pkg_snapy_${NAME}_k2613; HK12=$HP:$K12; HK13=$HP:$K13
pick_gpu() { best=; bestu=1000; for g in 0 1 2 3 4 5 6 7; do u=0; for i in 1 2 3; do u=$((u + $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g))); sleep 1; done; [ $u -lt $bestu ] && bestu=$u && best=$g; done; echo $best; }
GPU=$(pick_gpu); echo "DRYRUN_${NAME}_GPU=$GPU load=$(cut -d' ' -f1-3 /proc/loadavg)"
d349() { side=$1; PKG=$2; HK=$3; leg=$4; dt=$5; d=d349; [ $dt = float32 ] && d=d349f32; for a in 1 2 3; do rm -rf "${R:?}/${d:?}_${side:?}_${leg:?}"; (cd /tmp && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PKG:$HK $PY $D349/dumpset349.py --device $leg --dtype $dt --out $R/${d}_${side}_$leg > $R/${d}_${side}_$leg.log 2>&1); rc=$?; n=$(ls $R/${d}_${side}_$leg 2>/dev/null | grep -c 'bin$'); echo "DRYRUN_${NAME}_${d^^}_${side}_${leg}_RC=$rc bins=$n attempt=$a"; [ $n -ge 23 ] && break; [ $leg = cpu ] && break; sleep 60; done; }
h348() { side=$1; PKG=$2; HK=$3; leg=$4; rm -rf "${R:?}/h348_${side:?}_${leg:?}"; (cd /tmp && CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PKG:$HK $PY $H348 --device $leg --out $R/h348_${side}_$leg > $R/h348_${side}_$leg.log 2>&1); echo "DRYRUN_${NAME}_H348_${side}_${leg}_RC=$? files=$(ls $R/h348_${side}_$leg 2>/dev/null | wc -l)"; }
# ---- correctness legs: base_a, base_b (control), step (HEAD@2.6.12), step13 (HEAD@2.6.13)
for leg in cpu cuda; do for dt in float64 float32; do d349 basea $BASE $HK12 $leg $dt; d349 baseb $BASE $HK12 $leg $dt; d349 step $STEP $HK12 $leg $dt; d349 step13 $STEP13 $HK13 $leg $dt; done
  h348 basea $BASE $HK12 $leg; h348 baseb $BASE $HK12 $leg; h348 step $STEP $HK12 $leg; done
for leg in cpu cuda; do for d in d349 d349f32; do
  $PY $D349/compare.py $R/${d}_basea_$leg $R/${d}_baseb_$leg > $R/cmp_${d}_control_$leg.log 2>&1; echo "DRYRUN_${NAME}_${d^^}_CONTROL_${leg}_RC=$? $(tail -1 $R/cmp_${d}_control_$leg.log | cut -c1-120)"
  $PY $D349/compare.py $R/${d}_basea_$leg $R/${d}_step_$leg > $R/cmp_${d}_step_$leg.log 2>&1; echo "DRYRUN_${NAME}_${d^^}_BASE_VS_V3_${leg}_RC=$? $(tail -1 $R/cmp_${d}_step_$leg.log | cut -c1-120)"
  $PY $D349/compare.py $R/${d}_step_$leg $R/${d}_step13_$leg > $R/cmp_${d}_k13_$leg.log 2>&1; echo "DRYRUN_${NAME}_${d^^}_V3_2612_VS_2613_${leg}_RC=$? $(tail -1 $R/cmp_${d}_k13_$leg.log | cut -c1-120)"; done
  $PY $H348 --compare $R/h348_basea_$leg $R/h348_baseb_$leg 2>&1 | tail -1 | sed "s/^/DRYRUN_${NAME}_H348_CONTROL_$leg: /"; $PY $H348 --compare $R/h348_basea_$leg $R/h348_step_$leg 2>&1 | tail -2 | sed "s/^/DRYRUN_${NAME}_H348_BASE_VS_V3_$leg: /"; done
# ---- nb/pz (CPU): base, step, step13
for side in base step step13; do case $side in base) PKG=$BASE; HK=$HK12;; step) PKG=$STEP; HK=$HK12;; step13) PKG=$STEP13; HK=$HK13;; esac; rm -rf "${R:?}/nbpz_${side:?}"; (cd /tmp && CUDA_VISIBLE_DEVICES= PYTHONPATH=$PKG:$HK timeout 7200 $PY $NBPZ --device cpu --out $R/nbpz_$side > $R/nbpz_$side.log 2>&1; echo "DRYRUN_${NAME}_NBPZ_${side}_EXIT=$?"); done
$PY $NBPZ --compare $R/nbpz_base $R/nbpz_step > $R/nbpz_cmp_step.log 2>&1; echo "DRYRUN_${NAME}_NBPZ_BASE_VS_V3=$? $(tail -1 $R/nbpz_cmp_step.log | cut -c1-140)"
$PY $NBPZ --compare $R/nbpz_step $R/nbpz_step13 > $R/nbpz_cmp_k13.log 2>&1; echo "DRYRUN_${NAME}_NBPZ_V3_2612_VS_2613=$? $(tail -1 $R/nbpz_cmp_k13.log | cut -c1-140)"
# ---- x1_golden (cpu, cuda): base vs step (run_gate does base_a/base_b/step/step_perturb)
for leg in cpu cuda; do OUT=$R/x1g_$leg PYTHON=$PY CUDA_VISIBLE_DEVICES=$GPU bash $XG $leg "$BASE:$HK12" "$STEP:$HK12" > $R/x1g_$leg.log 2>&1; echo "DRYRUN_${NAME}_X1G_${leg}_EXIT=$?"; grep -E "^rc=|identical|differ" $R/x1g_$leg.log | tail -6 | sed "s/^/DRYRUN_${NAME}_X1G_$leg: /"; done
# ---- serial ctest by name (gates.sh subset), GPU visible: base tree, step tree (2.6.12), step13 tree (2.6.13)
ct() { tag=$1; D=$2; PKG=$3; HK=$4; port=$5; (cd $D && CUDA_VISIBLE_DEVICES=$GPU PET_MASTER_PORT=$port PYTHONPATH=$PKG:$HK ctest -j 1 --timeout 900 -R "$RE" --output-on-failure > $O/ctests_${NAME}_$tag.log 2>&1; echo "DRYRUN_${NAME}_CTESTS_${tag}_EXIT=$?"); grep -E "tests passed" $O/ctests_${NAME}_$tag.log | sed "s/^/DRYRUN_${NAME}_CTESTS_$tag: /"
  grep -E "Test +#[0-9]+:" $O/ctests_${NAME}_$tag.log | sed -E 's/^ *[0-9]+\/[0-9]+ +Test +#[0-9]+: +//; s/ +\.+ */ /' | awk '{print $1, $2}' | sort > $O/ctests_${NAME}_$tag.byname; }
ct base $BASESRC/build-x1 $BASE $HK12 29991; ct step $W/dev/snapy-$NAME/build-k2612 $STEP $HK12 29992; ct step13 $W/dev/snapy-$NAME/build-k2613 $STEP13 $HK13 29993
echo "== byname base -> step"; diff $O/ctests_${NAME}_base.byname $O/ctests_${NAME}_step.byname; echo "== byname step(2.6.12) -> step13(2.6.13)"; diff $O/ctests_${NAME}_step.byname $O/ctests_${NAME}_step13.byname && echo "DRYRUN_${NAME}_BYNAME_K13_IDENTICAL"
# ---- speed table (before = 730b3500@2.6.12, after = HEAD@2.6.12), one GPU
export SPEED_PKG_BEFORE=$SPEEDBASE SPEED_SRC_BEFORE=$SPEEDBASESRC SPEED_PKG_AFTER=$STEP SPEED_SRC_AFTER=$W/dev/snapy-$NAME SPEED_KINTERA_PKG=$K12 SPEED_RUNS=$W/aam/speed/runs_$NAME SPEED_SUMMARY=$W/aam/speed/summary_$NAME.json SPEED_DOC=$W/aam/speed/speed_$NAME.md
CUDA_VISIBLE_DEVICES=$GPU bash $W/aam/speed/run_all.sh > $O/speed_$NAME.log 2>&1; echo "DRYRUN_${NAME}_SPEED_EXIT=$?"; $PY $W/aam/speed/summarize.py > $O/speed_${NAME}_table.md 2>&1; echo "DRYRUN_${NAME}_SUMMARIZE_EXIT=$?"; $PY $W/aam/speed/gen_doc.py > /dev/null 2>&1; echo "DRYRUN_${NAME}_GENDOC_EXIT=$?"
echo "DRYRUN_${NAME}_DONE $(date -u +%FT%TZ)"
