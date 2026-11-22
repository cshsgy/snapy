#!/bin/bash
# speed audit "before": snapy v2.14.1 730b3500, same flags as pkg_snapy_w3fin_d3178f20 (CUDA=ON sm_90, Release, NETCDF=OFF, UCX=ON, kintera main 2.6.11, pyharp main)
set -x
W=/home/sam/.local/share/slack-agents/ziwei/workspace
VENV=$W/.venv; SP=$VENV/lib/python3.12/site-packages; export CUDACXX=/usr/local/cuda/bin/nvcc
KP=$W/aam/w1x/pkg_kintera_main; HP=$W/aam/w1x/pkg_pyharp_main
cd $W/dev/snapy-speed-before || exit 1
git rev-parse HEAD
PYTHONPATH=$HP:$KP /usr/bin/cmake -B build-x1 -DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DNETCDF=OFF -DPNETCDF=OFF -DUCX=ON -DUCX_INCLUDE_DIR=$SP/commux/include -DPython3_EXECUTABLE=$VENV/bin/python "-DCMAKE_PREFIX_PATH=$SP/torch/share/cmake;$KP/kintera;$HP/pyharp;$SP/pydisort" 2>&1 | tail -12
PYTHONPATH=$HP:$KP /usr/bin/cmake --build build-x1 --parallel 32 > $W/aam/speed/before_build.log 2>&1; echo SNAPY_SPB_BUILD_EXIT=$?
tail -3 $W/aam/speed/before_build.log
ln -sfn build-x1 build; [ -f build/lib/libnetcdf.so ] || echo '' | g++ -shared -fPIC -o build/lib/libnetcdf.so -x c++ -
PKG=$W/aam/w1x/pkg_snapy_sp_before_730b3500; mkdir -p $PKG
PYTHONPATH=$HP:$KP $VENV/bin/python -m pip install --no-build-isolation --no-deps --target $PKG . > $W/aam/speed/before_pip.log 2>&1; echo SNAPY_SPB_PIP_EXIT=$?
mkdir -p $PKG/snapy/lib; cp build/lib/libsnap*.so build/lib/libnetcdf.so $PKG/snapy/lib/; ls $PKG/snapy/lib
cd /tmp && PYTHONPATH=$PKG:$HP:$KP $VENV/bin/python -c "import snapy,pyharp,kintera;print('SNAPY_SPB_IMPORT_OK', snapy.__file__, kintera.__version__)" 2>&1 | tail -2
echo SNAPY_SPB_DONE
