#!/bin/bash
# Build one snapy tree and install it into a prefix, with the flags used for the X1 gate.
# usage: build_snapy.sh SRC PREFIX cpu|cuda
#   PYTHON  env, python with torch/pyharp/kintera installed (default: python)
#   ARCH    env, CUDA architecture for cuda builds (default: 75; the X1 runs used sm_75)
#   NJ      env, parallel build jobs (default: 6)
# Base and step must be built with identical flags.
set -eu
SRC=$1; PREFIX=$2; KIND=$3
PY=$(command -v "${PYTHON:-python}")
FLAGS="-G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DNETCDF=ON -DUCX=OFF -DPNETCDF=OFF -DPython3_EXECUTABLE=$PY -DSNAPY_TEST_PYTHONPATH=$PREFIX"
case $KIND in cuda) FLAGS="$FLAGS -DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${ARCH:-75}";; *) FLAGS="$FLAGS -DCUDA=OFF";; esac
cd "$SRC"
cmake -B build $FLAGS
cmake --build build --parallel ${NJ:-6}
rm -rf "$PREFIX"
MAX_JOBS=${NJ:-6} "$PY" -m pip install --no-deps --no-build-isolation -t "$PREFIX" .
