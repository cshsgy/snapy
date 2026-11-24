# X1 golden gate

The step-by-step golden used to gate V3 X1 (`32f2c7171881cb43b5950ac1e31d6e121723e0ed`
on base `6e9c7d5fa10873ee9d25711456452b7158ac5e74`). It is the X1 gate of X.rst
(`07d4eedf`, section 5): an N-step golden of `hydro_u` and `scalar_s`, Cartesian and
cubed sphere, FP64 and FP32, base vs step bitwise.

Files:

- `golden_x1.py`: runs the five cases and compares two runs byte for byte.
- `run_gate.sh`: the gate sequence. It runs base twice, the step, the 1-ulp control and,
  optionally, the mutant, then the compares.
- `build_snapy.sh`: builds one tree into an install prefix with the gate's flags.
- `mutant_u0_fp32.patch`: the code-level negative control. It applies to 32f2c717 and
  creates `u0` as FP32 regardless of the policy.

## Cases

Each case runs 10 steps, writing `hydro_u` and `scalar_s` after every step: 20 dumps per
single-block case and 120 for the cubed sphere, 200 in total.

| case | deck |
|---|---|
| `cart_moist_fp64`, `cart_moist_fp32` | `tests/test_anomaly_storage.yaml` with anomaly + perturbation flux, HLLC, a passive scalar |
| `jupiter_crm_fp64`, `jupiter_crm_fp32` | `examples/jupiter_crm.yaml` reduced as in `tests/test_fp32_jupiter.py`, VIC scheme 9, dt 0.1 |
| `cubed6_fp64` | six gnomonic panels from `tests/test_axial_momentum_flux_form.{yaml,py}` with the moist arm |

There is no cubed-sphere FP32 case. FP32 needs anomaly storage, which works only on
Cartesian grids, so the case fails on base and on step alike.

All inputs are read from the checkout's `tests/` and `examples/`, which are the same at the
base and at the step. Set `SNAPY_TESTS` to use another `tests/` directory. The snapy under
test is whatever `import snapy` resolves to: `run_gate.sh` sets `PYTHONPATH` to the given
prefix. The script needs torch, pyharp, kintera, numpy and PyYAML.

## Commands

The provider stack for the published numbers was pyharp 2.7.17 (`6045dc9f`) and
kintera 2.6.11 (`431fd144`), both built from source, with torch 2.10.0+cu128 and
pydisort 1.8.13.

```bash
# trees: base, step, and (CPU only) mutant = step + patch
git worktree add base 6e9c7d5fa10873ee9d25711456452b7158ac5e74
git worktree add step 32f2c7171881cb43b5950ac1e31d6e121723e0ed
git worktree add mut  32f2c7171881cb43b5950ac1e31d6e121723e0ed
git -C mut apply tools/v3/x1_golden/mutant_u0_fp32.patch

# identical flags for base and step:
# Release, Ninja, BUILD_EXAMPLES=ON, NETCDF=ON, UCX=OFF, PNETCDF=OFF, CUDA=ON|OFF, CUDA arch 75
H=step/tools/v3/x1_golden
PYTHON=python $H/build_snapy.sh base $PWD/prefix-base-cpu cpu
PYTHON=python $H/build_snapy.sh step $PWD/prefix-step-cpu cpu
PYTHON=python $H/build_snapy.sh mut  $PWD/prefix-mut-cpu  cpu
ARCH=75 PYTHON=python $H/build_snapy.sh base $PWD/prefix-base-cuda cuda
ARCH=75 PYTHON=python $H/build_snapy.sh step $PWD/prefix-step-cuda cuda

# gate (the mutant prefix is optional)
OUT=$PWD/x1-cpu  PYTHON=python $H/run_gate.sh cpu  $PWD/prefix-base-cpu  $PWD/prefix-step-cpu $PWD/prefix-mut-cpu
OUT=$PWD/x1-cuda PYTHON=python CUDA_VISIBLE_DEVICES=0 \
    $H/run_gate.sh cuda $PWD/prefix-base-cuda $PWD/prefix-step-cuda
```

The single steps:

```bash
PYTHONPATH=PREFIX python golden_x1.py run --device cpu|cuda --out DIR [--case NAME] [--perturb]
python golden_x1.py compare DIR_A DIR_B     # exit 0 = every dump byte-identical, dt identical
```

`--perturb` is the 1-ulp control. Before the first step, it adds one ulp to the vapor row
(`ICY`) of `hydro_u` at the middle cell.

## Expected results

These are from a Quadro RTX 4000 (sm_75), CPU and CUDA, with the builds above.

| compare | CPU | CUDA |
|---|---|---|
| base_a vs base_b | 200 identical, PASS | 200 identical, PASS |
| base vs step (both bases) | 200 identical, PASS | 200 identical, PASS |
| base vs step `--perturb` (1 ulp) | 132/200 differ, FAIL (exit 1) | 57/200 differ, FAIL (exit 1) |
| base vs mutant | 160/200 differ, FAIL (exit 1) | not run |

The FP32 cases report `u0` as float64 on the base and float32 on the step, which is
X1's change. Their dumps are still byte-identical. On other hardware, the control counts
(132, 57, 160) can differ, but they must stay above zero. The base-vs-step result must
stay at 200 identical.
