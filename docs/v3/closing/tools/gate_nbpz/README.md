# nb/pz bitwise gate (V3 wave 1, lane S step (3))

This is a gate tool for V3 wave 1, lane S step (3): `Exchange` takes the x1 relays out of
hydro (phase 1, transport only; S.rst rev 2.4 `76204d46`, :1471-1476 and :1949-1954;
DESIGN.rst Table 4 at `449636b3`). It is **not part of the wave PR**. A step passes when
every dump of every run is byte-identical to the base build's, in FP64, on the same device.

## What it runs

There are 5 decks, each with its nb/pz configurations, for 19 runs in all. Every run steps
the deck and writes each local block's final `hydro_u` (and `scalar_s`, if the deck has
scalars) as raw bytes, ghost cells included. That gives 59 dumps.

| deck | geometry and physics | schedule rows it reaches (S.rst section 3.3) | steps |
|---|---|---|---|
| `cart3d` | Cartesian 24 x 8 x 4, ideal-moist (vapor, cloud), passive scalar, weno5, LMARS, grav1 -10, flux covariance on (default) | 6-9, 13-16, 19, 22-25 | 10 |
| `cart3d_pert` | `cart3d` plus anomaly, perturbation-flux and HLLC | row 8 with its four fields (pref, dref, cell0, cell_delta) | 10 |
| `sph_co4` | spherical-polar column, 32 cells, conversion-order 4, plm | 3-4, 6-8 | 20 |
| `sph_centroid` | the same column with weno5, run with `SNAP_X1_CENTROID_EXACT=1` | 5, 10 | 20 |
| `sph_face` | the same column with `gravity-work: face` | 19 | 20 |

| config | split | blocks per process | processes | path exercised |
|---|---|---:|---:|---|
| `pz1_nb1` | none | 1 | 1 | no seam (the reference shape) |
| `pz1_nb2` | x2 (Cartesian decks only) | 2 | 1 | no x1 seam, two blocks |
| `pz2_nb2` | x1 | 2 | 1 | same-process boards |
| `pz2_nb1_mp` | x1 | 1 | 2 | process group; row 9 runs only here and below |
| `pz2x2_nb2_mp` | x1 and x2 (Cartesian decks only) | 2 | 2 | boards and process group together |

`nb` means blocks per process and `pz` means x1 partitions. `torch.set_num_threads(1)` is
used throughout (S.rst's "scalar CPU setting"). `config()`, `cart3d()` and `sph()` in
`gate_nbpz.py` define the decks. `decks/` holds the same YAML, written by `--write-decks`.

The script launches the two-process runs itself. It sets RANK, WORLD_SIZE, MASTER_PORT,
BLOCKS_PER_PROCESS, DEVICE and DEVICE_ID for each rank. The backend is gloo on CPU and ucx
(commux) on CUDA, because gloo carries CPU tensors only. `GATE_BACKEND` overrides it. On
CUDA, rank r uses GPU `r % device_count`.

## Running it

You need a snapy build of the base and of the step, each with its own Python package. You
also need kintera >= 2.6.9 and torch 2.10 on the path. Build both with identical flags.

```sh
# base: chengcli/snapy main 730b3500 (or the current v3/wave1 head)
PYTHONPATH=<base-pkg> python gate_nbpz.py --device cpu  --out runs/base_cpu
PYTHONPATH=<base-pkg> python gate_nbpz.py --device cuda --out runs/base_cuda
# step
PYTHONPATH=<step-pkg> python gate_nbpz.py --device cpu  --out runs/step_cpu
PYTHONPATH=<step-pkg> python gate_nbpz.py --device cuda --out runs/step_cuda
# gate: exit 0 only when all 59 dumps are present and identical and no run failed
python gate_nbpz.py --compare runs/base_cpu  runs/step_cpu
python gate_nbpz.py --compare runs/base_cuda runs/step_cuda
# negative control: a 1-ulp change in one cell must reach every cart3d run's dumps
PYTHONPATH=<step-pkg> python gate_nbpz.py --negative-control runs/step_cpu --device cpu
```

`--compare` refuses to compare a CPU run with a CUDA run. It checks the sha256 values in
both manifests and then the bytes of every dump. Each run directory gets the `*.bin` dumps,
a `*.shape` file per dump, one log per run, and `manifest.json` holding the sha256 and size
of each dump, the torch and snapy versions, and the wall time. A run directory is about
4 MB (3.5 MB of dumps).

If you have no local base build, you can check a run against the stored checksums instead.
They only match on builds and hardware that reproduce main's exact bits:

```sh
python gate_nbpz.py --check-manifest runs/step_cpu results/main-730b3500-cpu.json
```

## Code-level negative control

`negative_control/step3_unpack_swap.patch` applies to the step commit (`ac0af862`, branch
`v3/wave1-S`). It reverses the field order of `Exchange::x1_rows`'s multi-field unpack, which
is a transport bug. Build with the patch, run, and compare against the base. The gate must
fail. Measured on CPU: exit 1, 44/59 dumps differ. The 15 identical dumps all come from the
`pz1` runs, which have no x1 seam.

## Main against main, and main against the step (CPU and CUDA kept separate)

Builds:
- chengcli/snapy main `730b35007d16389c64116e4bf9d2fe93707e29e9` and step `ac0af86266812889322dc449fd6a4eeabbc6f40f`
- Release, `-DCUDA=ON -DNETCDF=ON -DBUILD_TESTS=ON`
- torch 2.10.0+cu126; kintera v2.6.9 (`b8e726ce`) from source; gcc 13.3; nvcc 12.9
- GPU: GTX 1080 Ti (sm_61)

The same build was used for CPU and CUDA.

| comparison | result | exit |
|---|---|---:|
| CPU: main run a vs main run b | 59/59 identical | 0 |
| CPU: main vs step | 59/59 identical | 0 |
| CPU: 1-ulp negative control | 22/22 cart3d dumps differ, 5/5 runs | 0 |
| CPU: main vs step with the unpack-swap patch | 15/59 identical, 44 differ | 1 |
| CUDA: main run a vs main run b | 59/59 identical | 0 |
| CUDA: main vs step | 59/59 identical | 0 |
| CUDA: 1-ulp negative control | 22/22 cart3d dumps differ, 5/5 runs | 0 |
| CUDA vs CPU (information only; `--compare` refuses it) | 0/59 identical by `cmp` | - |

`results/main-730b3500-cpu.json` and `results/main-730b3500-cuda.json` are the base manifests
(sha256 per dump). One whole run (19 runs) takes about 36 s on CPU and 47-49 s on CUDA.
