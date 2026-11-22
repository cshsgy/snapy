# V3 speed audit: v2.14.1 `730b3500` vs v3/wave3 `d3178f20`, and does cuda-graph pay off

Asked by Xi (#no-human-zone, 2026-11-22 17:48Z). Before = snapy v2.14.1 `730b35007d16389c64116e4bf9d2fe93707e29e9` (chengcli main before V3). After = v3/wave3 `d3178f207b60663ff966b7c7ef20025821ef52fa` (#313 head). Same machine, same flags, same kintera and pyharp packages for both. Numbers are wall time per time step (all RK3 stages) after a warm-up, median of 3 independent runs, with the spread of the 3 run medians in parentheses. Lower is better; speed-up = before / after.

## Tables

| case | before v2.14.1 `730b3500` | after v3/wave3 `d3178f20` | speed-up (before/after) |
|---|---|---|---|
| 1. Straka 2-D, CPU 1 rank, FP64 | 53.93 ms (53.18–56.41) | 61.19 ms (59.97–61.67) | 0.88× |
| 1. Straka 2-D, CUDA FP64 eager | 19.22 ms (18.45–20.27) | 24.42 ms (23.32–24.59) | 0.79× |
| 1. Straka 2-D, CUDA FP64 cuda-graph | n/a (no `execution` option) | 12.05 ms (11.79–12.27) | — |
| 1. Straka 2-D, CUDA FP32 eager | 21.33 ms (21.28–21.57) | 25.85 ms (25.23–26.17) | 0.83× |
| 1. Straka 2-D, CUDA FP32 cuda-graph | n/a (no `execution` option) | 12.40 ms (5.94–12.87) | — |
| 1d. shipped straka.yaml (ideal-gas, LMARS), CPU 1 rank | 40.76 ms (39.07–41.00) | 46.41 ms (45.96–46.94) | 0.88× |
| 1d. shipped straka.yaml, CUDA FP64 eager | 14.82 ms (14.45–15.25) | 18.60 ms (18.35–18.83) | 0.80× |
| 1d. shipped straka.yaml, CUDA `execution: cuda-graph` (refused → eager) | n/a (no `execution` option) | 18.78 ms (17.62–19.48) | — |
| 2. moist 3-D cartesian 64³, CPU 1 rank | 2735.28 ms (2625.91–2812.34) | 3134.68 ms (2991.62–4135.52) | 0.87× |
| 2. moist 3-D cartesian 64³, CPU 4 ranks (2×2 blocks) | 1290.31 ms (1083.95–1343.46) | 1572.00 ms (1490.37–1600.91) | 0.82× |
| 2. moist 3-D cartesian 64³, CUDA FP64 eager | 59.92 ms (59.89–66.43) | 72.71 ms (72.18–72.97) | 0.82× |
| 2. moist 3-D cartesian 64³, CUDA FP64 cuda-graph | n/a (no `execution` option) | 53.33 ms (47.26–53.51) | — |
| 3. cubed-sphere 6 panels moist, CPU 6 ranks | 401.28 ms (380.74–439.24) | 455.06 ms (426.44–467.90) | 0.88× |
| 3. cubed-sphere 6 panels moist, CUDA eager (6 blocks, one process) | 137.04 ms (119.47–139.33) | 199.10 ms (176.68–202.19) | 0.69× |
| 3. cubed-sphere 6 panels moist, CUDA `execution: cuda-graph` (refused → eager) | n/a (no `execution` option) | 186.92 ms (174.15–194.40) | — |

| after `d3178f20`, CUDA | eager | cuda-graph | graph speed-up | captures / replays (per run, 105 steps × 3 stages) |
|---|---|---|---|---|
| 1. Straka 2-D FP64 | 24.34 ms (23.32–25.50) | 12.05 ms (11.79–12.27) | 2.02× | 3 / 315 (eager stages 0) |
| 1. Straka 2-D FP32 | 26.17 ms (25.75–27.14) | 12.40 ms (5.94–12.87) | 2.11× | 3 / 315 (eager stages 0) |
| 1d. shipped straka.yaml (refused) | 18.60 ms (18.05–18.76) | 18.78 ms (17.62–19.48) | 0.99× | 0 / 0 (eager stages 315) |
| 2. moist 3-D cartesian 64³ | 72.75 ms (67.24–72.97) | 53.33 ms (47.26–53.51) | 1.36× | 3 / 315 (eager stages 0) |
| 3. cubed-sphere 6 panels (refused: Mesh) | 176.68 ms (172.61–196.37) | 186.92 ms (174.15–194.40) | 0.95× | 0 / 0 (eager stages 0) |

| case 4 (Straka 2-D FP64, after) | ms/step | captures / replays per run |
|---|---|---|
| cuda-graph, fixed dt | 12.05 ms (12.00–12.29) | 3 / 315 |
| cuda-graph, dt changing every step | 102.02 ms (101.18–104.14) | 303 / 315 |
| eager, dt changing every step | 24.02 ms (23.83–24.11) | 0 / 0 |

Reading: in every cell V3 eager is slower than v2.14.1 (CPU 0.82–0.88×; CUDA 0.69–0.83×, the cubed-sphere Mesh of 6 blocks being the worst at 0.69×). cuda-graph pays off on a single cartesian ideal-moist/HLLC block (Straka FP64 2.02× on the shared GPU, 4.4× in the clean window of pass 1; the 64³ moist column 1.36× shared, 1.57× clean) and is unavailable for the shipped ideal-gas/LMARS Straka deck, for any Mesh (multi-block or multi-process, including every cubed-sphere run) and for the cubed-sphere geometry; a dt that changes every step recaptures all three stages (3 captures per step) and is slower than eager.

## Refusals (cuda-graph requested, run eager or rejected)

- `examples/straka.yaml` (case 1d): capability `eager_reasons = ['equation of state is ideal-gas, not ideal-moist', 'riemann solver is lmars, not hllc']` → non-strict run goes eager (0 captures, the "graph" column is an eager run).
- cubed-sphere 6 panels (case 3): on a Mesh the blocks always run eagerly; `execution-strict: true` raises `RuntimeError: integration/execution-strict: a Mesh runs its blocks eagerly (CUDA graphs cover a single MeshBlock only)` (recorded in `runs/c3_strict_refusal.rank0.log`). The per-block capability also lists `'geometry is gnomonic-equiangle, not cartesian'`, `'forcing is not const-gravity alone'` and, with 6 blocks, `'more than one block'`.
- 4 ranks / 6 ranks: Mesh, eager by construction (`'more than one block'`).
- before `730b3500`: `MeshBlockOptions.from_yaml` rejects the key: `RuntimeError: unknown key 'integration/execution'. Valid keys: implicit-scheme, implicit-advection-cfl, shear-cfl, implicit-precision, implicit-fallback-fp64; read by pyharp: type, cfl, tlim, nlim, ncycle_out, verbose.` Its graph cells are n/a. (v2.14.1 ships `tests/straka_cuda_graph.py`, a Python-side experiment for the dry Straka only; not timed here.)
- Capture conditions at `d3178f20` (src/mesh/stage_executor.cpp `capture_reasons`): CUDA state, cartesian, ideal-moist EOS, HLLC, rk3, one block with no process group, const-gravity as the only forcing, no solid mask, no chemistry step. The graph key includes dt, the state store generation and the thermo table epoch.

## Machine and shared-GPU caveat

- 2 × AMD EPYC 7R13 (192 threads), 8 × NVIDIA H100 80GB, Ubuntu 22.04, gcc 11.4.0, nvcc 12.5 (V12.5.40), torch 2.10.0+cu128, Python 3.12, kintera 2.6.11 (`431fd144`, the same package for both trees), pyharp main package.
- CPU cells: load average 40–95 throughout, of which 16 resident `wekanode` processes are a constant ~1600% and Sam's four `pipeline_runner` processes an intermittent 100–200%. Per-run medians (ms): c1_cpu_fp64 before: 53.2, 53.9, 56.4; c1_cpu_fp64 after: 60.0, 61.2, 61.7; c1d_cpu_fp64 before: 39.1, 40.8, 41.0; c1d_cpu_fp64 after: 46.0, 46.4, 46.9; c2_cpu_1rank before: 2625.9, 2735.3, 2812.3; c2_cpu_1rank after: 2991.6, 3134.7, 4135.5; c2_cpu_4rank before: 1083.9, 1290.3, 1343.5; c2_cpu_4rank after: 1490.4, 1572.0, 1600.9; c3_cpu_6rank before: 380.7, 401.3, 439.2; c3_cpu_6rank after: 426.4, 455.1, 467.9.
- CUDA cells: the first pass (GPU 1, 17:53–18:23Z) was overtaken by Sam's four training jobs, which from about 18:10Z hold all eight GPUs (~60 GB each, utilisation 15–100%); those files are kept in `runs_pass1/` and are not used. The tables use the second pass (`rerun_cuda.sh`, GPU 3, 2026-11-22T18:29:08Z → 2026-11-22T18:41:25Z), in which every before/after pair and every eager/graph pair ran back to back, so the ratios are fair but the absolute numbers are inflated by the co-tenant: the GPU's utilisation read just before each cell ranged 1–100%. A clean-GPU pass is queued if a GPU is freed. For reference, the only clean window of pass 1 (after tree, 17:53–18:12Z) gave Straka FP64 eager 22.9 ms vs cuda-graph 5.16 ms (4.4×), FP32 24.5 vs 5.84, and the 64³ moist column 81.1 vs 51.6 ms (1.57×).

## Builds

Both trees: `cmake -DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release -DNETCDF=OFF -DPNETCDF=OFF -DUCX=ON …` with CMAKE_PREFIX_PATH = torch, the kintera 2.6.11 package, the pyharp package, pydisort; `pip install --no-build-isolation --no-deps --target <pkg> .`; the after package is the one used for the wave-3 dry run and the K5 leg (`aam/w1x_snapy_w3k5_d3178f20.sh` pattern), the before package was built by `speed/build_before.sh` with the identical command line. Outputs were disabled in every deck (`outputs: []`, `ncycle_out: 0`).

## Decks and initial states

- **1 / 4 Straka 2-D (`straka_hllc`)**: `tests/run_fp32_straka.py`'s case, built from `examples/straka.yaml` (cartesian 64 × 256 × 1, nghost 3, weno5, reflecting walls, rk3, cfl 0.9) with the single dry species of `tests/test_anomaly_storage.yaml`, EOS ideal-moist (limiter off), riemann HLLC, `anomaly: true`, `perturbation-flux: true`, one block (`nb2: 1`); the cold bubble of `run_fp32_straka.py` as the initial state; `precision` float64 or float32. This is the one Straka configuration whose FP32 path is supported and whose graph is capturable.
- **1d shipped `examples/straka.yaml`**: as shipped (ideal-gas, LMARS) with `nb2: 1`, same cold bubble.
- **2 moist 3-D cartesian (`cart3d_pert_64`)**: the nb/pz gate's `cart3d_pert` deck (dry + vapor ⇌ cloud `h2o_ideal` nucleation, ideal-moist, HLLC, anomaly + perturbation-flux, weno5, x1 reflecting, x2/x3 periodic, const gravity −10, one passive scalar, rk3 cfl 0.4) at 64 × 64 × 64 cells; one block (1 rank, CUDA) or `nb2: 2, nb3: 2` (4 ranks, one block each).
- **3 cubed-sphere 6 panels (`cubed6_moist`)**: `tests/test_flux_positivity_cubedsphere.yaml` (gnomonic-equiangle, r 6.0e6–8.0e6, 16 × 32 × 32 per panel, weno5 vertical with shock, plm horizontal, HLLC, rk3 cfl 0.5) with the vapor/cloud pair of `tests/test_flux_positivity_cubedsphere_moist.py` (`h2o_ideal` nucleation), EOS ideal-moist (limiter off), the bounded scalar removed; 6 blocks in one process (CUDA) or 6 processes of one block (CPU).
- Initial state for 2 and 3: the nb/pz gate's hydrostatic column with a 2% wave (`gate_nbpz.py` `worker`), applied through each block's `coord.x1v/x2v/x3v`.

## Method

`speed/bench.py` (one process = one rank): build `Mesh(MeshOptions.from_yaml)`, move the blocks to the device in FP64, initialise, then `dt0 = mesh.max_time_step` once; warm-up steps (CPU 3, CUDA 5; the first graph step captures), then N timed steps (CPU 30, CUDA 100), each step = `set_cycle` + the three RK3 stages; the clock is `time.perf_counter` around each step with `torch.cuda.synchronize` before each read on CUDA. A single block in a single process steps through `MeshBlock.forward` (the only path that captures graphs); multi-block or multi-process runs through `Mesh.forward` (eager in V3). Multi-process runs use the gate's environment scheme (RANK/WORLD_SIZE/…, gloo on CPU) via `speed/launch.sh`. Threads: `torch.set_num_threads(8)` and OMP_NUM_THREADS=8 per rank (4 for the 6-rank cubed-sphere). Every cell is 3 independent processes; before and after alternate; reported value = median of the 3 run medians, spread = min–max of those medians. dt: fixed = dt0 every step; "changing every step" = dt0 × (0.95 + 0.05·((7919 n) mod 13)/13). Each run writes a JSON with every per-step time, the capability dict and the executor's capture/replay counters (`runs/<cell>.<tree>.r<k>.rank0.json`); `summary.json` and the tables come from `speed/summarize.py`.

Commands: `speed/run_all.sh` (pass 1, CPU cells and the after-tree CUDA cells), `speed/rerun_before.sh` (the before CUDA cells after the key fix, and the strict-refusal record), `speed/rerun_cuda.sh` (pass 2, all CUDA cells interleaved). Harness, decks, scripts and `summary.json` are in `docs/v3/audit/speed/`; the raw per-run JSON files are on the host under `aam/speed/runs` (pass 2) and `aam/speed/runs_pass1`.

Not done: arm64 CPU rows (Jiachen), MPS, multi-GPU, the `--profile` breakdown of where V3's extra eager time goes.
