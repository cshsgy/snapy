#!/usr/bin/env python3
"""Compose docs/v3/audit/speed.md from aam/speed/summary.json and the run logs."""
import json, os, re, sys, subprocess
S = os.path.dirname(os.path.abspath(__file__))
s = json.load(open(os.path.join(S, "summary.json")))
log = open(os.path.join(S, "rerun_cuda.log")).read()
gpu = re.search(r"gpu=(\d+)", log).group(1)
utils = [int(x) for x in re.findall(r"gpu_util_before=(\d+),", log)]
t0 = re.search(r"RERUN_CUDA_START (\S+)", log).group(1); t1 = re.search(r"RERUN_CUDA_DONE (\S+)", log)
t1 = t1.group(1) if t1 else "running"
def c(name, tree):
    return s.get(name, {}).get(tree)
def f(cell):
    if not cell: return "n/a"
    return f"{cell['median_ms']:.2f} ms ({cell['spread_ms'][0]:.2f}–{cell['spread_ms'][1]:.2f})"
def ratio(a, b):
    return f"{a['median_ms']/b['median_ms']:.2f}×" if (a and b) else "—"
rows = [
 ("1. Straka 2-D, CPU 1 rank, FP64", "c1_cpu_fp64"),
 ("1. Straka 2-D, CUDA FP64 eager", "c1_cuda_fp64_eager"),
 ("1. Straka 2-D, CUDA FP64 cuda-graph", "c1_cuda_fp64_graph"),
 ("1. Straka 2-D, CUDA FP32 eager", "c1_cuda_fp32_eager"),
 ("1. Straka 2-D, CUDA FP32 cuda-graph", "c1_cuda_fp32_graph"),
 ("1d. shipped straka.yaml (ideal-gas, LMARS), CPU 1 rank", "c1d_cpu_fp64"),
 ("1d. shipped straka.yaml, CUDA FP64 eager", "c1d_cuda_fp64_eager"),
 ("1d. shipped straka.yaml, CUDA `execution: cuda-graph` (refused → eager)", "c1d_cuda_fp64_graph"),
 ("2. moist 3-D cartesian 64³, CPU 1 rank", "c2_cpu_1rank"),
 ("2. moist 3-D cartesian 64³, CPU 4 ranks (2×2 blocks)", "c2_cpu_4rank"),
 ("2. moist 3-D cartesian 64³, CUDA FP64 eager", "c2_cuda_eager"),
 ("2. moist 3-D cartesian 64³, CUDA FP64 cuda-graph", "c2_cuda_graph"),
 ("3. cubed-sphere 6 panels moist, CPU 6 ranks", "c3_cpu_6rank"),
 ("3. cubed-sphere 6 panels moist, CUDA eager (6 blocks, one process)", "c3_cuda_eager"),
 ("3. cubed-sphere 6 panels moist, CUDA `execution: cuda-graph` (refused → eager)", "c3_cuda_graph"),
]
T1 = ["| case | before v2.14.1 `730b3500` | after v3/wave3 `d3178f20` | speed-up (before/after) |", "|---|---|---|---|"]
for label, key in rows:
    b, a = c(key, "before"), c(key, "after")
    if key.endswith("_graph"): b = None
    bcol = f(b) if b else ("n/a (no `execution` option)" if key.endswith("_graph") else "n/a")
    T1.append(f"| {label} | {bcol} | {f(a)} | {ratio(b, a)} |")
T2 = ["| after `d3178f20`, CUDA | eager | cuda-graph | graph speed-up | captures / replays (per run, 105 steps × 3 stages) |", "|---|---|---|---|---|"]
for label, key in [("1. Straka 2-D FP64", "c1_cuda_fp64"), ("1. Straka 2-D FP32", "c1_cuda_fp32"), ("1d. shipped straka.yaml (refused)", "c1d_cuda_fp64"), ("2. moist 3-D cartesian 64³", "c2_cuda"), ("3. cubed-sphere 6 panels (refused: Mesh)", "c3_cuda")]:
    e, g = c(key + "_eagerB", "after") or c(key + "_eager", "after"), c(key + "_graph", "after")
    rep = g["execution_report"] if g else {}
    T2.append(f"| {label} | {f(e)} | {f(g)} | {ratio(e, g)} | {rep.get('captures','?')} / {rep.get('replays','?')} (eager stages {rep.get('eager','?')}) |")
T3 = ["| case 4 (Straka 2-D FP64, after) | ms/step | captures / replays per run |", "|---|---|---|"]
for label, key in [("cuda-graph, fixed dt", "c4_graph_fixed"), ("cuda-graph, dt changing every step", "c4_graph_varying"), ("eager, dt changing every step", "c4_eager_varying")]:
    g = c(key, "after"); rep = g["execution_report"] if g else {}
    T3.append(f"| {label} | {f(g)} | {rep.get('captures','?')} / {rep.get('replays','?')} |")
cpu_runs = []
for key in ("c1_cpu_fp64", "c1d_cpu_fp64", "c2_cpu_1rank", "c2_cpu_4rank", "c3_cpu_6rank"):
    for tree in ("before", "after"):
        x = c(key, tree); cpu_runs.append(f"{key} {tree}: " + ", ".join(f"{v:.1f}" for v in x["per_run_median_ms"]))
doc = f"""# V3 speed audit: v2.14.1 `730b3500` vs v3/wave3 `d3178f20`, and does cuda-graph pay off

Asked by Xi (#no-human-zone, 2026-11-22 17:48Z). Before = snapy v2.14.1 `730b35007d16389c64116e4bf9d2fe93707e29e9` (chengcli main before V3). After = v3/wave3 `d3178f207b60663ff966b7c7ef20025821ef52fa` (#313 head). Same machine, same flags, same kintera and pyharp packages for both. Numbers are wall time per time step (all RK3 stages) after a warm-up, median of 3 independent runs, with the spread of the 3 run medians in parentheses. Lower is better; speed-up = before / after.

## Tables

{chr(10).join(T1)}

{chr(10).join(T2)}

{chr(10).join(T3)}

Reading: in every cell V3 eager is slower than v2.14.1 (CPU 0.82–0.88×; CUDA 0.69–0.83×, the cubed-sphere Mesh of 6 blocks being the worst at 0.69×). cuda-graph pays off on a single cartesian ideal-moist/HLLC block (Straka FP64 {ratio(c("c1_cuda_fp64_eagerB","after"), c("c1_cuda_fp64_graph","after"))} on the shared GPU, 4.4× in the clean window of pass 1; the 64³ moist column {ratio(c("c2_cuda_eagerB","after"), c("c2_cuda_graph","after"))} shared, 1.57× clean) and is unavailable for the shipped ideal-gas/LMARS Straka deck, for any Mesh (multi-block or multi-process, including every cubed-sphere run) and for the cubed-sphere geometry; a dt that changes every step recaptures all three stages (3 captures per step) and is slower than eager.

## Refusals (cuda-graph requested, run eager or rejected)

- `examples/straka.yaml` (case 1d): capability `eager_reasons = ['equation of state is ideal-gas, not ideal-moist', 'riemann solver is lmars, not hllc']` → non-strict run goes eager (0 captures, the "graph" column is an eager run).
- cubed-sphere 6 panels (case 3): on a Mesh the blocks always run eagerly; `execution-strict: true` raises `RuntimeError: integration/execution-strict: a Mesh runs its blocks eagerly (CUDA graphs cover a single MeshBlock only)` (recorded in `runs/c3_strict_refusal.rank0.log`). The per-block capability also lists `'geometry is gnomonic-equiangle, not cartesian'`, `'forcing is not const-gravity alone'` and, with 6 blocks, `'more than one block'`.
- 4 ranks / 6 ranks: Mesh, eager by construction (`'more than one block'`).
- before `730b3500`: `MeshBlockOptions.from_yaml` rejects the key: `RuntimeError: unknown key 'integration/execution'. Valid keys: implicit-scheme, implicit-advection-cfl, shear-cfl, implicit-precision, implicit-fallback-fp64; read by pyharp: type, cfl, tlim, nlim, ncycle_out, verbose.` Its graph cells are n/a. (v2.14.1 ships `tests/straka_cuda_graph.py`, a Python-side experiment for the dry Straka only; not timed here.)
- Capture conditions at `d3178f20` (src/mesh/stage_executor.cpp `capture_reasons`): CUDA state, cartesian, ideal-moist EOS, HLLC, rk3, one block with no process group, const-gravity as the only forcing, no solid mask, no chemistry step. The graph key includes dt, the state store generation and the thermo table epoch.

## Machine and shared-GPU caveat

- 2 × AMD EPYC 7R13 (192 threads), 8 × NVIDIA H100 80GB, Ubuntu 22.04, gcc 11.4.0, nvcc 12.5 (V12.5.40), torch 2.10.0+cu128, Python 3.12, kintera 2.6.11 (`431fd144`, the same package for both trees), pyharp main package.
- CPU cells: load average 40–95 throughout, of which 16 resident `wekanode` processes are a constant ~1600% and Sam's four `pipeline_runner` processes an intermittent 100–200%. Per-run medians (ms): {"; ".join(cpu_runs)}.
- CUDA cells: the first pass (GPU 1, 17:53–18:23Z) was overtaken by Sam's four training jobs, which from about 18:10Z hold all eight GPUs (~60 GB each, utilisation 15–100%); those files are kept in `runs_pass1/` and are not used. The tables use the second pass (`rerun_cuda.sh`, GPU {gpu}, {t0} → {t1}), in which every before/after pair and every eager/graph pair ran back to back, so the ratios are fair but the absolute numbers are inflated by the co-tenant: the GPU's utilisation read just before each cell ranged {min(utils)}–{max(utils)}%. A clean-GPU pass is queued if a GPU is freed. For reference, the only clean window of pass 1 (after tree, 17:53–18:12Z) gave Straka FP64 eager 22.9 ms vs cuda-graph 5.16 ms (4.4×), FP32 24.5 vs 5.84, and the 64³ moist column 81.1 vs 51.6 ms (1.57×).

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
"""
open(os.path.join(S, "speed.md"), "w").write(doc)
print("doc written", len(doc.splitlines()), "lines")
