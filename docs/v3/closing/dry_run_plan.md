# V3 closing PR: dry-run and speed-table plan

Written 2026-11-23 so the runs can start the moment the graph stack merges into `v3/final`. Every leg
compares a V3 build against a baseline build made with the same flags, on the same machine and device.

## Shas and baselines

| role | sha | used for |
|---|---|---|
| correctness baseline | chengcli/snapy main `deeabfee597cd83f68f4586d245b01f6091eaa6e` (v2.14.3, the pre-V3 main of the wave and tip dry runs) | #349, #348 face, nb/pz, x1_golden, serial ctest by name |
| speed baseline | chengcli/snapy main `730b35007d16389c64116e4bf9d2fe93707e29e9` (v2.14.1, the pre-V3 head; no `execution` option, so eager only) | the speed table, as in `docs/v3/audit/speed.md` |
| V3 side | the final `v3/final` head after the graph stack merges (`git ls-remote https://github.com/UCzhangxi/snapy.git v3/final`, pinned in every post) | every leg |

Both sides are built from a clean worktree at the pinned sha with the same CMake flags, the same
kintera and pyharp packages on `PYTHONPATH`, and the same torch. The kintera version is the one the
V3 CI pins at that head (`2.6.13` once the move commit lands; `2.6.12` before it). The baseline is
built against the same kintera package, so the comparison isolates snapy.

## Legs, decks and who runs them

| leg | decks | compare | CUDA sm_90 + x86 (this node, Sihe) | sm_120 (Zoey) | sm_61 (Jiheng) | arm64, NETCDF=ON (Jiachen) | macOS MPS (fork CI) |
|---|---|---|---|---|---|---|---|
| #349 dump set | 23 cases of `aam/dumpset349/tools/dumpset349/dumpset349.py` (20 rk3 steps, `hydro_u` + `scalar_s`), FP64 and FP32 | byte-identical per file; `compare.py` lists differing cases | CPU + CUDA | CUDA | CUDA | CPU | — |
| #348 face decks | 8 decks of `aam/w2so/h348_face.py` (the face-gravity subset of harness #348; the cell decks are refused since I-7) | identical 8/8, worst energy-row rel diff | CPU + CUDA | CUDA | CUDA | CPU | — |
| nb/pz | 49 dumps of `aam/nbpz/tools/gate_nbpz/gate_nbpz.py` (cart3d, cart3d_pert, sph_face × 5 layouts; sph_co4/sph_centroid fail on both trees) | byte-identical | CPU | — | — | CPU | — |
| x1_golden | 5 cases / 200 dumps of `aam/x1g/tools/v3/x1_golden` (`run_gate.sh`: base_a, base_b, step, step_perturb) | identical; base_a vs base_b must be 200/200 | CPU + CUDA | CUDA | — | — | — |
| serial ctest by name | the `gates.sh` subset regex (below), `ctest -j 1` | every common entry keeps its verdict; new entries listed | x86 CPU (no GPU) + CUDA | CUDA | CUDA | CPU | CI runs the full suite |
| moist-mixture decks | Jiheng's 7-deck set | identical | — | — | CUDA | — | — |
| speed table | `aam/speed`: straka_hllc, `examples/straka.yaml`, cart3d_pert_64 (1 rank, 4 ranks), cubed6_moist (6 ranks CPU; 1 process 6 blocks CUDA) eager and cuda-graph on the same GPU, cart3d_moist_64 | ms/step medians, before/after ratio, graph speed-up | CPU + one H100, interleaved before/after, 3 reps | — | — | — | — |

Legs marked "—" are not run on that machine. Each owner posts one message per leg with both full shas,
the device, the kintera version, the counts and the worst difference.

## Commands (this node; the other machines substitute their paths)

Build one tree (`NAME`, `SHA`), package it, and verify the import (`aam/final/build_head.sh`):

    bash aam/final/build_head.sh NAME SHA        # cmake -DCUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
                                                 # -DNETCDF=OFF -DPNETCDF=OFF -DUCX=ON, kintera pkg + pyharp pkg on PYTHONPATH,
                                                 # pip install -t aam/w1x/pkg_snapy_NAME, then `import snapy,pyharp,kintera`

Correctness legs, base vs step (same device compared only with itself; a base_a vs base_b control first):

    PYTHONPATH=$PKG:$HK python aam/dumpset349/tools/dumpset349/dumpset349.py --device {cpu,cuda} --dtype {float64,float32} --out DIR
    python aam/dumpset349/tools/dumpset349/compare.py DIR_base DIR_step
    PYTHONPATH=$PKG:$HK python aam/w2so/h348_face.py --device {cpu,cuda} --out DIR ; python aam/w2so/h348_face.py --compare DIR_base DIR_step
    CUDA_VISIBLE_DEVICES= PYTHONPATH=$PKG:$HK python aam/nbpz/tools/gate_nbpz/gate_nbpz.py --device cpu --out DIR ; ... --compare DIR_base DIR_step
    OUT=DIR PYTHON=python bash aam/x1g/tools/v3/x1_golden/run_gate.sh {cpu,cuda} PREFIX_base PREFIX_step
    cd build-x1 && PYTHONPATH=$PKG:$HK ctest -j 1 --timeout 900 -R "$RE" --output-on-failure   # then verdicts by name, sorted, diffed

with `RE` = the `gates.sh` subset: axial|aam|closure|global_phase|coriolis|register|diffusion|radial|cubed|sphere|seam|exchange|implicit|vic|straka|shallow|redo|mesh|conservation|budget|options|gnomonic|momentum|hydro|split|combine|restart|output|coordinate|eos|riemann|reconstruct|scalar|fp32|sedimentation|forcing|capture|stage_executor|positivity|user|jit|netcdf

Speed table (`aam/speed/run_all.sh`, one GPU, before and after interleaved, 3 reps per cell; CPU cells
30 steps after 3 warm-up, CUDA cells 100 steps after 5; `bench.py` times each step with
`time.perf_counter` after `torch.cuda.synchronize`):

    CUDA_VISIBLE_DEVICES=G bash aam/speed/run_all.sh ; python aam/speed/summarize.py ; python aam/speed/gen_doc.py

`launch.sh` maps `before` to the 730b3500 package and `after` to the V3 package; the cubed-sphere
6-block deck runs eager and `execution: cuda-graph` on the same GPU (a Mesh may refuse the capture:
the table then records the refusal reason, as speed.md does today).

## Rules

1. Pin both shas with `ls-remote` before building and again before posting; a moved head means a rerun.
2. A control run (base vs a second base build or run) must be identical before any base-vs-step difference counts.
3. Every differing file is attributed to a named V3 change in the closing PR body, with its bound (as in `aam/w3fin/report.md`); nothing unexplained lands.
4. Report GPU model and utilisation for the speed cells; a shared GPU is stated as such.
5. Not run anywhere: #348 cell decks and the 6 nb/pz sph cell runs (not runnable on either tree), multi-node.
