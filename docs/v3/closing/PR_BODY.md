# V3 closing PR body (draft)

Draft for the closing PR `v3/final` -> `main`. Written 2026-11-24 against `v3/final`
f703419432858ffcbf245d956a026f1b7d16e2ca; the graph stack (graph-ab-d, C4, C, the Mesh-agreement
commit) is not in that head yet, so its rows below are written from the lane posts and are marked
"pending" until the trial integration (`v3-closing/trial-on-f7034194`) and then the merged head
replace them. Every sha is a commit on `v3/final` unless a fork is named.

## What V3 changes

Base: chengcli/snapy `main` deeabfee597cd83f68f4586d245b01f6091eaa6e (v2.14.3). Head: `v3/final`
f7034194 (99 commits on the base before the graph stack). By lane:

- **Register sources (lanes A, R).** `AxialMomentum` reads the flux register; `exact` replaces
  `full` and the closure key (3a6a2c46, A-ph2). `Hydro` holds `X1Momentum`, which builds m1 (11b35d8a,
  R-6); `FluxRegister::post_split` books the step's direct part of S_d (5b68d667). Diffusion posts
  its energy flux and its radial halves, and `exact` allows a viscosity with the viscous torque booked
  (ee0ce297, c33a67f9, A-ph3; test e09ea710). `riemann.central` on the HLLC perturbation-flux path
  (2642c0a5, A-ph4). R-7's outward step transfer (deaaa724) was dropped again in B1 (6f3dbfc9) because
  nothing read it.
- **Output (lane O).** Writer core without `AthenaArray`, netcdf and pnetcdf sinks, vectorize and
  super-resolution removed (f28908d3, O5); restart writer, bundle and reader in one module, the reader
  throws (40d54fb3, O7); `combine_parts` replaces `mppnccombine` (d7cee8bb, 74cbc7b8, O6a/b); the
  slice bounds and order checks join the output table's generated checks (baf68074, O8 sv); the
  budget consumer, `BudgetSeries` and `BudgetCarry` (e8cbe791, O8 budget); `src/interface` deleted
  (73c8ee4f, X8); `print_cycle_diagnostics` deleted (61c4b2e6).
- **Kintera retry and chemistry (lane S-K).** Kintera's failure counter in the step's reset list and
  retry stack (7ce3e938, S-K1); the `ThermoY` active set snapshotted at step start and restored on a
  retry (aa7b41bf, S-K2); the saturation adjustment inside the captured segment (08836980, S-K3); the
  chemistry step as a registered source after the last stage (2555f25d, S-K4); fixed buffers and no
  stage host reads in eos, scalar and sedimentation (ea53dfea, EF-8); explicit species sums in the
  ideal-moist EOS (b316baf4, EF-13, result-changing, see below).
- **Lane C (reconstruction, Riemann, FP32).** FP32 typed constants; uniform weno5 rows read `d`
  (e341ffdf, C9, result-changing for FP32, see below); the block-independent FP32 sweep gauge with the
  x2/x3 walls in the exchanged ghost slabs (4d41d797, C8.gauge); lane C under its 4,672-line ceiling
  (d3178f20, C.trim); one shock default (`false`) and density/pressure floors on the shock path
  (3bd63808, C10); one reader for a reconstruct section (55cb9d0b, C2).
- **Layout and seams (lane S).** `Exchange::seam_slab`, diffusion in one round (b4355db1, S M6); the
  same-process exchange state keyed by the Mesh (295741a0, S-L1); S's dead legacy ranges deleted
  (56569511).
- **Wall prescription and dt (lanes E, D).** The wall prescription works on the wall face planes
  only (8b587ce7, E) and `FluxRegister::reset()` drops it too (c3f2ff8c); dt is a CUDA-graph input,
  not part of the graph key, with a per-block device dt buffer (b84b8bdb, D).
- **Audits.** Audit 1 rows (33 commits, ff239db6..faa65213 range: one home for each duplicated
  helper across layout, coord, eos, riemann, recon, scalar and output; the aneos adiabatic index and
  sound speed in one place, be018f65; the deck readers take their defaults from the option tables).
  Audit 2 rows F1-F8, F10-F12, F14-F16 (0456087d, f8c0c66d, fec6b5e3, 14c7b221, b5212677: unreached
  code, unread fields, commented-out code, one-caller wrappers, a duplicate ctest entry). Option
  defaults read from their option structs (b9632696, 80ac12e9, c07afc5d, 1477afb0, d1eae7bf). F13 and
  F9 (the two unread output options; the six unused `Quantity` rows) are pending on
  happysky19/snapy `v3-closing/f13-f9-f7034194` (c57138ae, f173a0c2) and join this list when they
  land.
- **Build and CI.** kintera 2.6.11 from PyPI instead of a source build (b541a6db), then 2.6.12 on
  every leg (1497416a, 7d48cfa4), then 2.6.13 on every leg with the seam energy reference re-recorded
  (f7034194; see "kintera pin"). The parallel tests keep `snap_cu` linked (750a5b09). `SNAP_SQR` is a
  macro again so contracting compilers fuse as before ff66651a (7a70d38f) and
  `ANEOSThermoImpl::create` keeps the pointer it made (de04b2a5): the two macOS fixes, CI green on all
  four jobs at f7034194 (UCzhangxi/snapy run 45307532174).
- **CUDA graphs (pending: graph-ab-d 9b5c7a49, C4 a26cb4c4, C-on-C4, the Mesh-agreement commit).**
  Captured stage segments with per-block device dt (A, B, D); a cut at an exchange so a Mesh's blocks
  capture together and the cubed sphere captures (B3); the vertical implicit solve reads the device
  dt (C4); the implicit refusal lifted with an FP64 retry, the chemistry slot as its own eager op, Mesh
  chemistry captured (C1, C2, C5); a Mesh whose blocks disagree runs eager (pending). Gate on every
  row: graph = eager bitwise in FP64 and FP32 with dt changing every step. Not captured yet: more than
  one process, pz > 1 (P2, P3), and the FP32 cubed sphere cannot be built (G5).

## Result-changing rows and their bounds

Everything not listed here is byte-identical to the base on the dry-run gates (#349 dump set FP64 and
FP32, #348 face decks, nb/pz, x1_golden) on CPU and CUDA. Bounds are relative to the field (row)
maximum unless stated. Evidence: `aam/w3fin/report.md` (tip d3178f20 vs main deeabfee, this node and
Jiheng's sm_61) and the lane posts in thread 1794982799.250779.

| row | what moves | bound (measured) | declared |
|---|---|---|---|
| EF-13 b316baf4, explicit species sums | the moist decks, FP64: #349 7 files (4 moist sedimentation cases, hydro_u and scalar_s); x1_golden jupiter_crm_fp64; nb/pz cart3d and cart3d_pert (44/49 dumps) | #349 IVX 1.2e-13 CPU / 1.5e-13 CUDA, other fields <= 2e-15; x1_golden 1.2e-12 (one ulp of the state, equal to the 1-ulp u0 mutant); nb/pz <= 4.3e-15 of the dump max | "round-off change in the moist decks, FP64 <= 1.2e-13" |
| C9 e341ffdf, FP32 typed constants | every FP32 run on a weno/LMARS deck: #349 FP32 21/23 files; x1_golden cart_moist_fp32, jupiter_crm_fp32 | FP32-vs-own-FP64 worst unchanged (1.703e-3 vs 1.707e-3, gnomonic_weno5s IVX; per-field ratio 0.71-1.46); x1_golden at the mutant's level (2.4e-4, 1.6e-5) | "FP32 re-baselined against FP64; FP64 bitwise" |
| C8.gauge 4d41d797 | CUDA jupiter_crm_fp32 | 1 ulp | "on CUDA jupiter_crm_fp32 moves by 1 ulp" |
| G4 faa65213, boundary velocity reads the stored sine | declared round-off; the audit-1 gates (#349 FP64/FP32, #348 face, nb/pz) were byte-identical on CPU and CUDA sm_75 at c42d4fff | 0 on the gates | "round-off" |
| kintera 2.6.13 (f7034194): Rgas 8.31446 -> 8.31446261815324 (3.1e-7 relative) | every moist result; the seam test's energy flux sum | moist dump fields <= 1.1e-9 (nb/pz 44/49 dumps differ, the 5 dry sph_face dumps identical; measured on d3178f20); seam energy sum 7.6e-7, reference re-recorded 342168.95387051086 -> 342168.69510693091 (momentum sum unchanged to 1.5e-11) | pinned exactly so every leg compares like for like |
| macOS fix 7a70d38f (SNAP_SQR macro) | arm64/AppleClang only: lmars, hllc, roe, WENO and the implicit decomposition fuse multiply-adds as before ff66651a | Linux: #349 byte-identical, ctest by name identical; macOS: test_fp32_typed_constants 4/5 differing -> 0/0 | restores the pre-ff66651a numbers on contracting compilers |
| A-ph4 2642c0a5 | one new compiler warning (tests/test_riemann.cpp:585, mkstemp) | test-only | noted on #309 |
| CUDA graphs (pending rows) | none in eager mode; graph replay equals eager bitwise | 0 (32/32 hashes per case, FP64 and FP32, dt changing) | bitwise |

Environment-only failures, identical on base and head on every machine used: the Eigen entries the
ctest regex catches (Not Run), the restart/NetCDF tests on a NETCDF=OFF build, test_straka,
test_exchange_ucx_cuda (two GPUs), test_python_import_path (sandbox path), the budget tests without
NetCDF, the six nb/pz sph cell runs (refused since I-7).

## kintera pin

- `v3/final` ships pinned to kintera **2.6.13**: CI installs `kintera==2.6.13` on every leg
  (`.github/workflows/ci.yml`), `pyproject.toml` requires `kintera>=2.6.13`, `CMakeLists.txt`
  `find_package(Kintera 2.6.13 REQUIRED)`, release.yml pins `kintera>=2.6.13`. 2.6.13 is on PyPI
  (chengcli/kintera#159, v2.6.13): exact SI Rgas, `nfail_` dropped (K5), third-body and falloff
  layout fixes.
- Why the exact pin: 2.6.12 prepares the moist column's thermo state on MPS (kintera #158), 2.6.11
  asserts in PyTorch's dispatcher at the S-K2 prepare(); 2.6.13's Rgas moves the moist numbers (see
  the table), so every leg must compare like for like.
- The baseline (main deeabfee) does not build against 2.6.13: it reads `pthermo->nfail_`
  (src/mesh/meshblock_redo.cpp:124-126), dropped in 2.6.13. The baseline is not patched. Per
  `docs/v3/closing/dry_run_plan.md` (round 3): the correctness and speed legs build **both sides on
  kintera 2.6.12**, and one V3-only leg, V3@2.6.12 vs V3@2.6.13, measures the kintera effect by
  itself. Both comparisons are reported below.

## Follow-ups (not in this PR)

- O3 (deferred by the maintainer): later, as its own PR.
- #369 (terrain, the per-column immersed solid on the x1-wall path): after V3. The column case of
  issue thread 1794709389.746279 (a Cartesian rest column over a solid, uniform vs x1rat 1.05) is still
  present at b84b8bdb: earth fill redoes at step 1, matched fill leaves a steady 2-cell w (4.0 m/s
  uniform, 1.8 m/s stretched); the wall twin sits at round-off.
- CUDA graphs: more than one process and pz > 1 stay refused (P2, P3); the FP32 cubed sphere cannot
  be built (G5); a Mesh whose blocks disagree on capture must run eager (the Mesh-agreement commit,
  pending on the C chain); the implicit solve recaptures on every dt until C sits on C4.
- Known, not changed: the single-block gnomonic decks' sensitivity to the boundary.prescribed item
  (declared in wave 2); test_python_import_path depends on the sandbox's Python path.

## Dry-run table (placeholder; filled from the rehearsal on the trial head, then the merged head)

Both sides on kintera 2.6.12 unless the row says otherwise; base = main deeabfee built from a clean
worktree with the plan's flags; "identical" means byte-identical per file.

| leg | device (owner) | base_a vs base_b | base vs V3 | differing cases and worst bound | attributed to |
|---|---|---|---|---|---|
| #349 dump set FP64 (23 cases) | CPU + CUDA sm_90 (Sihe); sm_120 (Zoey); sm_61 (Jiheng); sm_75 (Xinyue) | | | | |
| #349 dump set FP32 (23 cases) | same | | | | |
| #348 face decks (8) | CPU + CUDA sm_90; sm_120; sm_61 | | | | |
| nb/pz (49 dumps) | CPU (Sihe, Xinyue) | | | | |
| x1_golden (5 cases, 200 dumps) | CPU + CUDA sm_90; sm_120; sm_75 | | | | |
| serial ctest by name (gates.sh subset) | CPU + CUDA sm_90; sm_120; sm_61; sm_75 | | | verdict changes by name | |
| moist-mixture decks (7) | CUDA sm_61 (Jiheng) | | | | |
| V3@2.6.12 vs V3@2.6.13 (kintera effect alone) | CPU + CUDA sm_90 (Sihe) | n/a | | | kintera 2.6.13 Rgas |

## Speed table (placeholder; `aam/speed`, before = main 730b3500, after = V3, both on kintera 2.6.12)

One GPU (H100) and the CPU, before and after interleaved, 3 reps per cell; medians of ms/step.

| deck | device | before ms/step (730b3500 eager) | after ms/step (V3 eager) | after ms/step (V3 cuda-graph) | after/before | graph speed-up |
|---|---|---|---|---|---|---|
| straka_hllc | CPU | | | n/a | | n/a |
| straka_hllc | H100 | | | | | |
| examples/straka.yaml | CPU | | | n/a | | n/a |
| examples/straka.yaml | H100 | | | | | |
| cart3d_pert_64 (1 rank) | CPU / H100 | | | | | |
| cart3d_pert_64 (4 ranks) | CPU | | | n/a | | n/a |
| cubed6_moist (6 ranks CPU; 1 process 6 blocks CUDA) | CPU / H100 | | | | | |
| cart3d_moist_64 | CPU / H100 | | | | | |
