# V3 closing PR body (draft, r3)

Draft for the closing PR `v3/final` -> `main`. r3 written 2026-11-25 against the batch that lands:
`v3/final` 4c023985 (169 commits on the base) plus the fixes listed under "Fixes landing with this
PR", which sit on forks until they are placed on `v3/final`; their shas are replaced by the landed
ones at that point. Every sha is a commit on `v3/final` unless a fork is named. Numbers live in the
cited documents and thread posts, not in this body, except the declared bounds of the
result-changing rows.

## What V3 changes

Base: chengcli/snapy `main` deeabfee597cd83f68f4586d245b01f6091eaa6e (v2.14.3). Head: `v3/final`
4c023985af5f80cd6b20ce9d5f9b6acfc7873d6a.

- **The flux register books every source.** `AxialMomentum` reads the flux register; `exact`
  replaces `full` and the closure key (3a6a2c46). `Hydro` holds `X1Momentum`, which builds the
  axial momentum (11b35d8a); `FluxRegister::post_split` books the step's direct part of the split
  source (5b68d667). Diffusion posts its energy flux and its radial halves, and `exact` allows a
  viscosity with the viscous torque booked (ee0ce297, c33a67f9; test e09ea710). `riemann.central`
  sits on the HLLC perturbation-flux path (2642c0a5). The outward step transfer (deaaa724) was
  dropped again (6f3dbfc9) because nothing read it. The stage's census masks, wall rates and status
  are batched once per stage and the stage amounts fold once per step on the device (3bf18620,
  ce297125, 871b90c0); the wall batch keeps fixed device storage and its slots follow the stage's
  push order (a78dc50f, fa734fff, 9766b2d7); the register re-prescribes after a source's post and
  rollback goes through the monitor (8a224e1b, 614c2772); only writers that can touch the
  prescribed faces mark a direction (89a38287, d7c7a93a); `place_amounts` checks all three amount
  tensors (939f7928) and casts on the host before a move, so MPS, which holds no float64, works
  (9003c961).
- **Output.** Writer core without `AthenaArray`; netcdf and pnetcdf sinks; vectorize and
  super-resolution removed (f28908d3); restart writer, bundle and reader in one module, the reader
  throws (40d54fb3); `combine_parts` replaces `mppnccombine` (d7cee8bb, 74cbc7b8); the slice bounds
  and order checks join the output table's generated checks (baf68074); the budget consumer,
  `BudgetSeries` and `BudgetCarry` (e8cbe791); `src/interface` deleted (73c8ee4f);
  `print_cycle_diagnostics` deleted (61c4b2e6).
- **Kintera retry and chemistry.** Kintera's failure counter is in the step's reset list and retry
  stack (7ce3e938); the `ThermoY` active set is snapshotted at step start and restored on a retry
  (aa7b41bf); the saturation adjustment runs inside the captured segment (08836980); the chemistry
  step is a registered source after the last stage (2555f25d); eos, scalar and sedimentation use
  fixed buffers and no stage host reads (ea53dfea); the ideal-moist EOS sums species explicitly
  (b316baf4, result-changing, see below).
- **Reconstruction, Riemann and FP32.** FP32 typed constants; uniform weno5 rows read `d`
  (e341ffdf, result-changing for FP32, see below); the block-independent FP32 sweep gauge with the
  x2/x3 walls in the exchanged ghost slabs (4d41d797); the reconstruction code under its line
  ceiling (d3178f20); one shock default (`false`) and density/pressure floors on the shock path
  (3bd63808); one reader for a reconstruct section (55cb9d0b); the stretched x1 reference rows live
  on the grid's device (069769ba, issue #316).
- **Layout and seams.** `Exchange::seam_slab`, diffusion in one round (b4355db1); the same-process
  exchange state keyed by the Mesh (295741a0); dead legacy ranges deleted (56569511); the
  panel-seam rotation operands are built once per grid and the cache holds its source grid storages
  (1a3c45e7, 70f6e15e).
- **Wall prescription and dt.** The wall prescription works on the wall face planes only
  (8b587ce7) and `FluxRegister::reset()` drops it too (c3f2ff8c); the positivity limiter keeps the
  x1 wall faces prescribed (0ec6b1d3); dt is a CUDA-graph input, not part of the graph key, with a
  per-block device dt buffer (b84b8bdb).
- **One home for each helper.** The first audit (33 commits, ff239db6..faa65213): one home for
  each duplicated helper across layout, coord, eos, riemann, recon, scalar and output; the aneos
  adiabatic index and sound speed in one place (be018f65); the deck readers take their defaults
  from the option tables. The second audit (0456087d, f8c0c66d, fec6b5e3, 14c7b221, b5212677):
  unreached code, unread fields, commented-out code, one-caller wrappers, a duplicate ctest entry.
  Option defaults read from their option structs (b9632696, 80ac12e9, c07afc5d, 1477afb0,
  d1eae7bf); the two unread output options (f4fa5f63) and the six unused `Quantity` rows (833aca42)
  removed. Comments name the issue, not people or iterations (d258cbf1).
- **Build and CI.** kintera from PyPI instead of a source build (b541a6db), then 2.6.12 on every
  leg (1497416a, 7d48cfa4), then 2.6.13 on every leg with the seam energy reference re-recorded
  (f7034194; see "kintera pin"). The parallel tests keep `snap_cu` linked (750a5b09). `SNAP_SQR` is
  a macro again so contracting compilers fuse as before ff66651a (7a70d38f) and
  `ANEOSThermoImpl::create` keeps the pointer it made (de04b2a5): the two macOS fixes. The graph
  tests skip unless the build has CUDA (b7d47b34).
- **CUDA graphs.** A Mesh owns one stage executor and runs its blocks' stages through it
  (1515539d); the stage is a schedule of segments keyed (block, stage, segment), with the
  stage-entry exchange as its own host-exchange segment and the tendencies, user forcings and
  combine as ops (7e487d25, d7b608f3, 629d1b58, 3d81ba4d, 52b7cdf7); a block's graphs share one
  memory pool and a block with no live graph captures into a fresh pool (1ffccfd0, 9bd69762); an
  outflow boundary and a user stage forcing are per-op refusals (f398794a, a647120f); a cut at an
  exchange lets a Mesh's blocks capture together, the cubed sphere's panel seams are cuts, the
  same-process seam copies are a graph, and a lone block's exchanges are no cut (30cf8389,
  28350ed5, 8c211e7c, 16256d62); the vertical implicit solve reads the device dt and runs under
  cuda-graph (0c939da6, 743d2931); a registered chemistry step is an eager segment and a Mesh with
  chemistry captures its blocks' stages (051212e9, 6529741a); a Mesh whose blocks disagree runs
  eager (4b21ca22); the ideal-gas EOS, lmars and no forcing capture, and capture allows ideal-gas,
  ideal-moist, lmars and hllc only (cb80feaa, 43fae31b); the register monitor's amounts and the
  block's `EfStatus` slots, `conversion_trips` included, are in the capture save list (e6fb9967,
  7ac0d8b7, 66614c32, 4c023985); a stretched cubed sphere captures (069769ba);
  `integration.execution` defaults to cuda-graph on a CUDA device (02e4ef05). Gate on every row:
  graph equals eager bitwise in FP64 and FP32 with dt changing every step.

## Fixes landing with this PR

These sit on the named forks, each on 4c023985, each checked by a non-author; they are placed on
`v3/final` before the PR opens. One entry per mechanism. Each entry has a RED test where behaviour
or a reachable path changed, and says so where it has none; each passed the dump set byte-identical
in FP64 and FP32 on CPU and CUDA and the graph-equals-eager tests.

- **Capture streams, split panels and many-block Meshes (issue #317).** Each block captures on a
  stream the executor creates and owns once; the seam copies capture on an owned stream too; the
  Mesh worker pool creates and keeps one stream per worker, so no stream is taken from the shared
  pool anywhere in `src`; the executor's exchange entry decides between a cut and a plain launch,
  and the lone-block special case is gone; the split-panel refusal is deleted. Split panels and
  Meshes of many blocks now capture and equal eager bitwise. REDs: a many-block Cartesian Mesh
  aborts under capture; 2x2 and 4x4 cubed spheres must capture; two blocks never share a worker
  stream at 33 or more blocks. jihenghu/snapy `v3-closing/audit2-executor`: 4f999511, 1ab00ddb,
  bf5d77da, dce9d7f0, 65d07e81, f7a7fa44, 10166261. The default execution's time against explicit
  eager on the tested GPU is in the issue's correction comment (UCzhangxi/snapy#317).
- **Executor schedule, dispatch and capture helper.** One schedule for a Mesh and for a bare
  block, with `advance_stage`, `advance_core` and `with_host` deleted (00b93e11); one shared
  dispatch for the local and the stage run (b4140033); one begin/end/abandon capture helper shared
  by the stage segments, the error path and the seam copies (4c798e8d); the advance op's prepare
  prepares the implicit correction (1e208dac); the execution allowed-values copy in `meshblock.cpp`
  is deleted, the option table keeps the one check (1695f685); a string compare replaced by
  `is_cubed_sphere()` (58eeeeef). No behaviour change, so no RED. jihenghu/snapy
  `v3-closing/audit2-executor` 00b93e11.
- **Register booking: the wall fold and the pre-divergence apply.** `overwrite_` is the one
  write-and-book for the prescribed planes and the field; the separate wall-plane apply is folded
  into it and both paths book through the wall batch (5146e32e); add and scale always mark a
  direction that has prescribed faces, the `keeps_prescribed` special case is gone (1842c69c, RED
  98c9bf47); a rebuilt weights tensor gets a new slot through the existing path (fee50d53, RED
  cb811e6a); the prescription is applied once per active direction after the sources' posts and
  before the divergence, so a source's post on a wall never reaches du (146f724e, RED 8c73503e);
  the face-floor and Riemann event counts have one counting path (7e4d399c). cshsgy/snapy
  `v3-closing/register-owner`.
- **Fixed-temperature walls conduct.** A wall whose boundary prescribes a temperature has no mass
  flux but conducts, as on main; the register learns this in its one home, `supply_prescribed`,
  which frees the energy row on that wall's plane, on the planes path and on the solid-mask path
  alike (68cbb3ff, RED 0cdfe483; 71f1bb6b, RED 3531039c; the reflecting leg of the test bounded on
  the column's energy scale, cca49795). Result-changing by design on decks with a prescribed wall
  temperature, see the table below. cshsgy/snapy `v3-closing/register-owner` cca49795.
- **Seam exchange in Exchange.** `Exchange::resolve_seam` and `Exchange::seam_exchange` own the
  panel-seam exchange; `reconcile_panel_seams` packs, calls them through the executor's cut, and
  unpacks; `seam_slab` is the allocating wrapper (b7e9c51c). No behaviour change, so no RED;
  verified by the cut trace and the state. cshsgy/snapy `v3-closing/register-owner`.
- **Diffusion energy and the implicit solve go through the register.** A disabled flux direction
  keeps its total defined and zero every stage; `post_fluxes` always adds the energy row; the
  `add()` energy fallback, the early return and `posted_` are deleted (158a5a6e, RED 9fbee19a).
  The implicit solve always receives the register and the x1 total, the null-register branch and
  the rate rebuild are deleted (bb39fc75, RED ed79c058). zoeyzyhu/snapy
  `v3-closing/diffusion-energy-register`, `v3-closing/implicit-register`.
- **Precision policy: `wide()` and the implicit row.** `PrecisionPolicy::wide()` is the one rule
  for the wide dtype on a device, used by the register and by every former copy of the MPS rule
  (b81fae01, cc49ed56); the implicit dtype reads the policy's implicit row (340ffb6c); the implicit
  FP64 retry dtype is that row's storage half, the literal FP64 and the w-dtype special case are
  gone (3a5588b5). No reachable path changes, so no RED; the retry-path dumps are byte-identical.
  XinyueWang05/snapy `v3-fix/precision-policy` 3a5588b5.
- **`conversion_trips` as an `EfStatus` slot.** The counter is bound to its own slot like the
  positivity meters, so the save-list loop saves and restores it and the name lookup and the
  separate hydro buffer are deleted (c4ad6cd3). No RED: name, type and allocation are unchanged and
  the existing graph-equals-eager meter test pins it. chengcli/snapy
  `v3/fix3-conversion-trips-slot` c4ad6cd3.
- **One x1 stencil, one key check, a layout option table.** `CoordinateImpl::x1_face_difference`
  is the only x1 stencil, called by the divergence and the gravity work (5f48c89e, with the FP32
  cast fixed in 344ebfa4); `from_node` takes the allowed keys and does the one check (36a6ddca);
  `layout.yaml` is a struct table like `internal_boundary` (c2c0ca59). gitlinffff/snapy_linfel
  `v3-closing/audit3-leads` 344ebfa4; its non-author check is pending.

## Result-changing rows and their bounds

Everything not listed here is byte-identical to the base on the dry-run gates (the #349 dump set
in FP64 and FP32, the #348 face decks, nb/pz, x1_golden) on CPU and CUDA. Bounds are relative to
the field (row) maximum unless stated. Evidence: `aam/w3fin/report.md` (d3178f20 vs main deeabfee),
the lane posts in thread 1794982799.250779 and `docs/v3/closing/dry_run_plan.md`.

| row | what moves | bound (measured) | declared |
|---|---|---|---|
| b316baf4, explicit species sums | the moist decks, FP64: #349 7 files (4 moist sedimentation cases, hydro_u and scalar_s); x1_golden jupiter_crm_fp64; nb/pz cart3d and cart3d_pert (44/49 dumps) | #349 IVX 1.2e-13 CPU / 1.5e-13 CUDA, other fields <= 2e-15; x1_golden 1.2e-12 (one ulp of the state, equal to the 1-ulp u0 mutant); nb/pz <= 4.3e-15 of the dump max | "round-off change in the moist decks, FP64 <= 1.2e-13" |
| e341ffdf, FP32 typed constants | every FP32 run on a weno/LMARS deck: #349 FP32 21/23 files; x1_golden cart_moist_fp32, jupiter_crm_fp32 | FP32-vs-own-FP64 worst unchanged (1.703e-3 vs 1.707e-3, gnomonic_weno5s IVX; per-field ratio 0.71-1.46); x1_golden at the mutant's level (2.4e-4, 1.6e-5) | "FP32 re-baselined against FP64; FP64 bitwise" |
| 4d41d797, the FP32 sweep gauge | CUDA jupiter_crm_fp32 | 1 ulp | "on CUDA jupiter_crm_fp32 moves by 1 ulp" |
| faa65213, boundary velocity reads the stored sine | declared round-off; the audit gates (#349 FP64/FP32, #348 face, nb/pz) were byte-identical on CPU and CUDA sm_75 at c42d4fff | 0 on the gates | "round-off" |
| kintera 2.6.13 (f7034194): Rgas 8.31446 -> 8.31446261815324 (3.1e-7 relative) | every moist result; the seam test's energy flux sum | moist dump fields <= 1.1e-9 (nb/pz 44/49 dumps differ, the 5 dry sph_face dumps identical; measured on d3178f20); seam energy sum 7.6e-7, reference re-recorded 342168.95387051086 -> 342168.69510693091 (momentum sum unchanged to 1.5e-11) | pinned exactly so every leg compares like for like |
| macOS fix 7a70d38f (SNAP_SQR macro) | arm64/AppleClang only: lmars, hllc, roe, WENO and the implicit decomposition fuse multiply-adds as before ff66651a | Linux: #349 byte-identical, ctest by name identical; macOS: test_fp32_typed_constants 4/5 differing -> 0/0 | restores the pre-ff66651a numbers on contracting compilers |
| 2642c0a5 | one new compiler warning (tests/test_riemann.cpp:585, mkstemp) | test-only | noted on #309 |
| the pre-divergence apply (146f724e) and the conducting wall (68cbb3ff, 71f1bb6b) | only decks where a source posts on a prescribed wall face: a diffusion or boundary-function temperature source on an x1 wall. The wall's conduction reaches du as on main; a source's post on a wall without a prescribed temperature no longer reaches du. No #349, #348, nb/pz or x1_golden case moves; the register's prescribed amounts move at round-off on the folded path | 0 on the gates; the conducting-column test holds the conduction to 1e-9 relative and the reflecting wall to 1e-12 of the column's energy | "the fixed-temperature wall conducts; other wall posts are removed and booked" |
| CUDA graphs | none in eager mode; graph replay equals eager bitwise | 0 (per-step hashes, FP64 and FP32, dt changing) | bitwise |

Environment-only failures, identical on base and head on every machine used: the Eigen entries
the ctest regex catches (Not Run), the restart/NetCDF tests on a NETCDF=OFF build, test_straka,
test_exchange_ucx_cuda (two GPUs), test_python_import_path (sandbox path), the budget tests without
NetCDF, the six nb/pz sph cell runs (refused since the gravity-work cell was removed).

## kintera pin

- `v3/final` ships pinned to kintera **2.6.13**: CI installs `kintera==2.6.13` on every leg
  (`.github/workflows/ci.yml`), `pyproject.toml` requires `kintera>=2.6.13`, `CMakeLists.txt`
  `find_package(Kintera 2.6.13 REQUIRED)`, release.yml pins `kintera>=2.6.13`. 2.6.13 is on PyPI
  (chengcli/kintera#159, v2.6.13): exact SI Rgas, `nfail_` dropped, third-body and falloff layout
  fixes.
- Why the exact pin: 2.6.12 prepares the moist column's thermo state on MPS (kintera #158), 2.6.11
  asserts in PyTorch's dispatcher at the retry's prepare; 2.6.13's Rgas moves the moist numbers
  (see the table), so every leg must compare like for like.
- The baseline (main deeabfee) does not build against 2.6.13: it reads `pthermo->nfail_`
  (src/mesh/meshblock_redo.cpp:124-126), dropped in 2.6.13. The baseline is not patched. Per
  `docs/v3/closing/dry_run_plan.md`: the correctness and speed legs build **both sides on kintera
  2.6.12**, and one V3-only leg, V3@2.6.12 vs V3@2.6.13, measures the kintera effect by itself.
- Building a V3 head on 2.6.12: heads after f7034194 carry `find_package(Kintera 2.6.13 REQUIRED)`
  and refuse the 2.6.12 package at configure. The dry-run driver (`docs/v3/closing/tools/dryrun.sh`)
  lowers that one floor to 2.6.12 for the k2612 configure only and restores the tree (markers in
  its log); no source changes. The seam test is RED on 2.6.12 by design (its reference is 2.6.13's).

## Gates on the landed head

The gates and their owners are in `docs/v3/closing/dry_run_plan.md`; the verdicts are posted in
thread 1794982799.250779 and summarised here without numbers.

- The #349 dump set, FP64 and FP32, CPU and CUDA: byte-identical between 4c023985 and its
  predecessor on each tested machine (H100 node, RTX 5090, GTX 1080 Ti, sm_75) and between every
  fix branch above and 4c023985; against main deeabfee, identical except the rows of the table
  above, each within its declared bound.
- The #348 face decks, nb/pz and x1_golden against main: identical except the rows of the table
  above, on the rehearsal head; the landed-head re-run of these three legs follows the owner table
  in `docs/v3/closing/dry_run_plan.md` and its verdicts are posted in the thread before the PR
  opens.
- Serial ctest compared by name, CPU and CUDA, on every machine above: no verdict changes between
  4c023985 and its predecessor, nor between any fix branch and 4c023985, beyond each branch's own
  new tests; the environment-only failures are listed above and are the same on both sides.
- Graph equals eager, bitwise, FP64 and FP32, with dt changing every step: on every captured deck,
  including the moist four-block Mesh with every `EfStatus` meter compared, the stretched cubed
  sphere, 2x2 and 4x4 cubed spheres and the 8-, 24- and 40-block Meshes.
- The speed gate: the CUDA default execution runs within the ruled bound of main eager on every
  cell, the CPU eager path within its bound, and the CUDA explicit-eager gap is reported. The table
  is `docs/v3/closing/s4_eager_gap.md` (sections 5.5 and 5.6 for the landed head; the bound as
  ruled in section 1).
- Fork CI (pre-commit, bitwise-cpu, ubuntu, macOS) is green on 4c023985.

## Follow-ups (not in this PR)

- Mesh chemistry and the closure (UCzhangxi/snapy#318): a deck with both runs chemistry after the
  closure through the `defer_chemistry` bypass; the reorder waits on a working deck that exercises
  both and on kintera's process-wide `LogSVPFunc::_op`, which running chemistry on the Mesh workers
  needs changed first.
- The CUDA eager gap: V3 explicit eager is slower than main eager by the amounts recorded in
  `docs/v3/closing/s4_eager_gap.md`; the remaining launches are the per-stage census and register
  work on the eager path, inside a graph they replay for free.
- Capture scope: capture allows the ideal-gas and ideal-moist EOS with lmars or hllc only; other
  EOS and solvers, more than one process and pz > 1 stay refused until each is tested; the FP32
  cubed sphere cannot be built.
- The FP64 retry-column count is kept twice (implicit_hydro.cpp and hydro.cpp, reset differently);
  the meters are not a bypass. Left for the implicit and hydro owners.
- In an FP64 block, swapping `hydro_w` to FP32 after initialize aborts with heap corruption, with
  the fallback off; not the retry's doing and not changed here.
- O3 (deferred by the maintainer): later, as its own PR.
- #369 (terrain, the per-column immersed solid on the x1-wall path): after V3. The column case of
  issue thread 1794709389.746279 (a Cartesian rest column over a solid, uniform vs x1rat 1.05) is still
  present at b84b8bdb: earth fill redoes at step 1, matched fill leaves a steady 2-cell w; the wall
  twin sits at round-off.
- Known, not changed: the single-block gnomonic decks' sensitivity to the boundary.prescribed item;
  test_python_import_path depends on the sandbox's Python path.
