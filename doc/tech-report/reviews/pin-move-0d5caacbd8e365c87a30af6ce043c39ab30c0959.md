# Pin move 73c45a7c -> 0d5caacb: what resolved, the changed bodies, and what was re-read

STYLE.md section 3.1: when the pin moves, each range is re-resolved by locating its anchor in the new file; where
the anchor has moved, been renamed or been deleted, the script lists the citation and the author re-reads the code
and rewrites the citation and, if the code changed, the claim.

Old pin `73c45a7c8c991209924e1137992352c335575062`, new pin `0d5caacbd8e365c87a30af6ce043c39ab30c0959`
(`chengcli/snapy` main after five squash merges: `chengcli/snapy#379`, the `gravity-work` and `gravity-work-fixer`
keys and `api/pd-run` removed; `#381`, `report()` replaced by `report_input` and the input schemas; `#382`, a
Mesh's block runs its chemistry in its own last stage; `#383`, the same-process CPU exchange runs in parallel with a
cached plan and the rendezvous states are erased with their Mesh; `#384`, CUDA graphs for the built-in forcings,
with `plume-forcing` the one source that refuses). The census that scoped this move was posted in the pin-move
thread; its numbers are repeated below where the PR relies on them.

## What resolved

Before the bump, `python3 tools/code_anchors.py diff 73c45a7c 0d5caacb` reported 7 distinct citations (9 lines)
that do not resolve at the new pin; they were fixed first (next section). `python3 tools/code_anchors.py bump
0d5caacb...` (Universal Ctags) then resolved every `{{< code >}}` citation: 1155 distinct symbols, 0 unresolved, 154
with a changed body (147 in the census at `a71bfa1b` plus the replacements), and wrote the new sha into
`code_pin.yml`; `check` passes with the legacy chapters skipped: three (ch14, ch16, appB) before #357 was merged
in, two (ch16, appB) after it, with ch14 checked in full (0 unresolved). The legacy list is not empty: ch16's 7 and
appB's 1 unresolved citations (`src/recon/ppm.cpp`, `src/recon/interp_simple.hpp`, `src/recon/interpolation.cpp`,
`ORDER_OFF` of `tests/test_wb_ref4_order.py`, `flux_positivity_scale_`) did not resolve at `73c45a7c` either; they
are their owners' content, not the move's.

## Resolved by hand

- ch06 `_cell.qmd`, `_fixer.qmd`, `06-gravity-energy.qmd`: the `gravity-work`/`gravity-work-fixer` branches of
  `ConstGravityOptionsImpl::from_yaml` are gone; the sections now describe the key check (`check_keys`, four
  keys) and the refusal `unknown key 'forcing/const-gravity/<key>'`, with `test_const_gravity_options_python`
  in place of `test_gravity_work_fixer_removed_python`, and the PROVENANCE pin at `0d5caacb`.
- ch15 `_options.qmd`: `options_table.pin_report` is gone; the row cites the three `InputReport.*` gtest cases
  of `test_input_report.release` and the Reporting paragraph describes `report_input`/`input_report_schema`.
- ch15 `_output.qmd`: `OutputPrecision.yaml_double_precision_defaults_off_and_is_reported` and
  `netcdf_output_without_netcdf_writes_nothing` are renamed `..._and_accepts_override` and `..._warns_once`.
- ch17 `17-known-limits.qmd`: the two `api/pd-run` rows are dropped (the script is gone), rows 265 and 276 reworded
  for the key check, the count table 152.
- preface: the `pyproject.toml` entry-point links name the pin; the `api/pd-run` sentence says the script is gone.

## Changed bodies, by symbol

The 147 cited symbols whose body differs between the pins (`code_anchors.py diff`, census of `a71bfa1b`), the
chapters that cite them, and what was done. A commit sha names the commit of this branch that carries the edit.

| # | symbol | lines (old -> new) | +/- | cited by | re-read |
|---|---|---|---|---|---|
| 001 | `.github/workflows/ci.yml` | file | +13/-61 | ch12(1) | #381/#379 CI: `pip install kintera==2.6.14`; the kintera pin commit (3b565bf1) and appE/preface (006d4c61) follow it; ch12 cites the file |
| 002 | `CMakeLists.txt` | file | +3/-1 | appe(2) appj(1) ch01(6) ch12(11) ch15(2) | `find_package(Kintera 2.6.14)`: the kintera pin choice (3b565bf1, v2.6.14 not v2.6.15); no claim names the line |
| 003 | `docs/derivations/curved-gravity-work-weight.md` | file | +15/-16 | ch06(5) ch17(2) | derivation note: comment wording (`gravity-work: face` -> face-based work); the cited sections and equations are unchanged |
| 004 | `pyproject.toml` | file | +2/-3 | appe(1) appj(2) ch01(7) ch15(2) | #379: the `api/pd-run` script is gone; preface rewritten (ea91b1f4), ch17 rows dropped (a3c18776) |
| 005 | `python/csrc/pycoord.cpp bind_coord` | L16-L131 -> L16-L124 | +5/-12 | ch01(2) ch03(1) | #381: `__repr__` through `options_repr`/`module_repr` instead of `report()`; no chapter describes `__repr__`; ch01/ch03/ch04/ch05 cite the binders as code maps (8: the `set_chemistry` docstring now says last-stage op, matching 7843b571) |
| 006 | `python/csrc/pyhydro.cpp bind_hydro` | L17-L68 -> L17-L63 | +1/-6 | ch01(3) ch04(2) ch05(1) | #381: `__repr__` through `options_repr`/`module_repr` instead of `report()`; no chapter describes `__repr__`; ch01/ch03/ch04/ch05 cite the binders as code maps (8: the `set_chemistry` docstring now says last-stage op, matching 7843b571) |
| 007 | `python/csrc/pylayout.cpp bind_layout` | L22-L137 -> L22-L130 | +5/-12 | ch01(3) | #381: `__repr__` through `options_repr`/`module_repr` instead of `report()`; no chapter describes `__repr__`; ch01/ch03/ch04/ch05 cite the binders as code maps (8: the `set_chemistry` docstring now says last-stage op, matching 7843b571) |
| 008 | `python/csrc/pymesh.cpp bind_mesh` | L19-L386 -> L19-L381 | +3/-8 | ch01(4) ch15(2) | #381: `__repr__` through `options_repr`/`module_repr` instead of `report()`; no chapter describes `__repr__`; ch01/ch03/ch04/ch05 cite the binders as code maps (8: the `set_chemistry` docstring now says last-stage op, matching 7843b571) |
| 009 | `python/csrc/pyrecon.cpp bind_recon` | L13-L82 -> L13-L72 | +2/-12 | ch01(2) | #381: `__repr__` through `options_repr`/`module_repr` instead of `report()`; no chapter describes `__repr__`; ch01/ch03/ch04/ch05 cite the binders as code maps (8: the `set_chemistry` docstring now says last-stage op, matching 7843b571) |
| 010 | `python/csrc/snapy.cpp PYBIND11_MODULE` | L31-L60 -> L34-L74 | +11/-0 | ch01(3) ch15(2) | #381: the module gains `report_input(filename, section)`; ch15 _options describes `report_input` (a3c18776) |
| 011 | `python/snapy/output.pyi` | file | +0/-8 | ch15(1) | #381: the `report` stub lines are gone; ch15 _options (a3c18776) |
| 012 | `src/CMakeLists.txt` | file | +1/-0 | appe(6) appj(1) ch01(3) ch02(1) ch07(1) ch12(2) | src/CMakeLists +1 source file; cited as a file by appE/appJ/ch01/ch02/ch07/ch12, no line claim |
| 013 | `src/bc/internal_boundary.hpp InternalBoundaryOptionsImpl` | L26-L50 -> L26-L42 | +0/-8 | ch07(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 014 | `src/coord/coordinate.hpp CoordinateOptionsImpl` | L32-L149 -> L32-L132 | +0/-17 | ch01(4) ch03(4) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 015 | `src/diagnostics/budget.cpp` | file | +1/-1 | appj(1) | comment wording only (`gravity-work: face` -> face-based) |
| 016 | `src/diagnostics/budget.cpp print_cycle_line` | L281-L392 -> L281-L392 | +1/-1 | ch06(4) ch08(2) ch17(2) | comment wording only (`gravity-work: face` -> face-based) |
| 017 | `src/eos/aneos.cpp ANEOSImpl::_cons2prim` | L79-L100 -> L82-L104 | +1/-0 | ch02(2) | ANEOS `_cons2prim` raises the velocity with the cell cosine: ch02 _aneos excerpt, Numerical method, Limits item 4 removed, test row added (e4ed75eb) |
| 018 | `src/eos/equation_of_state.cpp EquationOfStateImpl::apply_primitive_limiter_` | L459-L484 -> L459-L485 | +1/-0 | ch02(1) ch08(9) ch16(1) | shallow water returns after the density floor: ch08 _eoslimiter sentence (e4ed75eb) |
| 019 | `src/eos/equation_of_state.hpp EquationOfStateOptionsImpl` | L22-L63 -> L22-L46 | +0/-17 | ch02(3) ch08(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 020 | `src/eos/ideal_gas.cpp IdealGasImpl::_cons2prim` | L73-L92 -> L73-L98 | +15/-9 | ch02(3) | ideal-gas `_cons2prim` reads `cons` in `prim`'s dtype and row stride (new test `test_ideal_gas_cons2prim_layout`); ch02 describes the floors and the division, not the kernel's iterator: no claim change |
| 021 | `src/forcing/body_heat.cpp BodyHeatImpl::add` | L30-L40 -> L30-L40 | +1/-1 | ch07(3) ch17(1) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 022 | `src/forcing/const_gravity.cpp ConstGravityOptionsImpl::from_yaml` | L12-L45 -> L12-L29 | +1/-17 | appd(1) ch02(1) ch05(3) ch06(5) ch12(6) ch17(2) | #379: `check_keys` with four keys, the work-form and fixer branches gone: ch06 (a3c18776), ch02 _pecore, ch05 _oneoverr, ch12 _yaml/_radial-exact, ch17, appD (2447ca10) |
| 023 | `src/forcing/coriolis.cpp CoriolisImpl::add` | L110-L166 -> L110-L168 | +9/-7 | ch07(6) ch17(1) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 024 | `src/forcing/diffusion.cpp` | file | +6/-30 | ch07(12) | #384: `check_profiles` moved out of `reset` and skipped under capture; diffusion capturable: 110bd187 (ch07 _framework) |
| 025 | `src/forcing/diffusion.cpp DiffusionImpl::add` | L551-L570 -> L527-L546 | +2/-2 | ch07(6) | #384: `check_profiles` moved out of `reset` and skipped under capture; diffusion capturable: 110bd187 (ch07 _framework) |
| 026 | `src/forcing/diffusion.cpp DiffusionImpl::reset` | L392-L517 -> L367-L493 | +1/-0 | ch07(9) | #384: `check_profiles` moved out of `reset` and skipped under capture; diffusion capturable: 110bd187 (ch07 _framework) |
| 027 | `src/forcing/diffusion_cubed_sphere.cpp DiffusionImpl::cubed_sphere_fluxes` | L167-L357 -> L172-L383 | +30/-9 | ch07(8) ch17(3) | #384: cubed-sphere diffusion posts its four seam slabs then takes (`resolve_seam`/`seam_exchange`), send buffers retained across replays: ch07 _csdiffusion block-edge and placement text (110bd187); the flux formulas and the penalty are unchanged |
| 028 | `src/forcing/diffusion_cubed_sphere.cpp DiffusionImpl::forward_cubed_sphere` | L379-L394 -> L405-L420 | +1/-1 | ch07(2) | #384: cubed-sphere diffusion posts its four seam slabs then takes (`resolve_seam`/`seam_exchange`), send buffers retained across replays: ch07 _csdiffusion block-edge and placement text (110bd187); the flux formulas and the penalty are unchanged |
| 029 | `src/forcing/diffusion_cubed_sphere.cpp DiffusionImpl::post_fluxes` | L365-L377 -> L391-L403 | +1/-1 | ch07(5) | #384: cubed-sphere diffusion posts its four seam slabs then takes (`resolve_seam`/`seam_exchange`), send buffers retained across replays: ch07 _csdiffusion block-edge and placement text (110bd187); the flux formulas and the penalty are unchanged |
| 030 | `src/forcing/diffusion_cubed_sphere.cpp share_fluxes` | L102-L164 -> L103-L169 | +15/-11 | ch07(3) ch17(1) | #384: cubed-sphere diffusion posts its four seam slabs then takes (`resolve_seam`/`seam_exchange`), send buffers retained across replays: ch07 _csdiffusion block-edge and placement text (110bd187); the flux formulas and the penalty are unchanged |
| 031 | `src/forcing/diffusion_cubed_sphere.cpp weights` | L40-L83 -> L39-L83 | +3/-2 | ch07(2) | #384: cubed-sphere diffusion posts its four seam slabs then takes (`resolve_seam`/`seam_exchange`), send buffers retained across replays: ch07 _csdiffusion block-edge and placement text (110bd187); the flux formulas and the penalty are unchanged |
| 032 | `src/forcing/diffusion_radial.cpp DiffusionImpl::begin_split overload=2` | L14-L20 -> L23-L29 | +3/-3 | ch07(2) ch09(3) | #384: `begin_split` takes `StepDt`; `radial_conduction_step` validates the device step and cuts the graph; the SDIRK message ch09 _redo item 6 quotes is still at the pin (checked) |
| 033 | `src/forcing/diffusion_radial.cpp DiffusionImpl::radial_conduction_step` | L32-L149 -> L41-L176 | +33/-15 | ch07(6) ch09(2) ch17(1) | #384: `begin_split` takes `StepDt`; `radial_conduction_step` validates the device step and cuts the graph; the SDIRK message ch09 _redo item 6 quotes is still at the pin (checked) |
| 034 | `src/forcing/forcing.hpp PlumeForcingImpl` | L357-L368 -> L348-L364 | +7/-2 | ch07(1) | #384: `plume-forcing` declares `capturable = false`: ch11 _refusals row and caption, ch07 _framework (110bd187) |
| 035 | `src/forcing/layer_heat.cpp LayerHeatImpl::add` | L62-L85 -> L62-L90 | +9/-4 | ch07(5) ch17(3) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 036 | `src/forcing/relax_bot_comp.cpp RelaxBotCompImpl::add` | L65-L96 -> L64-L95 | +1/-1 | ch07(4) ch17(1) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 037 | `src/forcing/relax_bot_temp.cpp RelaxBotTempImpl::add` | L29-L72 -> L29-L72 | +1/-1 | ch07(7) ch17(3) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 038 | `src/forcing/relax_bot_velo.cpp RelaxBotVeloImpl::add` | L21-L45 -> L21-L45 | +1/-1 | ch07(3) ch17(2) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 039 | `src/forcing/source_term.hpp SourceRegistry` | L90-L162 -> L90-L174 | +12/-0 | ch07(2) | #384: `SourceRegistry::capture_reasons` ("<name> does not support CUDA graph capture"); ch07 _framework and ch11 _refusals (110bd187) |
| 040 | `src/forcing/source_term.hpp SourceTerm` | L54-L88 -> L54-L88 | +4/-4 | ch07(2) | #384: `SourceRegistry::capture_reasons` ("<name> does not support CUDA graph capture"); ch07 _framework and ch11 _refusals (110bd187) |
| 041 | `src/forcing/sponge.cpp SpongeImpl::add` | L47-L63 -> L47-L62 | +1/-2 | ch07(4) ch17(1) | #384: `::add` takes `StepDt` (device dt); the forcing's formula is unchanged; the capture claims are 110bd187 (ch07 _framework, _relax; ch11 _refusals) |
| 042 | `src/hydro/flux_register.cpp` | file | +16/-8 | appj(1) | #384: `_split_posted` flags replace the `_direct.defined()` test, `post_split` takes `StepDt`: ch10 _amounts code map, ch01 _stage `end_stage` excerpt (110bd187); the recursion @eq-ch01-stage-weight is unchanged |
| 043 | `src/hydro/flux_register.cpp FluxRegister::begin_attempt_` | L461-L479 -> L463-L481 | +1/-1 | ch10(2) | #384: `_split_posted` flags replace the `_direct.defined()` test, `post_split` takes `StepDt`: ch10 _amounts code map, ch01 _stage `end_stage` excerpt (110bd187); the recursion @eq-ch01-stage-weight is unchanged |
| 044 | `src/hydro/flux_register.cpp FluxRegister::end_stage` | L761-L776 -> L763-L778 | +1/-1 | ch01(4) ch10(3) ch17(1) | #384: `_split_posted` flags replace the `_direct.defined()` test, `post_split` takes `StepDt`: ch10 _amounts code map, ch01 _stage `end_stage` excerpt (110bd187); the recursion @eq-ch01-stage-weight is unchanged |
| 045 | `src/hydro/flux_register.cpp FluxRegister::post_split` | L778-L797 -> L780-L805 | +11/-5 | ch10(2) | #384: `_split_posted` flags replace the `_direct.defined()` test, `post_split` takes `StepDt`: ch10 _amounts code map, ch01 _stage `end_stage` excerpt (110bd187); the recursion @eq-ch01-stage-weight is unchanged |
| 046 | `src/hydro/flux_register.cpp FluxRegister::reset overload=1` | L404-L441 -> L404-L443 | +2/-0 | ch10(2) | #384: `_split_posted` flags replace the `_direct.defined()` test, `post_split` takes `StepDt`: ch10 _amounts code map, ch01 _stage `end_stage` excerpt (110bd187); the recursion @eq-ch01-stage-weight is unchanged |
| 047 | `src/hydro/flux_register.hpp FluxRegister` | L241-L402 -> L241-L405 | +7/-4 | appg(1) ch10(2) | #384: `_split_posted` flags replace the `_direct.defined()` test, `post_split` takes `StepDt`: ch10 _amounts code map, ch01 _stage `end_stage` excerpt (110bd187); the recursion @eq-ch01-stage-weight is unchanged |
| 048 | `src/hydro/hydro.cpp HydroImpl::radial_exact_work` | L322-L327 -> L306-L310 | +1/-2 | appd(2) ch06(3) ch12(4) ch17(2) | #379: the `gravity_work() == "face"` gate is gone: ch12 _radial-exact excerpt and gates (2447ca10) |
| 049 | `src/hydro/hydro.cpp HydroImpl::reset` | L27-L279 -> L27-L263 | +18/-34 | appd(6) ch02(2) ch03(5) ch04(10) ch05(8) ch06(2) ch07(1) ch08(3) ch09(2) ch10(7) ch12(14) ch17(5) | #379: `HydroImpl::reset` no longer checks the work form or the fixer; the warnings stay: ch12 _yaml (2447ca10); the ghost/cell guards the other chapters cite are unchanged (diff read) |
| 050 | `src/hydro/hydro.hpp HydroImpl` | L77-L326 -> L77-L327 | +2/-1 | ch01(1) ch04(2) | comment wording / `prelax_comp` member kept for `build_schedule`; no claim |
| 051 | `src/hydro/hydro.hpp HydroOptionsImpl` | L36-L72 -> L36-L72 | +1/-1 | ch01(1) ch04(2) ch17(1) | comment wording / `prelax_comp` member kept for `build_schedule`; no claim |
| 052 | `src/hydro/hydro_forward.cpp HydroImpl::forward` | L299-L1235 -> L299-L1235 | +3/-3 | appd(1) ch01(16) ch02(3) ch03(14) ch04(23) ch05(37) ch06(13) ch07(7) ch08(30) ch09(11) ch10(5) ch12(12) ch14(10) ch17(37) | comment wording / `prelax_comp` member kept for `build_schedule`; no claim |
| 053 | `src/hydro/hydro_options.cpp HydroOptionsImpl::from_yaml` | L19-L115 -> L20-L92 | +3/-27 | appd(3) ch01(1) ch04(4) ch05(5) ch07(7) ch08(1) ch09(1) ch12(4) ch14(1) ch15(3) | #381: `report()` calls -> `report_input` for `dynamics` and `forcing`; the key checks ch12 _yaml cites are unchanged |
| 054 | `src/hydro/register_forcing_modules.cpp HydroImpl::_register_forcings_module` | L8-L44 -> L8-L45 | +2/-1 | ch07(7) | comment wording / `prelax_comp` member kept for `build_schedule`; no claim |
| 055 | `src/implicit/implicit_hydro.cpp ImplicitHydroImpl::book_remainder_` | L713-L792 -> L713-L792 | +1/-1 | ch06(4) ch09(1) ch17(1) | comment wording / `prelax_comp` member kept for `build_schedule`; no claim |
| 056 | `src/implicit/implicit_hydro.cpp ImplicitHydroImpl::forward_masked` | L301-L623 -> L301-L623 | +1/-1 | ch02(1) ch06(3) ch08(3) ch09(16) ch12(2) ch17(3) | comment wording / `prelax_comp` member kept for `build_schedule`; no claim |
| 057 | `src/implicit/implicit_hydro.hpp ImplicitOptionsImpl` | L29-L60 -> L29-L50 | +0/-10 | ch09(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 058 | `src/implicit/vic_kernels.h vic_woodbury_column` | L139-L168 -> L125-L150 | +7/-11 | ch06(1) ch09(2) ch17(2) | the Woodbury column fuses its two loops and drops the clang fp-contract pragma; ch06/ch09 describe the rank-2 correction, unchanged |
| 059 | `src/layout/layout.cpp LayoutImpl::_prepare_local_exchange` | L341-L446 -> L392-L517 | +39/-19 | ch13(1) | #383: the rendezvous plan is cached, CPU copies run per block in parallel, `clear_local_exchange` erases the Mesh's rendezvous states: ch13 _ghost (110bd187) |
| 060 | `src/layout/layout.cpp LayoutImpl::create` | L303-L326 -> L371-L388 | +0/-6 | ch03(3) ch13(1) | #383: `LayoutImpl::create` no longer registers in `g_local_layouts` (the plan holds the layouts); no chapter names `g_local_layouts` |
| 061 | `src/layout/layout.cpp LayoutOptionsImpl::from_yaml` | L291-L301 -> L359-L369 | +1/-1 | appd(3) ch01(2) ch03(3) ch05(1) ch12(5) | #381: `LayoutOptionsImpl::from_yaml` reports through `report_input`; keys unchanged |
| 062 | `src/layout/layout.cpp clear_local_exchange` | L200-L207 -> L264-L275 | +4/-0 | ch13(1) | #383: the rendezvous plan is cached, CPU copies run per block in parallel, `clear_local_exchange` erases the Mesh's rendezvous states: ch13 _ghost (110bd187) |
| 063 | `src/layout/layout.cpp g_local_exchange_states` | L112-L112 -> L198-L200 | +2/-0 | ch13(2) | #383: the rendezvous plan is cached, CPU copies run per block in parallel, `clear_local_exchange` erases the Mesh's rendezvous states: ch13 _ghost (110bd187) |
| 064 | `src/layout/layout.hpp LayoutOptionsImpl` | L58-L139 -> L58-L117 | +0/-22 | ch03(2) ch13(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 065 | `src/mesh/exchange_cut.hpp` | file | +4/-0 | appj(1) | file links (appJ source map): the files exist at the pin |
| 066 | `src/mesh/mesh.cpp` | file | +4/-20 | appj(1) | file links (appJ source map): the files exist at the pin |
| 067 | `src/mesh/mesh.cpp MeshImpl::BlockWorkerPool` | L74-L243 -> L74-L246 | +4/-1 | ch01(2) | #383: `~BlockWorkerPool` calls `clear_local_exchange`: ch13 _ghost Limits (110bd187) |
| 068 | `src/mesh/mesh.cpp MeshImpl::forward` | L371-L426 -> L374-L410 | +0/-19 | ch01(8) ch07(1) ch09(7) ch11(3) ch13(2) ch14(3) ch17(2) | #382: `MeshImpl::forward` no longer defers the chemistry (-19 lines): ch01 _mesh, _stage, _driver, ch09 _exchange, _split, ch10 _kinetics, ch11 _eagerops, _segments, the ch01 Mesh figure (7843b571); ch14 (legacy) still describes the fixer deferral, see Open |
| 069 | `src/mesh/mesh.hpp MeshImpl` | L67-L111 -> L61-L105 | +2/-2 | appg(1) | #382: `defer_chemistry` gone, six redo flags, `_meters0`; comments; covered by 7843b571 and 110bd187 |
| 070 | `src/mesh/mesh.hpp MeshOptionsImpl` | L27-L64 -> L27-L58 | +0/-6 | ch03(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 071 | `src/mesh/meshblock.cpp` | file | +4/-0 | appj(1) | file links (appJ source map): the files exist at the pin |
| 072 | `src/mesh/meshblock.cpp MeshBlockImpl::register_flux_storage` | L440-L483 -> L440-L487 | +4/-0 | ch10(4) | the split register storage (`freg.rec`, `freg.direct`) is registered with the state store; ch10 cites the function for the wall batch and the empty-buffer capture, unchanged |
| 073 | `src/mesh/meshblock.hpp MeshBlockImpl` | L148-L398 -> L142-L392 | +6/-6 | ch08(1) ch09(1) | #382: `defer_chemistry` gone, six redo flags, `_meters0`; comments; covered by 7843b571 and 110bd187 |
| 074 | `src/mesh/meshblock.hpp MeshBlockImpl::set_chemistry` | L259-L263 -> L253-L257 | +1/-1 | ch10(1) | #382: `defer_chemistry` gone, six redo flags, `_meters0`; comments; covered by 7843b571 and 110bd187 |
| 075 | `src/mesh/meshblock.hpp MeshBlockOptionsImpl` | L47-L121 -> L47-L115 | +0/-6 | ch11(2) ch16(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 076 | `src/mesh/meshblock_advance.cpp` | file | +10/-6 | appj(1) | file links (appJ source map): the files exist at the pin |
| 077 | `src/mesh/meshblock_advance.cpp MeshBlockImpl::stage_advance` | L145-L233 -> L145-L238 | +6/-1 | ch01(8) ch07(3) ch08(3) ch09(9) ch10(4) ch11(1) ch17(2) | #384: the split sources take `StepDt`; the EfStatus meters are snapshotted after the state backup: ch08 _efstatus (110bd187) |
| 078 | `src/mesh/meshblock_advance.cpp MeshBlockImpl::stage_chemistry` | L384-L390 -> L388-L394 | +3/-3 | appe(1) ch01(1) ch09(3) | #382: `defer_chemistry` gone, six redo flags, `_meters0`; comments; covered by 7843b571 and 110bd187 |
| 079 | `src/mesh/meshblock_advance.cpp MeshBlockImpl::stage_combine` | L309-L344 -> L314-L348 | +1/-2 | appe(2) ch01(4) ch07(2) ch08(3) ch09(7) ch10(2) ch17(3) | #384: the split sources take `StepDt`; the EfStatus meters are snapshotted after the state backup: ch08 _efstatus (110bd187) |
| 080 | `src/mesh/meshblock_exchange.cpp` | file | +33/-33 | appj(1) | file links (appJ source map): the files exist at the pin |
| 081 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::_exchange_once` | L53-L59 -> L53-L59 | +2/-2 | ch03(1) | formatting (`&` placement) only |
| 082 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::begin_exchange` | L68-L73 -> L68-L73 | +2/-2 | ch03(1) | formatting (`&` placement) only |
| 083 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::exchange` | L33-L51 -> L33-L51 | +2/-2 | ch03(5) | formatting (`&` placement) only |
| 084 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::exchange_ghost_zones` | L118-L201 -> L118-L201 | +3/-3 | ch01(4) ch03(12) ch13(1) ch17(3) | formatting (`&` placement) only |
| 085 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::exchange_state` | L103-L116 -> L103-L116 | +2/-2 | ch03(5) | formatting (`&` placement) only |
| 086 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::finalize_exchange` | L98-L101 -> L98-L101 | +2/-2 | ch03(1) | formatting (`&` placement) only |
| 087 | `src/mesh/meshblock_exchange.cpp MeshBlockImpl::launch_exchange` | L75-L96 -> L75-L96 | +14/-14 | ch03(1) ch13(1) | `launch_exchange` launches directly outside a capture and through one `exchange_cut` lambda inside; ch03 _exchange and ch13 _cuts say "host step is an exchange cut of a captured segment": still true |
| 088 | `src/mesh/meshblock_options.cpp` | file | +15/-13 | appj(1) ch07(1) | file links (appJ source map): the files exist at the pin |
| 089 | `src/mesh/meshblock_options.cpp MeshBlockOptionsImpl::from_yaml` | L17-L137 -> L18-L139 | +14/-13 | appi(1) ch01(10) ch03(5) ch11(2) ch12(3) ch14(7) ch15(7) ch17(1) preface(1) | #381: `MeshBlockOptionsImpl::from_yaml` reports each deck section through `report_input` (geometry and distribute also when omitted): ch15 _options Reporting (a3c18776) |
| 090 | `src/mesh/meshblock_redo.cpp` | file | +9/-6 | appj(1) | file links (appJ source map): the files exist at the pin |
| 091 | `src/mesh/meshblock_redo.cpp MeshBlockImpl::apply_redo` | L32-L88 -> L32-L91 | +3/-0 | appe(2) ch01(3) ch07(2) ch08(7) ch09(4) ch10(4) ch17(1) | #382/#384: six flags, meters restored on a redo: ch09 _redo (110bd187), ch08 _efstatus |
| 092 | `src/mesh/meshblock_redo.cpp MeshBlockImpl::decide_retry` | L168-L198 -> L170-L201 | +3/-2 | ch01(2) ch08(2) ch09(3) | #382/#384: six flags, meters restored on a redo: ch09 _redo (110bd187), ch08 _efstatus |
| 093 | `src/mesh/meshblock_redo.cpp MeshBlockImpl::local_redo_flags` | L164-L166 -> L166-L168 | +1/-1 | ch09(1) | #382/#384: six flags, meters restored on a redo: ch09 _redo (110bd187), ch08 _efstatus |
| 094 | `src/mesh/meshblock_redo.cpp MeshBlockImpl::read_redo_flags` | L147-L162 -> L150-L164 | +2/-3 | ch08(3) ch09(4) ch11(2) ch17(1) | #382/#384: six flags, meters restored on a redo: ch09 _redo (110bd187), ch08 _efstatus |
| 095 | `src/mesh/stage_executor.cpp` | file | +33/-15 | appj(1) ch07(1) | file links (appJ source map): the files exist at the pin |
| 096 | `src/mesh/stage_executor.cpp StageExecutor::build_schedule` | L204-L320 -> L201-L321 | +6/-2 | ch01(7) ch08(2) ch09(7) ch11(3) | #384: `build_schedule` builds `relax-bot-comp`'s kintera tables before the capture: ch07 _relax, _framework (110bd187) |
| 097 | `src/mesh/stage_executor.cpp StageExecutor::capture` | L900-L1044 -> L917-L1062 | +3/-2 | ch11(2) ch13(1) | #384: a lone block with a split source also records with a cut context: ch13 _cuts Placement (e4ed75eb) |
| 098 | `src/mesh/stage_executor.cpp StageExecutor::chemistry_slot` | L184-L190 -> L185-L187 | +1/-5 | appg(1) ch10(1) ch11(2) | #382: `chemistry_slot` is `_chemistry != nullptr` for every block: ch10 _kinetics, ch11 _eagerops (7843b571) |
| 099 | `src/mesh/stage_executor.cpp StageExecutor::config_reasons` | L624-L682 -> L638-L692 | +2/-6 | ch02(3) ch03(1) ch11(2) ch13(3) ch16(1) | #384: the "forcing is not const-gravity alone" refusal is replaced by the sources' own reasons: ch11 _refusals (110bd187) |
| 100 | `src/mesh/stage_executor.cpp StageExecutor::run_segments` | L838-L870 -> L848-L887 | +7/-0 | ch11(2) ch13(1) | #384: `run_segments` checks the diffusion profiles and the radial step before the walk: ch11 _segments (2ce47983) |
| 101 | `src/mesh/stage_executor.hpp StageExecutor` | L99-L237 -> L99-L237 | +2/-2 | appg(1) | #382: `defer_chemistry` gone, six redo flags, `_meters0`; comments; covered by 7843b571 and 110bd187 |
| 102 | `src/options/forcing_body_heat.yaml` | file | +0/-1 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 103 | `src/options/forcing_bot_heat.yaml` | file | +0/-1 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 104 | `src/options/forcing_bot_sponge_lyr.yaml` | file | +0/-1 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 105 | `src/options/forcing_coriolis.yaml` | file | +0/-2 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 106 | `src/options/forcing_relax_bot_temp.yaml` | file | +0/-1 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 107 | `src/options/forcing_top_cool.yaml` | file | +0/-1 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 108 | `src/options/forcing_top_sponge_lyr.yaml` | file | +0/-1 | ch07(1) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 109 | `src/options/hydro.yaml` | file | +0/-3 | appd(6) ch01(1) ch03(2) ch05(1) ch07(1) ch12(5) ch15(6) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 110 | `src/options/layout.yaml` | file | +1/-1 | appd(3) ch01(3) ch03(3) ch13(1) ch15(4) | layout.yaml doc: `slab, cubed or cubed-sphere`; appJ already lists the three |
| 111 | `src/options/output.yaml` | file | +1/-3 | ch15(5) | #381: the `report_title`/`report_order` rows and the output `verbose` key leave the option yaml files; appI regenerated (dd49df61); ch15 _options test row names the refused `verbose` (2ce47983) |
| 112 | `src/output/budget_series.cpp` | file | +6/-1 | appj(1) | file links (appJ source map): the files exist at the pin |
| 113 | `src/output/budget_series.cpp BudgetOutput::bind` | L173-L179 -> L174-L184 | +4/-0 | ch10(2) | a budget output bound to a new block starts its series file again: ch10 _budget (2ce47983) |
| 114 | `src/output/netcdf.cpp` | file | +11/-2 | ch15(1) | NETCDF=OFF netcdf output says so once: ch15 _output test row (98ab72e5/a3c18776) |
| 115 | `src/output/output_type.hpp OutputOptionsImpl` | L31-L49 -> L31-L49 | +1/-1 | ch15(1) | comment only |
| 116 | `src/output/restart.cpp RestartOutput::RestartOutput` | L34-L38 -> L34-L40 | +4/-2 | ch15(2) | restart refuses `combine: false` instead of forcing it on: ch15 _restart (e4ed75eb) |
| 117 | `src/output/writer_core.cpp combine_on_root` | L314-L337 -> L312-L355 | +22/-2 | ch15(3) | the root's combine failure is MAX-reduced to every rank: ch15 _output (e4ed75eb) |
| 118 | `src/output/writer_core.cpp output_range` | L217-L249 -> L217-L247 | +0/-2 | ch15(3) ch17(1) | a FIXME comment removed |
| 119 | `src/recon/reconstruct.hpp InterpOptionsImpl` | L29-L43 -> L29-L38 | +0/-5 | ch17(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 120 | `src/recon/reconstruct.hpp ReconstructOptionsImpl` | L103-L132 -> L98-L121 | +0/-6 | ch04(1) ch08(1) ch17(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 121 | `src/riemann/riemann.cpp RiemannSolverImpl::reset` | L98-L126 -> L98-L126 | +1/-1 | ch04(1) | the shallow-Roe gnomonic check compares with `gnomonic-equiangle` and now fires: ch04 _shallow (2ce47983) |
| 122 | `src/riemann/riemann.hpp RiemannSolverOptionsImpl` | L42-L65 -> L42-L59 | +0/-6 | ch04(3) ch17(1) | #381: the options struct's `report()` member removed; no chapter quotes `report()`, the option tables are unchanged (ch15 `pin_report` case gone: a3c18776) |
| 123 | `src/utils/step_dt.cpp` | file | +7/-0 | appj(1) | file links (appJ source map): the files exist at the pin |
| 124 | `src/utils/step_dt.hpp StepDt` | L12-L42 -> L12-L43 | +1/-0 | ch11(2) | `StepDt` gains `operator/`; ch11 cites the class for `scaled`/`on_device`, unchanged |
| 125 | `tests/CMakeLists.txt` | file | +42/-5 | appc(6) appd(1) appj(1) ch01(3) ch02(1) ch03(2) ch04(1) ch08(8) ch11(1) ch12(25) ch15(2) ch16(18) ch17(1) | tests/CMakeLists: +6 registrations (shallow-water limiter, aneos metric, moist temp alias, ideal-gas layout, redo causes, input report), `test_gravity_work_fixer_removed` -> `test_const_gravity_options`: appC regenerated (dd49df61), ch06 test row (a3c18776) |
| 126 | `tests/test_conservation_budget.cpp Budget.closes_to_round_off` | L72-L205 -> L71-L202 | +4/-6 | ch10(1) | `Budget.closes_to_round_off` drops its work-form parameter; ch10's row describes the 15 decks, unchanged |
| 127 | `tests/test_flux_covariance_seams.py arm_conservation` | L62-L100 -> L62-L97 | +0/-3 | ch03(1) ch16(2) | `arm_conservation` no longer sets `gravity-work: face`; ch03 _cslrsync and ch16 describe the totals, unchanged |
| 128 | `tests/test_forcing.cpp` | file | +37/-10 | ch06(1) ch08(2) | test_forcing: the `GravityWork` suite and `gravity_work("face")` lines gone, three `InputReport` cases: ch06 (a3c18776), ch08 cites the file |
| 129 | `tests/test_graph_mesh_chemistry.py` | file | +3/-3 | ch11(1) | test docstring: chemistry as the eager segment of each block's last stage: ch11 _eagerops row (7843b571) |
| 130 | `tests/test_gravity_work_radial_exact.py` | file | +4/-4 | ch06(2) | gravity tests drop the `gravity-work` key from their decks and reword docstrings; ch03/ch06/ch09 rows unchanged |
| 131 | `tests/test_gravity_work_radial_exact.py setup_error` | L256-L284 -> L256-L284 | +1/-1 | ch03(1) | gravity tests drop the `gravity-work` key from their decks and reword docstrings; ch03/ch06/ch09 rows unchanged |
| 132 | `tests/test_implicit_face_work_operator.py` | file | +6/-6 | ch06(1) | gravity tests drop the `gravity-work` key from their decks and reword docstrings; ch03/ch06/ch09 rows unchanged |
| 133 | `tests/test_implicit_face_work_operator.py main` | L131-L173 -> L131-L173 | +1/-1 | ch09(1) | gravity tests drop the `gravity-work` key from their decks and reword docstrings; ch03/ch06/ch09 rows unchanged |
| 134 | `tests/test_lu_failure.cpp lu_failure.a_scheme_zero_module_is_refused` | L499-L553 -> L495-L549 | +1/-1 | ch09(1) | formatting |
| 135 | `tests/test_mesh_executor.cpp` | file | +98/-1 | ch11(1) | test_mesh_executor: `defer_chemistry` line gone; new `repeated_local_exchange_reads_current_buffers_after_failure` case (not yet in a Tests table, see Open) |
| 136 | `tests/test_options_table.cpp options_table.pin_defaults` | L64-L98 -> L65-L98 | +0/-1 | ch15(1) | `pin_defaults` no longer asserts `verbose`; ch15 row lists the defaults that remain (checked) |
| 137 | `tests/test_source_registry.cpp SourceRegistry.a_deck_declares_its_sources_in_deck_order` | L97-L124 -> L99-L123 | +4/-7 | ch07(1) | the ten registry sources form one capturable run: ch07 _framework test row (110bd187) |
| 138 | `tests/test_user_output.cpp` | file | +14/-9 | ch15(3) | `OutputPrecision` cases renamed: ch15 _output rows (a3c18776, 98ab72e5) |
| 139 | `tests/test_x1_seam_split.cpp X1SeamSplit.nghost_check_refuses_fewer_than_3_ghosts` | L339-L362 -> L337-L360 | +3/-3 | ch12(1) | x1-seam tests drop the work-form argument of `make_column`/`split_gap`; ch03/ch12/ch16 rows describe the comparisons, unchanged |
| 140 | `tests/test_x1_seam_split.cpp X1SeamSplit.switches_off_nghost_1_sets_up` | L324-L329 -> L322-L327 | +2/-2 | ch12(1) | x1-seam tests drop the work-form argument of `make_column`/`split_gap`; ch03/ch12/ch16 rows describe the comparisons, unchanged |
| 141 | `tests/test_x1_seam_split.cpp X1SeamSplit.the_x1_relays_are_seam_sides` | L422-L507 -> L420-L505 | +1/-1 | ch03(1) | x1-seam tests drop the work-form argument of `make_column`/`split_gap`; ch03/ch12/ch16 rows describe the comparisons, unchanged |
| 142 | `tests/test_x1_seam_split.cpp X1SeamSplit.x1_mass_covariance_needs_3_x1_cells` | L407-L420 -> L405-L418 | +2/-2 | ch12(3) ch16(2) | x1-seam tests drop the work-form argument of `make_column`/`split_gap`; ch03/ch12/ch16 rows describe the comparisons, unchanged |
| 143 | `tests/test_x1_seam_split.cpp radial_exact_split_gap` | L270-L287 -> L269-L285 | +1/-2 | ch03(1) ch16(2) | x1-seam tests drop the work-form argument of `make_column`/`split_gap`; ch03/ch12/ch16 rows describe the comparisons, unchanged |
| 144 | `tests/test_x1_seam_split.cpp split_gap` | L226-L255 -> L226-L254 | +4/-5 | ch03(1) | x1-seam tests drop the work-form argument of `make_column`/`split_gap`; ch03/ch12/ch16 rows describe the comparisons, unchanged |
| 145 | `tests/test_yaml_keys.cpp` | file | +8/-0 | ch15(1) | `outputs_refuses_verbose`: ch15 _options row (2ce47983) |
| 146 | `tools/gen_options.py` | file | +34/-11 | appj(1) ch15(4) | gen_options no longer emits `report()`, emits the input schemas: ch15 _options already describes `input_report_schema` (a3c18776) |
| 147 | `tools/gen_options.py emit_members` | L105-L119 -> L105-L108 | +0/-11 |  | gen_options no longer emits `report()`, emits the input schemas: ch15 _options already describes `input_report_schema` (a3c18776) |

### Changed bodies cited only by chapter 14 (merged with #357)

The census preceded #357; with ch14 in, `code_anchors.py diff` lists 163 changed symbols. Beyond the 147 above and
the 9 symbols that are new at the pin (the three `InputReport` cases, the two renamed `OutputPrecision` cases,
`report_input`, `test_removed_gravity_options`, `test_aneos_metric`'s case, `test_redo_positivity_meters.py`, all
cited by the sections this move rewrote) and `ConstGravityOptionsImpl` in `src/forcing/forcing.hpp` (the
`gravity_work`/`gravity_work_fixer` members gone; ch06, a3c18776), seven are cited by ch14 alone:

| symbol | lines (old -> new) | cited by | re-read |
|---|---|---|---|
| `python/csrc/pymesh.cpp` | file (+3/-8) | ch14(9) | `__repr__` through `options_repr`; the `set_chemistry` docstring says last-stage op; no ch14 claim names either |
| `python/snapy/mesh.pyi` | file | ch14(1) | the `set_chemistry` docstring; no claim |
| `src/bc/bc.hpp BoundaryFuncOptions` | L29-L43 -> L29-L37 | ch14(3) | #381: `report()` removed; the option fields ch14 lists are unchanged |
| `src/coord/coordinate.hpp` | file (-17) | ch14(1) | #381: `CoordinateOptionsImpl::report()` removed; no claim |
| `src/mesh/meshblock_exchange.cpp MeshBlockImpl::apply_boundaries` | L214-L286 -> L214-L286 | ch14(6) | formatting (`&` placement) only |
| `tests/test_implicit_stratified_solid.py` | file | ch14(1), ch16(1) | #379: the `work`/`fixer` arguments and deck keys dropped; the solid-mass assertions ch14 quotes are unchanged |
| `tests/test_implicit_stratified_solid.py solid_run` | L100-L142 -> L97-L139 | ch14(1), ch16(2) | as above |

## Generated fixtures

- `tools/package-pins.json`, `tools/env-pins.json`: the kintera pin stays at `0429d795` (v2.6.13). snapy at the pin sets the
  floor 2.6.14 (`CMakeLists.txt:122`, `.github/workflows/ci.yml:93`), the build and ctest records of this move ran against
  kintera 2.6.14, and appE says both; moving the kintera citations (67 ranges, byte-identical at v2.6.14 by
  `tools/kintera_pin_migration.py`) is a separate change, as the thread settled: only the snapy pin moves here.
- `gen_test_index.py --refresh <CPU build at 0d5caacb>`: 1190 registrations, 252 snapy rows (dd49df61); appC's counts
  and its refresh footnote re-run `[C1]`, `[C3]`, `[C4]`.
- `gen_deck_keys.py --refresh`: 171 keys, 155 linked (dd49df61).
- the reader glossary (`tools/reader-glossary.json`: the two retired keys), the glossary, topic, task and reader-task
  appendices, the derivation and notation indices and the reader-reference blocks were regenerated after the
  content edits (2447ca10); every `gen_*.py --check` passes on the head.
- the N6 contract generators (`gen_forcing_contract.py`, `gen_scheme_contract.py`, `gen_scheme_native.py`,
  `gen_species_contract.py`) carry `PIN = 0d5caacb...` and were run with `--write`; the native contract reads the
  README's new build and install lines (its patterns were rewritten) and wraps a shell line longer than the page
  with a backslash; `gen_boundary_contract.py` stays at `919936da` while ch14 is legacy.
- `src/snapy_report/ch01/figures/fig_mesh_flow.py`: the global-phase box no longer names the chemistry; the
  figure-data check already failed on the base (fig-ch06-face-order and 21 other series, a pre-existing finding),
  and this move does not touch `tools/figure-data-baseline.json`.
- `src/snapy_report/ch05/run_oneoverr_remainder.py`: its card no longer carries the retired keys (it would not load at
  the pin); the recorded run at `6fbb3975` keeps its record in `data/oneoverr_remainder.json`.

## Provenance tags and the ratchet

`tools/check_provenance.py` requires a tag's sha to be the pin or later, so every run tag at `73c45a7c` now counts
against its chapter: 77 problems at the base -> 198 at the head (`check_provenance.py --update`). The ratchet's
`offpin` count rises the same way, 225 -> 458 at the head (`ratchet_gate.py update`; the earlier figures of this
branch, 585 and 499, were measured before the how-to contracts were regenerated at the pin and before the #357
merge, and are superseded). Both tools ask for exactly this update in a pin move, and
both checks pass against `a71bfa1b` afterwards. New tags at the pin: ch06 `_cell`/`_fixer` and ch15
`_options`/`_output` (the replacement ctests, run under the node's ctest lock: 98ab72e5), appE's build record
(f41a1c7c), appC's refresh (dd49df61), and ch07 `_framework`'s port tag `[^fw-run]` (`framework_check.py` re-run,
byte-identical to the committed output; `tools/tests/test_provenance.py` requires the port's tag, now defined first
in the file, at the pin). Run tags kept at `73c45a7c`: every other one, ch07's CUDA ctest record `[^fw-ctest]`
included; their owners re-run after the merge.

Run tags still off-pin at the head, per chapter, with the owner who re-runs or re-words them (from
`check_provenance.py --list`; the sha column lists the tags' shas):

| chapter | off-pin tags | shas | owner |
|---|---|---|---|
| ch01 | 14 | 73c45a7c | jihenghu |
| ch02 | 17 | 73c45a7c | chengcli |
| ch03 | 17 | 73c45a7c | chengcli |
| ch04 | 23 | 68f4e066, 6fbb3975, 73c45a7c, 919936da, c0c38593 | XinyueWang05 |
| ch05 | 13 | 68f4e066, 6fbb3975, c0c38593 | XinyueWang05 |
| ch06 | 12 | 6fbb3975, 73c45a7c, c0c38593, ccf40f95 | chengcli |
| ch07 | 11 | 73c45a7c | chengcli |
| ch08 | 18 | 73c45a7c, ccf40f95 | JiachenLiu-Atmos |
| ch09 | 15 | 73c45a7c | zoeyzyhu |
| ch10 | 2 | 73c45a7c | JiachenLiu-Atmos |
| ch12 | 12 | 68f4e066, 73c45a7c, c0c38593 | XinyueWang05 |
| ch13 | 10 | 73c45a7c | jihenghu |
| ch14 | 10 | 73c45a7c | XinyueWang05 |
| ch15 | 4 | 73c45a7c | cshsgy |
| ch16 | 13 | 6fbb3975, c0c38593, ccf40f95 | gitlinffff |

191 of the 198 counted problems are these off-pin tags; the other 7 are tags that fail for another reason
(missing fields) and were failing at the base.

The PROVENANCE `pin:` line of every fragment whose claims this move edited was set to `0d5caacb`; fragments the move
did not touch keep their line.

## Content edits by chapter (the claims that changed)

- ch01 `_mesh`, `_stage`, `_driver`, the Mesh figure: the chemistry is each block's last-stage op (#382); the overview
  and the stage map point at @sec-ch11-segments, @sec-ch11-capture, @sec-ch11-refusals (the parked ch01 pointer).
- ch02 `_aneos`: the metric raise and `test_aneos_metric`; `_pecore` code-map row (no work key).
- ch03 `_x1seams`: "under the face gravity work" instead of the retired key.
- ch04 `_shallow`: the gnomonic `yz` check fires (`gnomonic-equiangle`).
- ch05 `_oneoverr`: the card without the retired keys; the run is dated to `6fbb3975`, not "the pin".
- ch06: see Resolved by hand.
- ch07 `_framework`, `_relax`, `_csdiffusion`: all ten registry sources capture (one run), `plume-forcing` refuses;
  the seam exchange posts four slabs then takes.
- ch08 `_efstatus`: the meters are restored on a redo; `_eoslimiter`: the shallow-water return.
- ch09 `_redo`: six flags, bit 64 gone, Limits item 6 rewritten; `_exchange`, `_split`: the Mesh's chemistry order.
- ch10 `_kinetics`, `_amounts`, `_budget`: the chemistry slot, `_split_posted`, the series file restart.
- ch11 `_eagerops`, `_segments`, `_refusals`: the Mesh's chemistry segment, the sources' capture reasons, the
  pre-walk checks of `run_segments`.
- ch12 `_yaml`, `_radial-exact`, `_centroid`, `_mass-covariance`, `_wb-reference`, `_opening`: the key check replaces
  the work-form selection; the deck excerpts load at the pin.
- ch13 `_ghost`, `_cuts`: the cached plan, the erased rendezvous states, the lone-block cut.
- ch15 `_options`, `_output`, `_restart`: the input report, the renamed cases, restart combine, the shared failure.
- ch16 `_gates`, `_invariants` (legacy chapter, text only): the retired keys named as refused.
- ch17, appA, appD, preface: the retired keys; `api/pd-run` gone.

`reviews/code-anchors-reread.tsv` is the output of `tools/convert_code_links.py` (the link conversion) and is not
appended to by hand; the per-symbol table above is this move's re-read record.

## The merge of #357 (chapter 14)

`tech-report` `24513bdb` was merged in (`69d4df6c`): ch14 left the legacy list there and is checked in full at the
pin (0 unresolved); of the changed bodies it cites (`MeshImpl::forward`, `MeshBlockOptionsImpl::from_yaml`,
`HydroOptionsImpl::from_yaml`, `HydroImpl::forward`, `ImplicitHydroImpl::forward_masked`, `stage_combine`,
`exchange`, `exchange_ghost_zones`, `build_schedule`) only `MeshImpl::forward` carried a claim, the Mesh timing
paragraph of `_timing.qmd` (the chemistry in the global phase), rewritten for #382; `gen_boundary_contract.py`
carries the pin and its fragment was rewritten; the baselines were re-measured as the pin-move update against
`24513bdb` (offpin 458, provenance 198; ch14's 23 run-tag mentions of `73c45a7c` are off-pin now).

## Open

- ch16 and appB stay on the legacy list with 8 citations that resolve at neither pin (above); their owners.
- The kintera pin (v2.6.13 -> v2.6.14, the floor snapy sets at the pin): its own change.
- ch09 `redo_check.py` ports the seven-flag logic of `73c45a7c`; its `[C6]` finding is kept as a finding at that sha.
  A re-port to six flags is the chapter owner's.
- ch05 `_oneoverr`: the remainder run used the cell work form at `6fbb3975`; a re-run at the pin (face work, the only
  form) is not made here.
- ch13: `test_mesh_executor.cpp`'s new `repeated_local_exchange_reads_current_buffers_after_failure` case (#383) has
  no Tests-table row yet.
- Every run tag at `73c45a7c` (198 provenance problems, 458 off-pin mentions): chapter owners re-run or re-word after
  the merge, as in the previous move.
- The figure-data gate fails on the base and on the head alike (22 changed series of ch06 and others, 2 broken
  identity links, 28 coverage differences); not touched by this move.
