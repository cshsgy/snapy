#236 arm A: moist-mixture species carry their enthalpy and momentum

At a limited face, arm A withholds species mass together with its energy and its momentum. For moist-mixture the specific enthalpy is `u_n + z_n R_n T` for a vapour, plus kinetic energy. `u_n` and `z_n` come from kintera (`eval_intEng_R` / `eval_czh` in `MoistMixtureImpl::species_enthalpy`), then `Rgas * inv_mu`. It is not a hard-coded linear formula. Ideal-moist is unchanged and stays bitwise equal to main on every row and face.

The code commit is the squash of `5281c51fd272662c8625142c01c29584af42afb7` and `562d4979b92e4f48d5bd1a5043b9d9e81068ef75` onto main `5eeb9b6761ae484a98b3993aae18314fc58cf862`. Its tree matches the second of those. The test `flux_positivity.moist_mixture_withheld_mass_keeps_its_energy_and_momentum` pins the carry. On main the same column was pinned the other way (`moist_mixture_withholds_no_energy_or_momentum_yet`), so the new test fails there and passes here. A second test checks the NASA-9 / H2 species sum against internal energy plus pressure (relative tolerance `1e-9`; the column residual is `1.4e-16`). Neither test uses `debug_disable_flux_positivity`. That knob stays out. An unlimited reference is `limiter: false`.

Study artefacts stay on `cshsgy/snapy` `fix/236-gate2-arm-a`, by commit, not by branch name:

- rescore, 16 cards, species-only census: `03b16e17c7dbf288d35520b398273de2f27b5f34`
- uranus 500 and `study/236/gate2/RESULTS.md`: `819334e41102adf9b9edb45dc7c476b5736fc411`

Gate 2, CPU, fp64, kintera 2.5.13, `use_nasa9_cp=0`, `use_h2_cp=0`, `Rgas=8.31446`. Species census is 0 on all 16 cards. O3, T, I1, and I2 pass. Worst O3 enthalpy ratio `9.42e-4`, worst `|dT|` `1.14e-13` K, worst I1 `2.75e-5`. I2 is bitwise. Limited-face counts match the Gate 0 list: adv 5, settling 5, x2 30, donors 4, mixed 5, vapor-u0 5. IDN flux is bitwise equal to the unlimited flux. Worst energy ratio `1.21e-4`, worst momentum ratio `1.30e-4`, worst column ratio `4.73e-5`. The five moist-mixture mutations each turn O3 and the energy carry red by at least `1e3` times the tolerance (smallest O3 ratio `9.04e10`, mutation 4; smallest energy ratio `1.22e10`). B2's flux is bitwise equal to `limiter: false` on all 16 (`max abs 0`).

Uranus, 500 cycles, same seeded driver, no redo and no abort. The effect is small. max `|dT|` of arm A minus main is `1.232e-9` K. `positivity_hits` is 484761 against main 484774. Total energy differs by 768 J, relative `5.09e-16`.

Not in this PR:

- `z != 1` is not done.
- Arm G is not done. Its positivity proof needs CFL at most 0.5, and the vertical direction is implicit.
- Dry IDN below `1e-10` at `dt = 1` is outside #236. Unmodified main shows the same minima.
- `equilibrate_tp did not converge after 5 iterations`, 9682 times on both main and arm A in the uranus run. Same count on both, so it is not an arm A defect, and it does not belong in this PR.
