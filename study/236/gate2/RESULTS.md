# Gate 2 arm A, CPU — cards PASS under the rescore

The 16 dumps were not rerun. The bar change is Xi's rescore, not a plan revision: arm A's census counts species rows only and must be 0; IDN is reported beside unmodified main; B2 is valid when it is bitwise equal to `limiter: false`; the x2 repair throw is noted and is not an arm A defect. Uranus, 500 cycles, is below.

Base: snapy main `5eeb9b6761ae484a98b3993aae18314fc58cf862`. Port of `8c6b3fc` / `842a116` is `5281c51fd272662c8625142c01c29584af42afb7` plus `562d4979b92e4f48d5bd1a5043b9d9e81068ef75`. The card record is `be5b3378a44d00730a1c15d5b048d0842bd518d5`. CPU, fp64, one process per card, dt = 1. kintera 2.5.13. `Rgas = 8.31446`. `use_nasa9_cp = 0`, `use_h2_cp = 0` on every card.

## Cards

O3 (vapor and cloud), T, I1, and I2: PASS on all 16. Worst O3 enthalpy ratio is `9.42e-4` (moist-mixture donors, vapor). Worst `|dT|` is `1.14e-13` K (ideal-moist donors). Worst I1 ratio is `2.75e-5`. I2 neighbor is bitwise on every card, and `|h-KE| / S_n = 0`.

`M_n` versus the YAML composition: relative `1.20e-16`. Not VOID.

Limited-face counts match the list on every card (adv 5, settling 5, x2 30, donors 4, mixed 5, vapor-u0 5). IDN flux is bitwise A versus B2 on every card. Species carry `R_s` ratio is 0. Worst energy ratio `1.21e-4`, worst momentum ratio `1.30e-4`, worst column ratio `4.73e-5`. All of those are PASS. B2 no longer voids them: its flux is bitwise equal to this build's `limiter: false` arm on all 16 (`max abs 0`), and a negative drained cell is what that reference is for.

Arm A species census is 0 on all 16 (`species_only` 0 and `both` 0). Minimum species density is `9.095e-15` on the uniform cards and `1.819e-14` on both settling cards.

IDN is not an arm A clamp. On the 14 non-settling cards the old census was `IDN < 1e-10` only. Side by side with unmodified main `5eeb9b6`, shipped limiter, same cards, every minimum and every interior count matches to 17 digits (the control already posted). Equal, so that comparison passes. The dry undershoot is the dt = 1 cards.

x2, both EOS: B2's repair throws `Failed to fix vapor mass fractions` after the census is stored. `fix_vapor` only repairs an x1 column, on main as well. Arm A does not throw. Noted, not a failure.

Moist-mixture mutations 1–5, all 8 cards: each turns O3 and the energy carry red by at least `1e3` times the tolerance. Smallest O3 ratio is `9.04e10` (mutation 4). Smallest energy ratio is `1.22e10`.

## Uranus

`examples/uranus.yaml` unchanged (`nlim: -1`). The run file is `study/236/gate2/uranus_nlim500.yaml`: the same input with `nlim: 500` and `ncycle_out: 1`. Both processes exited 0 on the cycle limit, `cycle=500`, `time=2783.84` (under `tlim=1e5`). `check_redo` did not redo and did not abort. No CUDA: `CUDA_VISIBLE_DEVICES` was empty. kintera 2.5.13. One thread.

The driver is `examples/run_hydro.cpp` in this commit, and the same file compiled against unmodified main `5eeb9b6` (that library has no #266 test diff in `src/`). `torch::manual_seed(236500)` is called immediately before the two `rand_like` draws. `positivity_hits` is the hydro accumulator (interior `theta<1`, never reset). Energy is the interior sum of `u[IPR]*V`, the same reduction as `print_cycle_info`. Temperature is interior `W->T` after `U->W` on a clone of the conserved state, except cycle 0, which uses the initial primitive. `ncell=20000`.

Cycle 0 is bitwise the same state. max `|dT|` (A minus main) is 0 on all 20000 cells. `positivity_hits=0`. Energy `1.5102600646786102e+18`.

After 500 successful forwards:

| | main `5eeb9b6` | arm A |
| --- | --- | --- |
| time | `2783.8358967327667` | `2783.835896732764` |
| positivity_hits | `484774` | `484761` |
| energy | `1.5098307230627448e+18` | `1.5098307230627456e+18` |

max `|dT|` at cycle 500 is `1.2320242603891529e-09` K (19821 of 20000 cells differ; median `|dT|` `1.251e-12` K). Energy differs by 768 J, relative `5.09e-16`. Arm A has 13 fewer positivity hits. Neither run is a crash, and there is no pass/fail bar on these three numbers in the ask, so this is the report, not a plan change.

Both logs print `equilibrate_tp did not converge after 5 iterations` 9682 times. The count matches, and the integrator still reached cycle 500. No PR.
