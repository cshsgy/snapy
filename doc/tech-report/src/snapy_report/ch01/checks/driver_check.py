"""Executable check of the example driver's step loop (chapter 1, _driver.qmd).

What is checked: the time loop of main in examples/run_hydro.cpp, the step of MeshBlockImpl::global_time_step
(src/mesh/meshblock_advance.cpp, 2^-r times cfl times the bound) and the cycle and redo bookkeeping of
MeshBlockImpl::apply_redo (src/mesh/meshblock_redo.cpp) at snapy@73c45a7c8c991209924e1137992352c335575062, ported as a
state machine driven by a scripted sequence of check_redo results. Claims C1-C3.

Run: python3 driver_check.py   (numpy only; exits with the number of failed claims)
"""
import sys


def report(tag, ok, label, numbers):
    print("[%s] %s %s: %s" % (tag, "PASS" if ok else "FAIL", label, numbers))
    return ok


CFL, BOUND, MAX_REDO = 0.9, 2.0, 5


def run(results, tlim=100.):
    """results: what check_redo returns at each attempt (1 redo, 0 accept, -1 stop)."""
    time, cycle, redo, log = 0., 0, 0, []
    outputs = [time]                              # written once before the loop
    for res in results:
        if time >= tlim:
            break
        dt = 2. ** -redo * CFL * BOUND           # MeshBlockImpl::max_time_step
        log.append((cycle, dt))                  # print_cycle_info sees the accepted-step count
        if res > 0:                              # apply_redo: count up, state restored; the cycle is untouched
            redo += 1
            continue
        if res < 0:
            break
        redo = 0                                 # acceptance resets the count
        cycle += 1                               # run_hydro.cpp main: cycle and clock advance only after
        time += dt                               # an accepted step
        outputs.append(time)
    return time, cycle, log, outputs


def check_C1_clock_moves_only_on_acceptance():
    t, c, log, out = run([0, 1, 1, 0, 0])
    want = 1.8 + 0.45 + 1.8
    ok = abs(t - want) < 1e-15 and c == 3 and len(out) == 4
    return report("C1", ok, "a rejected step leaves the clock and the cycle count where they were; outputs follow only "
                  "accepted steps", "time %.2f (= 1.8 + 0.45 + 1.8), cycle %d, %d outputs" % (t, c, len(out)))


def check_C2_dt_halves_per_redo():
    _, _, log, _ = run([1, 1, 1, 0])
    dts = [d for _, d in log]
    ok = dts == [1.8, 0.9, 0.45, 0.225]
    return report("C2", ok, "each redo halves the step of the retry", "dt per attempt %s" % dts)


def check_C3_negative_return_stops():
    t, c, log, out = run([0, 1, -1, 0, 0])
    ok = abs(t - 1.8) < 1e-15 and len(log) == 3 and len(out) == 2
    return report("C3", ok, "a negative check_redo ends the loop at the time of the last accepted step",
                  "time %.2f after %d attempts, %d outputs" % (t, len(log), len(out)))


def main():
    checks = [check_C1_clock_moves_only_on_acceptance, check_C2_dt_halves_per_redo, check_C3_negative_return_stops]
    sys.exit(sum(not c() for c in checks))


if __name__ == "__main__":
    main()
