"""Check Chapter 1 stage algebra without snapy.

Formula pin: snapy@73c45a7c8c991209924e1137992352c335575062. carry_dry_source there adds the anomaly reference to its
base density when anomaly storage is on, which leaves the identity in totals unchanged; the port uses totals.
Run: python3 stage_check.py from any working directory.
C1 checks the stage-weight product w2_s * prod_{t>s} w1_t, the weight FluxRegister::end_stage
(src/hydro/flux_register.cpp) gives a stage's flux total in the step sum (S <- w1 S + w2 dt T).
C2 mirrors the upwind/shift/difference of MeshBlockImpl::stage_advance (src/mesh/meshblock_advance.cpp).
C3 checks MeshBlockImpl::carry_dry_source (src/mesh/meshblock_advance.cpp), including its zero-base fallback.
C4 ports the donor cap of the implicit tracer transfer, donor_cap_cpu and donor_cap_cuda
(src/implicit/implicit_dispatch.cpp, .cu).
These are symbolic identities and numpy formula checks, not C++ runtime tests.
All cited ranges are at the full formula pin above.
"""
import numpy as np
import sympy as sp


def check_C1_weights():
    a = sp.symbols('a0:3')
    b = sp.symbols('b0:3')
    c = sp.symbols('c0:3')
    d = sp.symbols('d0:3')
    initial = sp.Symbol('initial')
    state = initial
    for stage in range(3):
        state = a[stage]*initial + b[stage]*state + c[stage]*d[stage]
    residuals = [sp.simplify(sp.diff(state, d[s])-c[s]*sp.prod(b[s+1:])) for s in range(3)]
    ok = all(r == 0 for r in residuals)
    print(f"[C1] {'PASS' if ok else 'FAIL'} supplied-stage coefficients: residuals={residuals}")
    return ok


def check_C2_transfer():
    # Exact code operation sequence: shift ratio below, upwind by sign, shift
    # transfer above, subtract and divide by volume. Last slot is a guard.
    ratio = np.array([0.2, 0.7, 0.3, 0.9, 0.4], dtype=np.float64)
    volume = np.array([1., 2., 3., 4., 1.])
    residuals = []
    tolerances = []
    for mass in (np.array([0., 0.4, -0.2, 0.1, 0.]),
                 np.array([-0.3, 0.4, -0.2, 0.1, 0.5])):
        below = np.zeros_like(ratio)
        below[1:] = ratio[:-1]
        transfer = np.where(mass > 0., below, ratio) * mass
        above = np.zeros_like(transfer)
        above[:-1] = transfer[1:]
        increment = (transfer-above)/volume
        residuals.append(abs(np.sum(volume[:-1]*increment[:-1])-(transfer[0]-transfer[-1])))
        tolerances.append(64*np.finfo(np.float64).eps*max(1., np.sum(abs(transfer))))
    ok = all(r <= t for r,t in zip(residuals, tolerances))
    print(f"[C2] {'PASS' if ok else 'FAIL'} telescoping closed/open ends: max_residual={max(residuals):.17g}; tolerance={max(tolerances):.17g}")
    return ok


def check_C3_dry_source():
    density, tracer, weight, source = sp.symbols('density tracer weight source', nonzero=True)
    residual = sp.factor((tracer+weight*source*tracer/density)/(density+weight*source)-tracer/density)
    base_density = np.array([2., 0., 2.])
    base_tracer = np.array([0.6, 0., 0.6])
    entry = np.array([0.8, 0.8, 0.8])
    dry = np.array([-0.5, -0.5, 0.5])
    # np.divide(where=...) expresses the selected quotient without emitting an
    # irrelevant zero-division warning; torch::where selection is the claim.
    removal = np.divide(base_tracer, base_density, out=entry.copy(), where=base_density != 0)
    carry = np.where(dry < 0., removal, entry)
    got = carry*dry
    expected = np.array([-0.15, -0.4, 0.4])
    error = np.max(abs(got-expected))
    tolerance = 64*np.finfo(np.float64).eps
    ok = residual == 0 and error <= tolerance
    print(f"[C3] {'PASS' if ok else 'FAIL'} dry-source ratio and branches: symbolic_residual={residual}; max_error={error:.17g}; tolerance={tolerance:.17g}")
    return ok


def donor_cap(P, held, skip_test=True):
    """donor_cap_cpu on one column; P[i] is the transfer through the face between cells i-1 and i."""
    P = P.copy()
    n = len(held)
    if skip_test:  # the CPU pre-test: sweep only when some face would bind
        cap = held[:-1].copy()
        cap[1:] += P[1:-1]
        if not (np.any(P[1:] > cap) or np.any(P[1:] < -held[1:])):
            return P
    avail = held[0]
    for i in range(1, n):
        P[i] = max(min(P[i], avail), -held[i])
        avail = held[i] + P[i]
    return P


def check_C4_donor_cap():
    rng = np.random.default_rng(4)
    worst_neg, worst_sum, same, binds = 0., 0., True, 0
    for _ in range(2000):
        n = 12
        held = rng.uniform(0., 1., n) * (rng.uniform(size=n) > 0.3)
        P = np.r_[0., rng.normal(0., 1., n - 1)]
        Q = donor_cap(P, held)
        binds += not np.array_equal(Q, P)
        same &= np.array_equal(Q, donor_cap(P, held, skip_test=False))
        after = held + Q - np.r_[Q[1:], 0.]
        worst_neg = max(worst_neg, -after.min() / max(held.max(), 1e-300))
        worst_sum = max(worst_sum, abs(after.sum() - held.sum()) / held.sum())
    # a uniform ratio r through a dry transfer that obeys the same cap: the tracer cap never binds
    worst_ratio = 0.
    for _ in range(2000):
        hd = rng.uniform(0.1, 1., 12)
        M = donor_cap(np.r_[0., rng.normal(0., 3., 11)], hd, skip_test=False)
        r = rng.uniform(0.1, 2.)
        Q = donor_cap(r * M, r * hd)
        worst_ratio = max(worst_ratio, np.abs(Q - r * M).max() / np.abs(r * M).max())
    eps = np.finfo(np.float64).eps
    ok = worst_neg <= 4 * eps and worst_sum <= 64 * eps and same and binds > 0 and worst_ratio <= 4 * eps
    print(f"[C4] {'PASS' if ok else 'FAIL'} donor cap: no cell below zero, column conserved, pre-test exact, "
          f"uniform ratio unchanged: columns_capped={binds}/2000; deficit/max_held={worst_neg:.3g}; "
          f"column_change={worst_sum:.3g}; skip_equal={same}; uniform_ratio_change={worst_ratio:.3g}")
    return ok


def main():
    return sum(not check() for check in (check_C1_weights, check_C2_transfer, check_C3_dry_source,
                                         check_C4_donor_cap))


if __name__ == '__main__':
    raise SystemExit(main())
