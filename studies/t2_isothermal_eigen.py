"""Independent isothermal acoustic-gravity eigen solver for snapy #250 T2.

Chebyshev collocation of the linearised compressible Euler equations.
The check is the rigid-lid isothermal dispersion relation, not snapy.
"""

import numpy as np
from numpy.linalg import eig


def chebyshev_diff_matrix(n):
    """Gauss-Lobatto nodes and derivative matrix on [-1, 1]. n is the polynomial degree."""
    if n < 2:
        raise ValueError("n >= 2")
    j = np.arange(n + 1)
    x = np.cos(np.pi * j / n)
    c = np.ones(n + 1)
    c[0] = c[-1] = 2.0
    c = c * ((-1.0) ** j)
    X = np.subtract.outer(x, x)
    D = np.outer(c, 1.0 / c) / (X + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return x, D


def isothermal_omegas(cs, H, kx, kz, gamma, g):
    """Analytic roots of the isothermal rigid-lid dispersion relation. Positive omega."""
    n2 = (gamma - 1.0) / gamma * g / H
    a = 1.0
    b = -(cs**2) * (kx**2 + kz**2 + 1.0 / (4.0 * H**2))
    c = n2 * (cs**2) * (kx**2)
    # omega^4 + b omega^2 + c = 0
    disc = b**2 - 4.0 * a * c
    roots = []
    for sign in (+1.0, -1.0):
        w2 = (-b + sign * np.sqrt(disc)) / 2.0
        if w2 > 0:
            roots.append(np.sqrt(w2))
    return sorted(roots)


def solve_column(n, L, H, gamma, g, kx):
    """Eigenvalues lambda = -i omega of one Fourier mode. Rigid lids, w = 0."""
    x, D = chebyshev_diff_matrix(n)
    # z = 0 at the bottom (x = 1) and z = L at the top (x = -1)
    z = 0.5 * L * (1.0 - x)
    Dz = (-2.0 / L) * D
    rho0 = np.exp(-z / H)
    cs2 = gamma * g * H
    m = n + 1
    # state: u, w, p, rho  each length m
    N = 4 * m
    A = np.zeros((N, N), dtype=complex)

    def sl(var, i):
        return var * m + i

    for i in range(m):
        # du/dt = -i kx p / rho0
        A[sl(0, i), sl(2, i)] = -1j * kx / rho0[i]
        # dw/dt = -1/rho0 dp/dz - g rho/rho0
        for j in range(m):
            A[sl(1, i), sl(2, j)] = -Dz[i, j] / rho0[i]
        A[sl(1, i), sl(3, i)] = -g / rho0[i]
        # drho/dt = -i kx rho0 u + (rho0/H) w - rho0 dw/dz
        A[sl(3, i), sl(0, i)] = -1j * kx * rho0[i]
        A[sl(3, i), sl(1, i)] = rho0[i] / H
        for j in range(m):
            A[sl(3, i), sl(1, j)] += -rho0[i] * Dz[i, j]
        # dp/dt = -i kx rho0 cs^2 u - rho0 cs^2 dw/dz + rho0 g w
        A[sl(2, i), sl(0, i)] = -1j * kx * rho0[i] * cs2
        A[sl(2, i), sl(1, i)] = rho0[i] * g
        for j in range(m):
            A[sl(2, i), sl(2, j)] += 0.0
            A[sl(2, i), sl(1, j)] += -rho0[i] * cs2 * Dz[i, j]

    # Rigid lid: replace the w-rows at the two endpoints with w = 0,
    # enforced as dw/dt = -w so the eigenvalue of a constrained mode is not zero
    # from a free w. The row becomes lambda w = -w, i.e. those modes sit at
    # lambda = -1 and are discarded. Interior w equations stay.
    for i in (0, m - 1):
        A[sl(1, i), :] = 0.0
        A[sl(1, i), sl(1, i)] = -1.0

    lam = eig(A)[0]
    # physical frequencies: lambda = -i omega, omega real and not the lid constraint
    omega = []
    for value in lam:
        if abs(value + 1.0) < 1e-8:
            continue
        if abs(value.real) > 1e-6 * max(1.0, abs(value.imag)):
            continue  # growing/damped: not a linear mode of this conservative system
        w = abs(value.imag)
        if w > 1e-8:
            omega.append(w)
    return sorted(omega)


def main():
    gamma = 1.4
    g = 1.0
    H = 1.0
    cs = np.sqrt(gamma * g * H)
    L = 4.0 * H  # four scale heights, rigid lids
    n = 48
    cases = []
    print(f"cs={cs:.6f} N2={(gamma-1)/gamma * g/H:.6f} L/H={L/H:.1f} N={n}")
    worst = 0.0
    for kx in (0.0, 0.5, 1.0, 2.0):
        numerical = solve_column(n, L, H, gamma, g, kx)
        for mode in (1, 2, 3):
            kz = mode * np.pi / L
            analytic = isothermal_omegas(cs, H, kx, kz, gamma, g)
            # match each analytic root to the nearest numerical frequency
            for wa in analytic:
                if not numerical:
                    err = np.inf
                else:
                    wn = min(numerical, key=lambda w: abs(w - wa))
                    err = abs(wn - wa) / wa
                worst = max(worst, err)
                cases.append((kx, mode, wa, err))
                print(
                    f"kx={kx:.2f} n={mode} omega={wa:.6e} rel={err:.3e}"
                )
    print(f"worst relative frequency error {worst:.3e}")
    if worst > 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
