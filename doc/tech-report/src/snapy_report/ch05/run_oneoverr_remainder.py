"""The 1/R content of the one-step effective stratification on a spherical-polar column (chapter 5, _oneoverr.qmd).

The closure run behind fig_oneoverr_remainder. It is the spherical-polar counterpart of snapy's Cartesian one-step
harness, tests/test_horizontal_flux_covariance.py: an isentropic column (T0 = 1 - z/Cp, R = g = 1, gamma = 1.4,
depth H = 1) on x1 = r in [R, R + H], seeded with a small convective roll. One RK3 step of the seeded column minus
one of the unseeded column gives the entropy tendency S, which is projected on the seeded w at the roll's
wavenumber:
    eps_eff = Re sum_i rho0 T0 (S^ conj(w^))_i / (Cp sum_i rho0 |w^_i|^2),
with unit weight per x1 cell (the harness's metric). The physical value is 0. The quantity of the figure is the
1/R content
    D(R) = R [eps_eff n1^2 (R) - eps_eff n1^2 (inf)],
with eps_eff n1^2 (inf) from the same column in Cartesian geometry, run in the same arm (the switch implies
SNAP_WB_REF4, which acts in Cartesian too). An O(h^2/R) term of the scheme gives a D that is constant in R and in n1.

Setup, chosen so that R -> inf is the Cartesian column exactly:
  - the roll is in x3 = phi at the equator: one theta cell centred on pi/2, 2 n1 cells in phi over an arc of
    length LX = 2 sqrt(2) at r = R, so the horizontal wavenumber is m = k R with k = 2 pi / LX;
  - the seed is anelastic on the sphere, rho w = k sin(pi z) (R/r)^2 cos(m phi) and
    rho u_phi = -pi cos(pi z) (R/r) sin(m phi), largest |w| 1e-4 by default (--amp); in Cartesian it is the roll of the
    Cartesian harness;
  - rho, p and the momenta are initialised as exact cell averages in x1 (six-point Gauss-Legendre, weight r^2 on
    the sphere, 1 in Cartesian) and as point values in the horizontal; the velocity is <rho w> / <rho>;
  - WENO5 in its smooth limit (scale, no shock flag), LMARS, RK3, the face gravity work (the only form at the pin;
    the recorded run at 6fbb3975 set the cell form, fixer off), SNAP_FLUX_COVARIANCE=1
    (the x2/x3 face-flux covariance of #289/#293), every other SNAP_* switch unset;
  - one step of dt = DT_CFL / n1, the same for every R and for the Cartesian column.
Arms: SNAP_X1_CENTROID_EXACT unset ("off") and =1 ("on"). Every switch is read once per process, so each arm runs
in its own child process, which also runs that arm's Cartesian reference.

Run with the snapy build to be measured in the active Python, from anywhere:
    python3 run_oneoverr_remainder.py --out oneoverr_remainder.json [--rh 5 20 160 1000] [--nz 64 128]
The JSON records the snapy version string and, with --snapy-sha, the commit it was built from; nothing in the
script depends on the commit.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

GAMMA, CP, CV, NG = 1.4, 3.5, 2.5, 3
IV1, IV3 = 1, 3
LX = 2.0 * math.sqrt(2.0)
H = 1.0
AMP = 1.0e-4
AMP_SPREAD = (0.9, 1.0, 1.1)  # the linear response does not depend on them: their spread is the round-off
DT_CFL = 0.2  # dt = DT_CFL / n1; the sound speed is at most sqrt(1.4), so the CFL number is <= 0.24
GLX, GLW = np.polynomial.legendre.leggauss(6)
RH = (5.0, 10.0, 20.0, 40.0, 160.0, 1000.0)
NZ = (64, 128)
ARMS = {"off": None, "on": "1"}
SWITCHES = ("SNAP_X1_CENTROID_EXACT", "SNAP_FLUX_COVARIANCE", "SNAP_WB_REF4", "SNAP_X1_MASS_COVARIANCE",
            "SNAP_GRAVITY_WORK_RADIAL_EXACT")


def config(nz, radius):
    """The card: spherical-polar column at radius R, or the Cartesian column when radius is None."""
    if radius is None:
        geometry = {"type": "cartesian",
                    "bounds": {"x1min": 0.0, "x1max": H, "x2min": 0.0, "x2max": 1.0, "x3min": 0.0, "x3max": LX},
                    "cells": {"nx1": nz, "nx2": 1, "nx3": 2 * nz, "nghost": NG}}
        bc2 = "periodic"
    else:
        half = 0.5 * H / nz / radius
        geometry = {"type": "spherical-polar",
                    "bounds": {"x1min": radius, "x1max": radius + H,
                               "x2min": 0.5 * math.pi - half, "x2max": 0.5 * math.pi + half,
                               "x3min": 0.0, "x3max": LX / radius},
                    "cells": {"nx1": nz, "nx2": 1, "nx3": 2 * nz, "nghost": NG}}
        bc2 = "reflecting"
    return {
        "geometry": geometry,
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA, "weight": 8.31446,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-12, "limiter": True},
            "reconstruct": {"vertical": {"type": "weno5", "scale": True, "shock": False},
                            "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {"x1-inner": "reflecting", "x1-outer": "reflecting",
                                            "x2-inner": bc2, "x2-outer": bc2,
                                            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -1.0}},
    }


def cell_means(x1f, x1min, radius):
    """x1 cell averages (weight r^2 on the sphere, 1 in Cartesian) of rho, p and of the seed's x1 profiles.

    Returns rho, p, a and b, with rho w = k a cos(k x) and rho u = -pi b sin(k x)."""
    c, h = 0.5 * (x1f[1:] + x1f[:-1]), 0.5 * (x1f[1:] - x1f[:-1])
    r = c[:, None] + h[:, None] * GLX[None, :]
    wt = GLW[None, :] * (r * r if radius is not None else np.ones_like(r))
    z = r - x1min
    T0 = 1.0 - z / CP
    p = T0 ** CP
    rho = p / T0
    if radius is None:
        a, b = np.sin(math.pi * z), np.cos(math.pi * z)
    else:
        a, b = np.sin(math.pi * z) * (radius / r) ** 2, np.cos(math.pi * z) * (radius / r)
    mean = lambda q: (q * wt).sum(1) / wt.sum(1)  # noqa: E731
    return mean(rho), mean(p), mean(a), mean(b)


def build(nz, radius, seed, amp=AMP):
    import torch
    import yaml
    from snapy import MeshBlock, MeshBlockOptions, kIDN, kIPR

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(nz, radius), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device("cpu"), torch.float64)
    x1f = block.buffer("coord.x1f").cpu().numpy()
    x3v = block.buffer("coord.x3v").cpu().numpy()
    x1min = radius if radius is not None else 0.0
    k = 2.0 * math.pi / LX
    rho, p, a, b = cell_means(x1f, x1min, radius)
    x = x3v * radius if radius is not None else x3v  # arc length at r = R
    w = block.buffer("hydro.D").clone().zero_()
    t = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(w)  # noqa: E731
    w[kIDN] = t(rho)[None, None, :]
    w[kIPR] = t(p)[None, None, :]
    if seed:
        I = slice(NG, -NG)
        rw = k * a[None, :] * np.cos(k * x[I])[:, None]
        ru = -math.pi * b[None, :] * np.sin(k * x[I])[:, None]
        scale = amp / float(np.abs(rw / rho[None, :]).max())
        w[IV1, I, 0, :] = t(scale * rw / rho[None, :])
        w[IV3, I, 0, :] = t(scale * ru / rho[None, :])
    bv, _ = block.initialize({"hydro_w": w})
    return block, bv, x


def eps_eff_nz2(nz, radius, amp=AMP):
    """eps_eff n1^2 of one step of the column at radius R (None: Cartesian)."""
    import torch
    from snapy import kIDN, kIPR

    b0, bv0, x = build(nz, radius, False)
    b1, bv1, _ = build(nz, radius, True, amp)
    eos = b1.module("hydro.eos")
    dt = DT_CFL / nz
    I = slice(NG, -NG)

    def prim(bv):
        return eos.compute("U->W", [bv["hydro_u"]]).cpu()

    def s_of(W):
        return CV * torch.log(W[kIPR, I, 0, I]) - CP * torch.log(W[kIDN, I, 0, I])

    W0, W1 = prim(bv0), prim(bv1)
    for block, bv in ((b0, bv0), (b1, bv1)):
        block.inc_cycle()
        for st in range(len(block.intg.stages)):
            block.forward(bv, dt, st)
        assert block.check_redo(bv) == 0
    S = ((s_of(prim(bv1)) - s_of(W1)) - (s_of(prim(bv0)) - s_of(W0))) / dt
    k = 2.0 * math.pi / LX
    emk = torch.exp(-1j * k * torch.from_numpy(x[I]).to(torch.complex128))[:, None]
    nx = emk.shape[0]
    rho0 = W0[kIDN, NG, 0, I]
    T0 = W0[kIPR, NG, 0, I] / rho0
    what = (2.0 / nx) * (W1[IV1, I, 0, I] * emk).sum(0)
    Shat = (2.0 / nx) * (S * emk).sum(0)
    num = (rho0 * (T0 * Shat * what.conj()).real).sum()
    den = (CP * rho0 * what.abs() ** 2).sum()
    return float(num / den) * nz ** 2


def child(out, rh, nzs, amp):
    res = {}
    for f in AMP_SPREAD:
        res[f"{f:g}"] = {"cartesian": {str(nz): eps_eff_nz2(nz, None, f * amp) for nz in nzs},
                         "spherical": {str(nz): {f"{r:g}": eps_eff_nz2(nz, r * H, f * amp) for r in rh}
                                       for nz in nzs}}
    json.dump(res, open(out, "w"))


def run_child(tmp, name, env_value, rh, nzs, amp):
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env["SNAP_FLUX_COVARIANCE"] = "1"
    if env_value is not None:
        env["SNAP_X1_CENTROID_EXACT"] = env_value
    out = os.path.join(tmp, name + ".json")
    cmd = [sys.executable, os.path.abspath(__file__), "--child", out, "--rh", *map(str, rh),
           "--nz", *map(str, nzs), "--amp", repr(amp)]
    subprocess.run(cmd, env=env, check=True)
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rh", type=float, nargs="+", default=RH, help="R/H values")
    ap.add_argument("--nz", type=int, nargs="+", default=NZ, help="x1 cell counts n1")
    ap.add_argument("--amp", type=float, default=AMP, help="seed amplitude, the largest |w| of the seed; it is "
                    "also run at 0.9 and 1.1 times this, and half their spread is recorded as the noise")
    ap.add_argument("--out", help="JSON file to write")
    ap.add_argument("--snapy-sha", default=None, help="commit the snapy build was made from (recorded only)")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        child(a.child, a.rh, a.nz, a.amp)
        return 0
    from importlib.metadata import version
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmp:
        arms = {name: run_child(tmp, name, value, a.rh, a.nz, a.amp) for name, value in ARMS.items()}
    def remainder(arm, nz):
        return [r * H * (arm["spherical"][nz][f"{r:g}"] - arm["cartesian"][nz]) for r in a.rh]

    d, noise = {}, {}
    for name, arm in arms.items():
        d[name] = {nz: remainder(arm["1"], nz) for nz in map(str, a.nz)}
        spread = {nz: np.array([remainder(arm[f"{f:g}"], nz) for f in AMP_SPREAD]) for nz in map(str, a.nz)}
        noise[name] = {nz: (0.5 * (v.max(0) - v.min(0))).tolist() for nz, v in spread.items()}
    result = {
        "provenance": {
            "snapy_version": version("snapy"),
            "sha": a.snapy_sha,
            "torch_version": version("torch"),
            "kintera_version": version("kintera"),
            "run": "python3 run_oneoverr_remainder.py " + " ".join(sys.argv[1:]),
            "deck": "src/snapy_report/ch05/run_oneoverr_remainder.py (isentropic column T0 = 1 - z/Cp, R = g = 1, "
                    "gamma 1.4, H = 1, anelastic roll in phi at the equator, max |w| %g, x1 cell-average IC, "
                    "WENO5 smooth, LMARS, RK3 one step dt = %g / n1, face gravity work, "
                    "SNAP_FLUX_COVARIANCE=1)"
                    % (a.amp, DT_CFL),
            "arms": "SNAP_X1_CENTROID_EXACT unset (off) and =1 (on), each with its own Cartesian reference (same card, "
                    "same switches)",
            "device": "cpu", "precision": "double", "ranks": 1,
        },
        "quantity": "R/H [eps_eff n1^2 (R) - eps_eff n1^2 (cartesian, same arm)]",
        "rh": list(a.rh),
        "nz": list(a.nz),
        "amp_factors": list(AMP_SPREAD),
        "eps_nz2": {name: arm["1"] for name, arm in arms.items()},
        "remainder": d,
        "remainder_noise": noise,
    }
    for nz in map(str, a.nz):
        for name, arm in arms.items():
            print(f"n1 {nz:4s} {name:3s} cartesian eps n1^2 {arm['1']['cartesian'][nz]:+.6f}  R/H "
                  + "  ".join(f"{r:g}: {v:+.5f}({n:.0e})" for r, v, n in zip(a.rh, d[name][nz], noise[name][nz])))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
