#!/usr/bin/env python3
"""Rebuilt #348 bitwise harness for the vertical implicit solver (VIC).

The harness that chengcli/snapy#348 describes was never committed. This is a
rebuild from that description. It is a gate tool, not part of any PR.

16 decks = VIC matrix {full (implicit-scheme 9), partial (1)}
         x gravity work {face, cell}
         x geometry {cartesian, stretched cartesian (x1rat 1.05),
                     spherical-polar, gnomonic-equiangle}
All decks use weno5, lmars, rk3, reflecting x1 walls, ideal gas and FP64.
Each deck starts from a perturbed hydrostatic column and runs NSTEP steps
at a fixed dt. dt gives a vertical acoustic Courant number of about 20, so
the VIC carries the step. A redo at any step is an error. hydro_u is saved
after every step.

usage:
  h348.py --device cpu|cuda --out DIR      run the 16 decks, write DIR/<deck>.pt
                                           and DIR/manifest.json (sha256 of each
                                           dump's bytes)
  h348.py --compare DIR_A DIR_B            byte compare two runs; also prints the
                                           per-step energy-row relative difference
                                           max|E_B - E_A| / max|E_A| (I-3 target)
  h348.py --check-manifest DIR MANIFEST    compare DIR's checksums with a stored
                                           manifest (e.g. results/main-*.json)
  h348.py --write-decks DIR                write the 16 input YAML decks
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import tempfile
import time

import torch
import yaml

GRAV, GAMMA, RD, T0, PS = 23.1, 1.403461, 3515.0, 786.0, 1.0e7
NZ, DZ, NX2 = 45, 29946.8085106, 8
RADIUS = 7.e7
CS = math.sqrt(GAMMA * RD * T0)
DT = 20.0 * DZ / CS  # vertical acoustic Courant ~20: the VIC carries the step
NGHOST = 3
IPR = 4  # energy row of hydro_u for ideal gas (IDN, IVX, IVY, IVZ, IPR)

SCHEMES = {"full": 9, "partial": 1}
WORKS = ("face", "cell")
GEOMS = ("cartesian", "stretched", "spherical-polar", "gnomonic-equiangle")


def decks():
    for s, gw, g in itertools.product(SCHEMES, WORKS, GEOMS):
        yield f"{s}_{gw}_{g}", SCHEMES[s], gw, g


def config(scheme, work, geometry):
    rgas = 8.31446261815324
    cfg = {
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0.0, "x1max": NZ * DZ, "x2min": 0.0,
                                "x2max": 31415926.53589793, "x3min": 0.0,
                                "x3max": 3926990.8169872416},
                     "cells": {"nx1": NZ, "nx2": NX2, "nx3": 1,
                               "nghost": NGHOST}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": GAMMA,
                                  "weight": rgas / RD,
                                  "density-floor": 3.4682e-18,
                                  "pressure-floor": 3.4682e-18,
                                  "temperature-floor": 1.e-6, "limiter": True},
            "reconstruct": {
                "vertical": {"type": "weno5", "scale": True, "shock": False},
                "horizontal": {"type": "weno5", "scale": True, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.5, "implicit-scheme": scheme,
                        "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -GRAV, "gravity-work": work}},
    }
    if work == "face":
        cfg["forcing"]["const-gravity"]["gravity-work-fixer"] = False
    # cell: the fixer key is omitted, so the shipped default (fixer on) runs
    if geometry == "stretched":
        cfg["geometry"]["cells"]["x1rat"] = 1.05
    elif geometry == "spherical-polar":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"]["nx2"] = 1
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min": math.pi / 2 - 0.2, "x2max": math.pi / 2 + 0.2,
            "x3min": 0.0, "x3max": 0.05}
    elif geometry == "gnomonic-equiangle":
        cfg["geometry"]["type"] = geometry
        cfg["geometry"]["cells"].update(nx2=4, nx3=4)
        cfg["geometry"]["bounds"] = {
            "x1min": RADIUS, "x1max": RADIUS + NZ * DZ,
            "x2min_pi": -0.25, "x2max_pi": 0.25,
            "x3min_pi": -0.25, "x3max_pi": 0.25}
        for side in ("x2-inner", "x2-outer", "x3-inner", "x3-outer"):
            cfg["boundary-condition"]["external"][side] = "reflecting"
    return cfg


def initial_w(block):
    """Isothermal hydrostatic p(z) at the cell centres, density times
    (1 + 1e-3 sin(0.7 i + 1.3 j + 2.1 k)) over the whole array, ghosts included."""
    from snapy import kIDN, kIPR
    bufs = dict(block.named_buffers())
    x1v = bufs["coord.x1v"].cpu()
    z = x1v - x1v[NGHOST] + 0.5 * (x1v[NGHOST + 1] - x1v[NGHOST])
    p = PS * torch.exp(-GRAV * z / (RD * T0))
    w = bufs["hydro.D"].clone().zero_()
    shape = w.shape[1:]
    k = torch.arange(shape[0]).view(-1, 1, 1)
    j = torch.arange(shape[1]).view(1, -1, 1)
    i = torch.arange(shape[2]).view(1, 1, -1)
    pert = 1.e-3 * torch.sin(0.7 * i + 1.3 * j + 2.1 * k)
    rho = (p / (RD * T0)).view(1, 1, -1).expand(shape)
    w[kIDN] = rho.to(w) * (1. + pert.to(w))
    w[kIPR] = p.view(1, 1, -1).expand(shape).to(w)
    return w


def run_deck(scheme, work, geometry, device, nstep):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(config(scheme, work, geometry), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    block_vars, _ = block.initialize({"hydro_w": initial_w(block)})
    nstage = len(block.module("intg").stages)
    out = []
    for n in range(nstep):
        for stage in range(nstage):
            block.forward(block_vars, DT, stage)
        if block.check_redo(block_vars):
            raise RuntimeError(f"redo at step {n + 1}")
        out.append(block_vars["hydro_u"].detach().cpu().clone())
    return torch.stack(out)  # [nstep, nvar, nc3, nc2, nc1]


def sha256(u):
    return hashlib.sha256(u.contiguous().numpy().tobytes()).hexdigest()


def produce(args):
    torch.set_default_dtype(torch.float64)
    os.makedirs(args.out, exist_ok=True)
    decks_out = {}
    t0 = time.time()
    for name, scheme, work, geom in decks():
        u = run_deck(scheme, work, geom, args.device, args.nstep)
        torch.save(u, os.path.join(args.out, name + ".pt"))
        fin = bool(torch.isfinite(u).all())
        decks_out[name] = {"sha256": sha256(u), "shape": list(u.shape),
                           "finite": fin}
        print(f"{name:34s} {decks_out[name]['sha256'][:16]} finite={fin}",
              flush=True)
    wall = time.time() - t0
    shas = [v["sha256"] for v in decks_out.values()]
    man = {"device": args.device, "nstep": args.nstep, "dt": DT,
           "wall_s": round(wall, 2), "pairwise_distinct": len(set(shas)) == len(shas),
           "torch": torch.__version__,
           "python": platform.python_version(), "decks": decks_out}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"decks {len(decks_out)}, pairwise distinct {man['pairwise_distinct']}, "
          f"wall {wall:.2f} s")


def compare(a, b):
    ident, worst = 0, 0.0
    names = [n for n, *_ in decks()]
    for n in names:
        ua = torch.load(os.path.join(a, n + ".pt"))
        ub = torch.load(os.path.join(b, n + ".pt"))
        same = ua.shape == ub.shape and torch.equal(
            ua.view(torch.int64), ub.view(torch.int64))
        ident += same
        if ua.shape != ub.shape:
            print(f"{n:34s} shape differs {list(ua.shape)} {list(ub.shape)}")
            continue
        e_a, e_b = ua[:, IPR], ub[:, IPR]
        rel = ((e_b - e_a).abs().flatten(1).max(1).values /
               e_a.abs().flatten(1).max(1).values)  # one value per step
        r = rel.max().item()
        worst = max(worst, r)
        print(f"{n:34s} identical={same} maxabs={(ub - ua).abs().max().item():.3e} "
              f"energy_rel_per_step_max={r:.3e}")
    print(f"identical {ident}/{len(names)}; worst energy-row rel diff {worst:.3e}")
    return ident == len(names)


def check_manifest(d, ref):
    with open(os.path.join(d, "manifest.json")) as f:
        got = json.load(f)["decks"]
    with open(ref) as f:
        want = json.load(f)["decks"]
    ok = 0
    for n, *_ in decks():
        same = got[n]["sha256"] == want[n]["sha256"]
        ok += same
        print(f"{n:34s} {'same' if same else 'DIFFERENT'}")
    print(f"matching checksums {ok}/{len(want)}")
    return ok == len(want)


def write_decks(d):
    os.makedirs(d, exist_ok=True)
    for name, scheme, work, geom in decks():
        with open(os.path.join(d, name + ".yaml"), "w") as f:
            f.write(f"# #348 harness deck {name}; initial state: h348.py "
                    "initial_w(); dt = %.17g s; 10 steps\n" % DT)
            yaml.safe_dump(config(scheme, work, geom), f, sort_keys=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out")
    ap.add_argument("--nstep", type=int, default=10)
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--check-manifest", nargs=2)
    ap.add_argument("--write-decks")
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    if args.check_manifest:
        sys.exit(0 if check_manifest(*args.check_manifest) else 1)
    if args.write_decks:
        write_decks(args.write_decks)
        sys.exit(0)
    produce(args)
