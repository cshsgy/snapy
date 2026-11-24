#!/usr/bin/env python3
"""nb/pz bitwise gate for V3 wave 1, lane S step (3) (S.rst rev 2.4 76204d46,
:1471-1476, :1949-1954): moving the x1 relays out of hydro must not change
one bit. A gate tool, not part of the wave PR.

5 decks x their nb/pz configs = 19 runs, FP64. Every run steps the deck and
writes each local block's final hydro_u (and scalar_s) as raw bytes; two
builds pass when every dump is byte-identical. Configs:

  pz1_nb1       1 block                                     1 process
  pz1_nb2       x2 split, 2 blocks (cartesian decks only)   1 process
  pz2_nb2       x1 split, 2 blocks (same-process boards)    1 process
  pz2_nb1_mp    x1 split, 1 block per process               2 processes
  pz2x2_nb2_mp  x1 and x2 split, 2 blocks per process       2 processes
                (boards and process group; cartesian decks only)

usage:
  gate_nbpz.py --device cpu|cuda --out DIR [--decks a,b] [--perturb-ulp]
      run every deck x config with this Python's snapy; DIR gets the raw
      dumps, one log per run and manifest.json (sha256, size of each dump)
  gate_nbpz.py --compare DIR_A DIR_B
      byte compare two runs; exit 0 only when both hold the same, non-empty
      set of dumps and every pair is identical
  gate_nbpz.py --check-manifest DIR MANIFEST
      compare DIR's checksums with a stored manifest (results/*.json)
  gate_nbpz.py --negative-control DIR --device cpu|cuda
      rerun one deck with a 1-ulp change in one cell next to the x1 seam into
      DIR/negctl and require --compare to report a difference (exit 0 when
      the change is seen)
  gate_nbpz.py --write-decks DIR
      write the 19 input YAML decks (config() is the source)

Two-process runs are launched by this script (RANK, WORLD_SIZE, MASTER_PORT,
BLOCKS_PER_PROCESS, DEVICE, DEVICE_ID per rank). Backend: gloo on CPU; on
CUDA, ucx (commux), since gloo takes CPU tensors only. Override with
GATE_BACKEND. Never compare dumps across devices.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NSTEP = {"cart3d": 10, "cart3d_pert": 10, "sph_co4": 20, "sph_centroid": 20, "sph_face": 20}
# name: (nb1 = pz, nb2, blocks per process, processes)
CONFIGS = {"pz1_nb1": (1, 1, 1, 1), "pz1_nb2": (1, 2, 2, 1), "pz2_nb2": (2, 1, 2, 1),
           "pz2_nb1_mp": (2, 1, 1, 2), "pz2x2_nb2_mp": (2, 2, 2, 2)}
ENV = {"sph_centroid": {"SNAP_X1_CENTROID_EXACT": "1"}}


def cart3d():
    """moist 3D column, passive scalar, flux covariance on (default): rows 6-9,
    13-16, 19, 22-25"""
    return {
        "reference-state": {"Tref": 0., "Pref": 1.e5},
        "species": [
            {"name": "dry", "composition": {"H": 1.6667, "He": 0.16667}, "cv_R": 2.5},
            {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 2.5, "u0_R": 0.},
            {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.},
        ],
        "reactions": [{"equation": "vapor <=> cloud", "type": "nucleation",
                       "rate-constant": {"formula": "h2o_ideal"}}],
        "geometry": {"type": "cartesian",
                     "bounds": {"x1min": 0., "x1max": 1.e4, "x2min": 0., "x2max": 1.e4,
                                "x3min": 0., "x3max": 5.e3},
                     "cells": {"nx1": 24, "nx2": 8, "nx3": 4, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-moist", "density-floor": 1.e-10,
                                  "pressure-floor": 1.e-10, "limiter": False},
            "reconstruct": {"vertical": {"type": "weno5", "scale": False, "shock": False},
                            "horizontal": {"type": "weno5", "scale": False, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "periodic", "x2-outer": "periodic",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -10.}},
        "scalar": {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "weno5"}},
    }


def cart3d_pert():
    """cart3d with anomaly + perturbation-flux, hllc: row 8's paired 4-field form"""
    d = cart3d()
    d["dynamics"].update({"precision": "float64", "anomaly": True, "perturbation-flux": True,
                          "riemann-solver": {"type": "hllc"}})
    return d


def sph(gw="cell", co4=False):
    """spherical-polar column; co4: conversion-order 4 + plm (rows 3-4);
    gravity-work face: row 19; with SNAP_X1_CENTROID_EXACT: rows 5, 10"""
    rec = "plm" if co4 else "weno5"
    d = {
        "geometry": {"type": "spherical-polar",
                     "bounds": {"x1min": 5., "x1max": 7., "x2min": 1.5207963267948966,
                                "x2max": 1.6207963267948966, "x3min": 0., "x3max": 0.1},
                     "cells": {"nx1": 32, "nx2": 1, "nx3": 1, "nghost": 3}},
        "dynamics": {
            "equation-of-state": {"type": "ideal-gas", "gammad": 1.4, "weight": 8.31446,
                                  "density-floor": 1.e-12, "pressure-floor": 1.e-12,
                                  "temperature-floor": 1.e-12, "limiter": True},
            "reconstruct": {"vertical": {"type": rec, "scale": False, "shock": False},
                            "horizontal": {"type": rec, "scale": False, "shock": False}},
            "riemann-solver": {"type": "lmars"}},
        "integration": {"type": "rk3", "cfl": 0.4, "implicit-scheme": 0, "nlim": -1, "tlim": 1.e9},
        "forcing": {"const-gravity": {"grav1": -1., "non-hydrostatic": 1., "gravity-work": gw}},
        "boundary-condition": {"external": {
            "x1-inner": "reflecting", "x1-outer": "reflecting",
            "x2-inner": "reflecting", "x2-outer": "reflecting",
            "x3-inner": "periodic", "x3-outer": "periodic"}},
    }
    if co4:
        d["dynamics"]["conversion-order"] = 4
    return d


DECKS = {"cart3d": cart3d, "cart3d_pert": cart3d_pert, "sph_co4": lambda: sph(co4=True),
         "sph_centroid": sph, "sph_face": lambda: sph(gw="face")}


def runs():
    for dn in DECKS:
        for cn, (nb1, nb2, bpp, nproc) in CONFIGS.items():
            if not dn.startswith("cart3d") and nb2 > 1:
                continue
            yield dn, cn, nb1, nb2, bpp, nproc


def config(dn, nb1, nb2, bpp):
    c = copy.deepcopy(DECKS[dn]())
    c["distribute"] = {"layout": "cubed" if nb1 > 1 else "slab", "nb1": nb1, "nb2": nb2,
                       "nb3": 1, "blocks_per_process": bpp}
    return c


def write_decks(d):
    os.makedirs(d, exist_ok=True)
    for dn, cn, nb1, nb2, bpp, _ in runs():
        with open(os.path.join(d, f"{dn}.{cn}.yaml"), "w") as f:
            yaml.safe_dump(config(dn, nb1, nb2, bpp), f)


# ------------------------------------------------------------------ worker
def worker(deck, nstep, device, prefix, perturb):
    """one rank: step the deck, dump every local block"""
    import torch
    from snapy import Mesh, MeshOptions
    torch.set_num_threads(1)
    cfg = yaml.safe_load(open(deck))
    b = cfg["geometry"]["bounds"]
    opts = MeshOptions.from_yaml(deck)
    mesh = Mesh(opts)
    blocks = list(mesh.blocks)
    prank = int(os.environ.get("RANK", "0"))
    if device == "cuda":
        dev = torch.device("cuda", int(os.environ.get("DEVICE_ID", "0")))
    else:
        dev = torch.device("cpu")
    for blk in blocks:
        blk.to(dev, torch.float64)
    moist = cfg["dynamics"]["equation-of-state"]["type"] == "ideal-moist"
    ps, rs = (1.e5, 1.e5 / (3900. * 320.)) if moist else (1., 1.)
    vs = math.sqrt(ps / rs) * 0.02
    nscalar = cfg.get("scalar", {}).get("nvar", 0)
    mesh_vars = []
    for blk in blocks:
        buf = dict(blk.named_buffers())
        z = ((buf["coord.x1v"] - b["x1min"]) / (b["x1max"] - b["x1min"])).cpu()
        y = ((buf["coord.x2v"] - b["x2min"]) / (b["x2max"] - b["x2min"])).cpu()
        x = ((buf["coord.x3v"] - b["x3min"]) / (b["x3max"] - b["x3min"])).cpu()
        z, y, x = z.view(1, 1, -1), y.view(1, -1, 1), x.view(-1, 1, 1)
        w = buf["hydro.D"].detach().clone().zero_().cpu()
        shape = w.shape[1:]
        p = torch.exp(-2. * z).expand(shape)
        rho = p * (1. + 0.02 * torch.sin(2. * math.pi * y) * torch.cos(2. * math.pi * x)
                   * torch.exp(-((z - 0.5) / 0.15) ** 2))
        w[0] = rs * rho  # IDN
        w[4] = ps * p    # IPR
        inside = ((z > 0.) & (z < 1.)).to(w.dtype)
        w[1] = (vs * inside * torch.sin(math.pi * z) * (1. + 0.3 * torch.cos(2. * math.pi * y))).expand(shape)
        if shape[1] > 1:
            w[2] = (vs * (0.5 + z) * torch.ones_like(y)).expand(shape)
        if shape[0] > 1:
            w[3] = (0.5 * vs * (1. - z) * torch.ones_like(x)).expand(shape)
        for n in range(5, w.shape[0]):
            w[n] = (0.002 * torch.exp(-3. * z) * (1. + 0.5 * torch.sin(2. * math.pi * y))).expand(shape)
        v = {"hydro_w": w.to(dev)}
        if nscalar:
            r = torch.stack([(0.001 * (k + 1) * torch.exp(-4. * z)
                              * (1. + 0.5 * torch.sin(2. * math.pi * y))).expand(shape)
                             for k in range(nscalar)])
            v["scalar_r"] = r.to(dev).contiguous()
        mesh_vars.append(v)
    if perturb:  # 1 ulp in one interior cell of the first block, next to its top
        w = mesh_vars[0]["hydro_w"]
        ng = cfg["geometry"]["cells"]["nghost"]
        k, j, i = min(ng, w.shape[1] - 1), min(ng, w.shape[2] - 1), w.shape[3] - ng - 1
        c = w[0, k, j, i].clone()
        w[0, k, j, i] = torch.nextafter(c, c + 1.)
    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    t = 0.
    for n in range(nstep):
        mesh.set_cycle(n + 1)
        dt = mesh.max_time_step(mesh_vars)
        for s in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, s)
        t += dt
    for blk, v in zip(blocks, mesh_vars):
        gr = blk.get_layout().options.rank()
        for key in ("hydro_u", "scalar_s"):
            if key in v:
                a = v[key].detach().cpu().contiguous()
                with open(f"{prefix}.block{gr:02d}.{key}.bin", "wb") as f:
                    f.write(a.numpy().tobytes())
                with open(f"{prefix}.block{gr:02d}.{key}.shape", "w") as f:
                    f.write(" ".join(map(str, a.shape)) + "\n")
    print(f"rank {prank}: {len(blocks)} blocks, {nstep} steps, t = {t:.17g}", flush=True)


# ------------------------------------------------------------------ driver
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(deck, nstep, device, prefix, bpp, nproc, extra_env, log, perturb):
    args = [sys.executable, os.path.abspath(__file__), "--worker", deck, str(nstep),
            device, prefix] + (["--perturb-ulp"] if perturb else [])
    base = dict(os.environ, **extra_env, BLOCKS_PER_PROCESS=str(bpp), DEVICE=device)
    if device == "cuda":
        import torch
        ngpu = max(1, torch.cuda.device_count())
    else:
        ngpu = 1
    procs = []
    port = str(free_port()) if nproc > 1 else None
    backend = os.environ.get("GATE_BACKEND", "ucx" if device == "cuda" else "gloo")
    with open(log, "w") as lf:
        for r in range(nproc):
            env = dict(base, DEVICE_ID=str(r % ngpu))
            if nproc > 1:
                env.update(RANK=str(r), WORLD_SIZE=str(nproc), PROCESS_RANK=str(r),
                           PROCESS_WORLD_SIZE=str(nproc), LOCAL_RANK=str(r),
                           LOCAL_WORLD_SIZE=str(nproc), MASTER_ADDR="127.0.0.1",
                           MASTER_PORT=port, BACKEND=backend)
            procs.append(subprocess.Popen(args, env=env, stdout=lf, stderr=subprocess.STDOUT))
        rcs = [p.wait(timeout=1800) for p in procs]
    return max(rcs, key=abs)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_all(out, device, only, perturb=False):
    import torch
    os.makedirs(out, exist_ok=True)
    deck_dir = os.path.join(out, "decks")
    write_decks(deck_dir)
    failed = []
    t0 = time.time()
    for dn, cn, nb1, nb2, bpp, nproc in runs():
        if only and dn not in only:
            continue
        name = f"{dn}.{cn}"
        rc = launch(os.path.join(deck_dir, name + ".yaml"), NSTEP[dn], device,
                    os.path.join(out, name), bpp, nproc, ENV.get(dn, {}),
                    os.path.join(out, name + ".log"), perturb)
        print(f"{name:26s} rc={rc}", flush=True)
        if rc != 0:
            failed.append(name)
    dumps = sorted(f for f in os.listdir(out) if f.endswith(".bin"))
    import snapy
    manifest = {
        "device": device, "nstep": NSTEP, "configs": CONFIGS,
        "torch": torch.__version__, "snapy": getattr(snapy, "__version__", "?"),
        "wall_s": round(time.time() - t0, 1), "failed_runs": failed,
        "dumps": {f: {"sha256": sha256(os.path.join(out, f)),
                      "bytes": os.path.getsize(os.path.join(out, f))} for f in dumps},
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(dumps)} dumps, {len(failed)} failed runs -> {out}/manifest.json")
    return 1 if failed else 0


def load(d_or_manifest):
    p = d_or_manifest
    if os.path.isdir(p):
        p = os.path.join(p, "manifest.json")
    with open(p) as f:
        return json.load(f)


def compare(a, b, quiet=False):
    ma, mb = load(a), load(b)
    if ma.get("device") != mb.get("device"):
        print(f"FAIL: devices differ ({ma.get('device')} vs {mb.get('device')}); "
              "never compare across devices")
        return 1
    da, db = ma["dumps"], mb["dumps"]
    names = sorted(set(da) | set(db))
    same = diff = missing = 0
    for n in names:
        if n not in da or n not in db:
            missing += 1
            verdict = "MISSING"
        else:
            ok = da[n]["sha256"] == db[n]["sha256"]
            pa, pb = os.path.join(a, n), os.path.join(b, n)
            if ok and os.path.isdir(a) and os.path.isdir(b) and os.path.exists(pa) and os.path.exists(pb):
                with open(pa, "rb") as fa, open(pb, "rb") as fb:  # bytes, not only hashes
                    ok = fa.read() == fb.read()
            same, diff = same + ok, diff + (not ok)
            verdict = "identical" if ok else "DIFFERS"
        if not quiet:
            print(f"{n:48s} {da.get(n, {}).get('sha256', '-')[:16]} "
                  f"{db.get(n, {}).get('sha256', '-')[:16]} {verdict}")
    fails = ma.get("failed_runs", []) + mb.get("failed_runs", [])
    print(f"{same}/{len(names)} dumps identical, {diff} differ, {missing} missing; "
          f"failed runs: {len(fails)}")
    return 0 if names and same == len(names) and not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=("cpu", "cuda"))
    ap.add_argument("--out")
    ap.add_argument("--decks", default="")
    ap.add_argument("--perturb-ulp", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--check-manifest", nargs=2)
    ap.add_argument("--negative-control")
    ap.add_argument("--write-decks")
    ap.add_argument("--worker", nargs=4)
    a = ap.parse_args()
    if a.worker:
        deck, nstep, device, prefix = a.worker
        worker(deck, int(nstep), device, prefix, a.perturb_ulp)
        return 0
    if a.write_decks:
        write_decks(a.write_decks)
        return 0
    if a.compare:
        return compare(*a.compare)
    if a.check_manifest:
        return compare(*a.check_manifest)
    if a.negative_control:
        # one deck that crosses x1 seams in every transport; a 1-ulp input change
        # must reach the dumps, else the gate could not see a transport error
        ctl = os.path.join(a.negative_control, "negctl")
        run_all(ctl, a.device, {"cart3d"}, perturb=True)
        base = load(a.negative_control)["dumps"]
        neg = load(ctl)["dumps"]
        keys = sorted(k for k in neg if k.startswith("cart3d."))
        d = sum(base.get(k, {}).get("sha256") != neg[k]["sha256"] for k in keys)
        runs_hit = sorted({k.split(".block")[0] for k in keys
                           if base.get(k, {}).get("sha256") != neg[k]["sha256"]})
        print(f"negative control: {d}/{len(keys)} cart3d dumps differ from {a.negative_control}; "
              f"runs that see the change: {len(runs_hit)}/{len({k.split('.block')[0] for k in keys})}")
        return 0 if keys and len(runs_hit) == len({k.split('.block')[0] for k in keys}) else 1
    if a.device and a.out:
        return run_all(a.out, a.device, set(filter(None, a.decks.split(","))))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
