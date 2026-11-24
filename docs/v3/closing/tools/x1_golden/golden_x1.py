#!/usr/bin/env python3
"""X1 gate (X.rst 07d4eedf section 5, row 1): N-step golden of hydro_u and scalar_s,
Cartesian and cubed sphere, FP64 and FP32; base vs step must be byte-identical.

  python golden_x1.py run --device cpu --out DIR [--case NAME] [--perturb]
  python golden_x1.py compare DIR_A DIR_B

The snapy under test is whatever `import snapy` finds (set PYTHONPATH to the build's install
prefix); SNAPY_TESTS overrides the tests/ directory the decks are read from.

Each case runs in a child process and writes DIR/<case>.s<k>.<block>.<field>.bin (raw C-order
bytes, ghosts included) after every accepted step k = 1..N, plus DIR/<case>.json. --perturb adds
one ulp to hydro_u's vapor row (ICY) at the middle cell before the first step (negative control).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

# snapy tests/ directory (decks and helper modules); default: this checkout's tests/
TESTS = os.environ.get("SNAPY_TESTS", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "..", "..", "tests"))
NSTEP = 10
CASES = ["cart_moist_fp64", "cart_moist_fp32", "jupiter_crm_fp64", "jupiter_crm_fp32", "cubed6_fp64"]
# cubed6_fp32 is refused at the base: float32 needs perturbation-flux, which needs anomaly
# storage, which is Cartesian only (capability matrix: cubed sphere eager FP32 is G5)


def cart_moist(precision):
    import yaml
    with open(os.path.join(TESTS, "test_anomaly_storage.yaml")) as f:
        cfg = yaml.safe_load(f)
    # as tests/test_perturbation_flux.py make(True, precision=...), plus a passive scalar
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True, "conversion-order": 2,
                            "precision": precision, "riemann-solver": {"type": "hllc"}})
    cfg["geometry"]["cells"]["nx3"] = 4
    cfg["geometry"]["bounds"]["x3max"] = 4000.
    cfg["boundary-condition"]["external"].update({"x3-inner": "periodic", "x3-outer": "periodic"})
    cfg["integration"]["implicit-scheme"] = 0
    cfg["forcing"]["const-gravity"]["non-hydrostatic"] = 1.
    cfg["scalar"] = {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "weno5"}}
    return cfg


def jupiter_crm(precision):
    import yaml
    with open(os.path.join(TESTS, "../examples/jupiter_crm.yaml")) as f:
        cfg = yaml.safe_load(f)
    # as tests/test_fp32_jupiter.py run(): H2O/NH3 CRM, VIC scheme 9, FP64 kintera
    names = {"dry", "H2O", "NH3", "H2O(l)", "H2O(l,p)", "NH3(s)", "NH3(s,p)"}
    cfg["species"] = [s for s in cfg["species"] if s["name"] in names]
    cfg["reactions"] = [r for r in cfg["reactions"]
                        if "H2S" not in r["equation"] and "NH4SH" not in r["equation"]]
    cfg["geometry"] = {"type": "cartesian",
                       "bounds": {"x1min": 0.0, "x1max": 2.0e5, "x2min": 0.0, "x2max": 2.0e4,
                                  "x3min": 0.0, "x3max": 2.0e4},
                       "cells": {"nx1": 24, "nx2": 4, "nx3": 4, "nghost": 3}}
    cfg["distribute"] = {"layout": "slab", "nb2": 1, "nb3": 1}
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True, "precision": precision,
                            "riemann-solver": {"type": "hllc"}})
    cfg["integration"].update({"implicit-scheme": 9, "cfl": 1.0, "ncycle_out": 0})
    cfg["forcing"] = {"const-gravity": {"grav1": -24.79},
                      "bot-heat": {"flux": 100.0, "depth": 3},
                      "top-cool": {"flux": -100.0, "depth": 3},
                      "diffusion": {"kappa_iso": 10.0, "nu_iso": 10.0}}
    cfg["sedimentation"] = {"radius": {}, "density": {},
                            "const-vsed": {"H2O(l,p)": -20.0, "NH3(s,p)": -20.0}}
    cfg["outputs"] = []
    cfg.pop("problem", None)
    cfg["scalar"] = {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "weno5"}}
    return cfg


def cubed6(precision):
    import yaml
    with open(os.path.join(TESTS, "test_axial_momentum_flux_form.yaml")) as f:
        cfg = yaml.safe_load(f)
    # six gnomonic-equiangle panels in one process (Mesh), the #314 deck with its moist arm
    # (h): vapor and implicit-scheme 9; FP32 needs the perturbation flux
    import test_axial_momentum_flux_form as amf
    cfg["species"] = cfg["species"][:1] + amf.MOIST_SPECIES
    cfg["reactions"] = amf.MOIST_REACTIONS
    cfg["dynamics"]["equation-of-state"]["type"] = "ideal-moist"
    cfg["integration"]["implicit-scheme"] = 9
    # anomaly storage is Cartesian only ("Step 1"), so the panels run without it
    cfg["dynamics"].update({"perturbation-flux": precision == "float32", "precision": precision})
    cfg["scalar"] = {"nvar": 1, "names": ["trace"], "reconstruct": {"type": "weno5"},
                     "riemann-solver": {"type": "upwind"}}
    return cfg


def write_yaml(cfg):
    import yaml
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        return f.name


def child(name, device, out, perturb):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    sys.path.insert(0, TESTS)
    import snapy
    from snapy import kIDN, kICY
    geo, prec = name.rsplit("_", 1)
    precision = {"fp64": "float64", "fp32": "float32"}[prec]
    dev = torch.device(device)
    if geo == "cubed6":
        import test_axial_momentum_flux_form as amf
        cfg = cubed6(precision)
        mesh, blocks = amf.build(cfg, device)
        mvars = amf.initial_state(cfg, blocks, lambda lon, lat, rc: amf.W_SB * rc
                                  + amf.U_PERT * torch.cos(lat) ** 2 * torch.sin(2.0 * lon), True,
                                  w_up=amf.W_UP, vapor=amf.QV)
        for b, v in zip(blocks, mvars):
            x2 = b.module("coord").buffer("x2v").to(v["hydro_w"]).view(1, -1, 1)
            v["scalar_r"] = (0.5 + 0.4 * torch.sin(4.0 * x2)).expand(1, *v["hydro_w"].shape[1:]).clone()
        mvars, _ = mesh.initialize(mvars)
        intg = blocks[0].module("intg")

        def step(cycle):
            mesh.set_cycle(cycle)
            dt = mesh.max_time_step(mvars)
            for stage in range(len(intg.stages)):
                mesh.forward(mvars, dt, stage)
            return mesh.check_redo(mvars), dt
        states = mvars
    else:
        from test_anomaly_storage import initial_state
        cfg = cart_moist(precision) if geo == "cart_moist" else jupiter_crm(precision)
        path = write_yaml(cfg)
        try:
            block = snapy.MeshBlock(snapy.MeshBlockOptions.from_yaml(path))
        finally:
            os.unlink(path)
        block.to(dev)
        blocks = [block]
        if geo == "cart_moist":
            w = initial_state(block)["hydro_w"]
            w[snapy.kIV1 + 2] = .05 * torch.sin(block.module("coord").buffer("x3v").view(-1, 1, 1)
                                               / 4000. * 2. * torch.pi)
        else:
            w = jupiter_initial(block, torch, snapy)
        x2 = block.module("coord").buffer("x2v").to(w).view(1, -1, 1)
        r = (0.5 + 0.4 * torch.sin(x2 / 1.0e3)).expand(1, *w.shape[1:]).clone()
        bv, _ = block.initialize({"hydro_w": w, "scalar_r": r})
        intg = block.module("intg")

        def step(cycle):
            dt = block.max_time_step(bv) if geo == "cart_moist" else 0.1
            for stage in range(len(intg.stages)):
                block.forward(bv, dt, stage)
            return block.check_redo(bv), dt
        states = [bv]
    if perturb:
        u = states[0]["hydro_u"]
        k, j, i = (s // 2 for s in u.shape[1:])
        v = u[kICY, k, j, i]  # vapor: non-zero in every case (IDN anomalies start at 0)
        v.copy_(torch.nextafter(v, torch.tensor(float("inf"), dtype=v.dtype, device=v.device)))
    meta = {"case": name, "device": device, "nstep": NSTEP, "dt": [], "files": {},
            "dtype": {}, "u0_dtype": [str(dict(b.named_buffers())["u0"].dtype) for b in blocks]}
    for cycle in range(1, NSTEP + 1):
        err, dt = step(cycle)
        if err != 0:
            raise RuntimeError(f"{name}: check_redo {err} at step {cycle}")
        meta["dt"].append(float(dt))
        for ib, v in enumerate(states):
            for key in ("hydro_u", "scalar_s"):
                if key not in v:
                    continue
                a = v[key].detach().contiguous().cpu()
                fn = f"{name}.s{cycle:02d}.b{ib}.{key}.bin"
                with open(os.path.join(out, fn), "wb") as f:
                    f.write(a.numpy().tobytes())
                meta["files"][fn] = hashlib.sha256(a.numpy().tobytes()).hexdigest()
                meta["dtype"][key] = str(a.dtype)
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump(meta, f, indent=1)


def jupiter_initial(block, torch, snapy):
    # tests/test_fp32_jupiter.py's initial column (its run() also steps, with fixed dt 0.1)
    from snapy import kIDN, kIPR, kICY, kIV1
    ty = block.module("hydro.eos.thermo")
    w = block.module("hydro").buffer("D").clone().zero_()
    z = block.module("coord").buffer("x1v")
    temp = (400.0 - 1.0e-3 * z).clamp_min(150.0)
    rd = 8.31446261815324 * ty.buffer("inv_mu")[0]
    pressure = 1.0e7 * (temp / 400.0) ** (24.79 / (rd * 0.001))
    w[kIPR] = pressure
    w[kIDN] = pressure / (rd * temp)
    w[kICY].fill_(0.01)
    w[kICY + 1].fill_(0.001)
    w[kICY + 3].fill_(1.0e-5)
    w[kICY + 5].fill_(1.0e-6)
    w[kIV1] = 0.01 * torch.sin(torch.pi * z / 2.0e5)
    return w


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--case", action="append")
    r.add_argument("--perturb", action="store_true")
    r.add_argument("--child", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        os.makedirs(a.out, exist_ok=True)
        if a.child:
            child(a.case[0], a.device, a.out, a.perturb)
            return 0
        status = {}
        for name in a.case or CASES:
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--child", "--case", name,
                   "--device", a.device, "--out", a.out] + (["--perturb"] if a.perturb else [])
            p = subprocess.run(cmd, capture_output=True, text=True)
            status[name] = p.returncode
            print(f"{name}: rc={p.returncode}", flush=True)
            if p.returncode:
                print(p.stdout[-2000:], p.stderr[-3000:], flush=True)
        with open(os.path.join(a.out, "status.json"), "w") as f:
            json.dump(status, f)
        return 0 if all(v == 0 for v in status.values()) else 1
    return compare(a.a, a.b)


def compare(da, db):
    import numpy as np
    sa = json.load(open(os.path.join(da, "status.json")))
    sb = json.load(open(os.path.join(db, "status.json")))
    rc, same, differ, missing = 0, 0, 0, 0
    for name in sorted(set(sa) | set(sb)):
        if sa.get(name) != 0 or sb.get(name) != 0:
            print(f"{name}: FAILED RUN (a rc={sa.get(name)}, b rc={sb.get(name)})")
            rc = 1
            continue
        ma = json.load(open(os.path.join(da, f"{name}.json")))
        mb = json.load(open(os.path.join(db, f"{name}.json")))
        bad = []
        for fn in sorted(set(ma["files"]) | set(mb["files"])):
            if fn not in ma["files"] or fn not in mb["files"]:
                missing += 1
                bad.append(fn + " (missing)")
                continue
            ba = open(os.path.join(da, fn), "rb").read()
            bb = open(os.path.join(db, fn), "rb").read()
            if ba == bb:
                same += 1
            else:
                differ += 1
                dt = np.float32 if ma["dtype"][fn.rsplit(".", 2)[1]] == "torch.float32" else np.float64
                x, y = np.frombuffer(ba, dt), np.frombuffer(bb, dt)
                bad.append(f"{fn} max|diff|={np.nanmax(np.abs(x - y)):.3e} n={int((x != y).sum())}")
        dtsame = ma["dt"] == mb["dt"]
        print(f"{name}: {len(ma['files'])} dumps, dt identical={dtsame}, dtype={ma['dtype']}, "
              f"u0 a={ma['u0_dtype'][0]} b={mb['u0_dtype'][0]}, differ={len(bad)}")
        for line in bad[:5]:
            print("   ", line)
        if bad or not dtsame:
            rc = 1
    print(f"SUMMARY {same} identical, {differ} differ, {missing} missing; RESULT {'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
