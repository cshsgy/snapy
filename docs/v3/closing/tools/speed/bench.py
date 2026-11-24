#!/usr/bin/env python3
"""V3 speed audit step timer. One process = one rank; launch N ranks with RANK/WORLD_SIZE env
(the nb/pz gate's scheme). Times every step after a warm-up with time.perf_counter (CUDA:
torch.cuda.synchronize before reading the clock) and writes a JSON report.

  python bench.py --deck DECK --device cpu|cuda --out report.json [--precision float64|float32]
                  [--execution eager|cuda-graph] [--warmup 5] [--steps 50] [--dt-mode fixed|varying]
                  [--nb2 N --nb3 N] [--init generic|straka]
DECK is a yaml path or the builtin name 'straka_hllc' (run_fp32_straka's dry single-species
ideal-moist HLLC Straka, the one FP32 path both trees support).
"""
import argparse, json, math, os, sys, time, tempfile, importlib.metadata, hashlib
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent


def straka_hllc_deck(tree):
    cfg = yaml.safe_load((tree / "examples/straka.yaml").read_text())
    moist = yaml.safe_load((tree / "tests/test_anomaly_storage.yaml").read_text())
    cfg["reference-state"] = moist["reference-state"]
    cfg["species"] = moist["species"][:1]
    cfg["reactions"] = []
    cfg["dynamics"].update({"anomaly": True, "perturbation-flux": True,
                            "riemann-solver": {"type": "hllc"}})
    cfg["dynamics"]["equation-of-state"] = {"type": "ideal-moist", "limiter": False,
                                            "density-floor": 1.0e-10, "pressure-floor": 1.0e-10}
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--tree", default=None, help="snapy source tree (for the builtin deck)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", default=None)
    ap.add_argument("--execution", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dt-mode", default="fixed", choices=["fixed", "varying"])
    ap.add_argument("--nb1", type=int, default=None)
    ap.add_argument("--nb2", type=int, default=None)
    ap.add_argument("--nb3", type=int, default=None)
    ap.add_argument("--init", default="generic", choices=["generic", "straka"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--driver", default="auto", choices=["auto", "block", "mesh"],
                    help="block: MeshBlock.forward (the only path that captures CUDA graphs; one local block, one rank); mesh: Mesh.forward (eager, any decomposition)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", type=int, default=0, help="after the timed steps, profile this many more steps with torch.profiler; writes OUT.prof.txt")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    import snapy
    from snapy import Mesh, MeshOptions

    if a.deck == "straka_hllc":
        cfg = straka_hllc_deck(Path(a.tree))
    else:
        cfg = yaml.safe_load(open(a.deck))
    cfg["outputs"] = []
    cfg.setdefault("integration", {})["ncycle_out"] = 0
    cfg["integration"]["nlim"] = -1
    cfg["integration"]["tlim"] = 1.0e30
    if a.precision:
        cfg["dynamics"]["precision"] = a.precision
    if a.execution and a.execution != "eager":  # eager is the default in both trees; 730b3500 rejects the key
        cfg["integration"]["execution"] = a.execution
    for k in ("nb1", "nb2", "nb3"):
        v = getattr(a, k)
        if v is not None:
            cfg.setdefault("distribute", {})[k] = v
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        deck_path = f.name
    os.environ["DEVICE"] = a.device
    os.environ.setdefault("DEVICE_ID", "0")
    dev = torch.device("cuda", int(os.environ["DEVICE_ID"])) if a.device == "cuda" else torch.device("cpu")
    rank = int(os.environ.get("RANK", "0"))

    mesh = Mesh(MeshOptions.from_yaml(deck_path))
    os.unlink(deck_path)
    blocks = list(mesh.blocks)
    for blk in blocks:
        blk.to(dev, torch.float64)
    b = dict(cfg["geometry"]["bounds"])
    for k in ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max"):  # cubed-sphere decks give angles as *_pi
        if k not in b and k + "_pi" in b:
            b[k] = math.pi * float(b[k + "_pi"])
        b[k] = float(b[k])
    moist = cfg["dynamics"]["equation-of-state"]["type"] == "ideal-moist"
    nscalar = cfg.get("scalar", {}).get("nvar", 0)
    mesh_vars = []
    for blk in blocks:
        buf = dict(blk.named_buffers())
        w = buf["hydro.D"].detach().clone().zero_().cpu().double()
        shape = w.shape[1:]
        if a.init == "straka":
            # run_fp32_straka's cold bubble (dry, hydrostatic 300 K potential temperature)
            if moist:
                inv_mu = blk.module("hydro.eos.thermo").buffer("inv_mu")[0].item()
                rd = 8.31446261815324 * inv_mu
            else:
                rd = 8.31446261815324 / cfg["dynamics"]["equation-of-state"]["weight"]
            cp = 3.5 * rd
            z = buf["coord.x1v"].cpu().double().view(1, 1, -1)
            x = buf["coord.x2v"].cpu().double().view(1, -1, 1)
            temp = 300.0 - 9.8 * z / cp
            w[4] = 1.0e5 * (temp / 300.0) ** (cp / rd)
            radius = ((x / 4000.0) ** 2 + ((z - 3000.0) / 2000.0) ** 2).sqrt()
            temp = temp + torch.where(radius <= 1, -7.5 * (torch.cos(torch.pi * radius) + 1.0), 0.0)
            w[0] = w[4] / (rd * temp)
            v = {"hydro_w": w.to(dev)}
            if nscalar:
                ratio = (torch.zeros_like(w[:1]) + (z / 6400.0).clamp(0.0, 1.0)).expand(nscalar, *shape)
                v["scalar_r"] = ratio.to(dev).contiguous()
        else:
            # the nb/pz gate's generic hydrostatic column with a 2% wave (works on any geometry)
            z = ((buf["coord.x1v"] - b["x1min"]) / (b["x1max"] - b["x1min"])).cpu().double()
            y = ((buf["coord.x2v"] - b["x2min"]) / (b["x2max"] - b["x2min"])).cpu().double()
            x = ((buf["coord.x3v"] - b["x3min"]) / (b["x3max"] - b["x3min"])).cpu().double()
            z, y, x = z.view(1, 1, -1), y.view(1, -1, 1), x.view(-1, 1, 1)
            ps, rs = (1.e5, 1.e5 / (3900. * 320.)) if moist else (1., 1.)
            vs = math.sqrt(ps / rs) * 0.02
            p = torch.exp(-2. * z).expand(shape)
            rho = p * (1. + 0.02 * torch.sin(2. * math.pi * y) * torch.cos(2. * math.pi * x)
                       * torch.exp(-((z - 0.5) / 0.15) ** 2))
            w[0] = rs * rho
            w[4] = ps * p
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
    mesh_vars, _ = mesh.initialize(mesh_vars)
    intg = blocks[0].module("intg")
    nstage = len(intg.stages)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    driver = a.driver
    if driver == "auto":
        driver = "block" if (len(blocks) == 1 and world == 1) else "mesh"
    if driver == "block":
        assert len(blocks) == 1 and world == 1, "block driver needs one block in one process"
    capability = None
    try:
        capability = dict(blocks[0].execution_capability(mesh_vars[0]))
    except Exception as e:  # before tree: no such API
        capability = str(type(e).__name__)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    def step(n, dt):
        mesh.set_cycle(n + 1)
        if driver == "block":
            for s in range(nstage):
                blocks[0].forward(mesh_vars[0], dt, s)
        else:
            for s in range(nstage):
                mesh.forward(mesh_vars, dt, s)

    sync()
    dt0 = float(mesh.max_time_step(mesh_vars))
    t_setup = time.perf_counter()
    for n in range(a.warmup):
        step(n, dt0)
    sync()
    warm_s = time.perf_counter() - t_setup
    times = []
    dts = []
    for n in range(a.warmup, a.warmup + a.steps):
        dt = dt0 if a.dt_mode == "fixed" else dt0 * (0.95 + 0.05 * ((n * 7919) % 13) / 13.0)
        dts.append(dt)
        sync()
        t0 = time.perf_counter()
        step(n, dt)
        sync()
        times.append(time.perf_counter() - t0)
    finite = all(bool(torch.isfinite(v["hydro_u"]).all()) for v in mesh_vars)
    if getattr(a, "profile", 0) and a.profile > 0:  # speed-gate profiles: where the step time goes (CPU ops, CUDA kernels, graph launches)
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])
        n0 = a.warmup + a.steps
        sync()
        with profile(activities=acts) as prof:
            for n in range(n0, n0 + a.profile):
                dt = dt0 if a.dt_mode == "fixed" else dt0 * (0.95 + 0.05 * ((n * 7919) % 13) / 13.0)
                step(n, dt)
            sync()
        key = "cuda_time_total" if dev.type == "cuda" else "cpu_time_total"
        with open(a.out + ".prof.txt", "w") as pf:
            pf.write(f"profile of {a.profile} steps after the timed loop; sorted by {key}\n")
            pf.write(prof.key_averages().table(sort_by=key, row_limit=int(os.environ.get("SPEED_PROF_ROWS", "40"))))
            pf.write("\n\nsorted by cpu_time_total\n")
            pf.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=int(os.environ.get("SPEED_PROF_ROWS", "25"))))
    ts = sorted(times)
    med = ts[len(ts) // 2]
    rep = {
        "deck": a.deck, "device": str(dev), "precision": a.precision, "execution": a.execution,
        "dt_mode": a.dt_mode, "dt0": dt0, "rank": rank, "world_size": world, "driver": driver, "capability": capability,
        "blocks_in_process": len(blocks), "cells": cfg["geometry"]["cells"], "distribute": cfg.get("distribute"),
        "warmup": a.warmup, "steps": a.steps, "warmup_seconds": warm_s, "threads": a.threads,
        "median_s": med, "min_s": ts[0], "max_s": ts[-1], "p10_s": ts[len(ts) // 10], "p90_s": ts[(9 * len(ts)) // 10],
        "mean_s": sum(ts) / len(ts), "finite": finite, "times": times,
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "torch": torch.__version__, "snapy_file": snapy.__file__,
        "kintera": importlib.metadata.version("kintera") if dev else None,
        "native_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                          for p in (Path(snapy.__file__).parent / "lib").glob("libsnap*") if p.is_file()},
    }
    try:
        rep["execution_report"] = {k: int(v) for k, v in blocks[0].execution_report().items()}
    except Exception as e:  # before tree: no such API
        rep["execution_report"] = str(type(e).__name__)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"rank {rank}: {len(blocks)} blocks, median {med*1e3:.3f} ms/step (min {ts[0]*1e3:.3f}, max {ts[-1]*1e3:.3f}), finite={finite}", flush=True)


if __name__ == "__main__":
    main()
