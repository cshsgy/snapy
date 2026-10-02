#!/usr/bin/env python3
"""check_redo must reject a step that floored a cell and restore hydro_u AND hydro_w.

The floor is planted in hydro_u only: a detector reading the (one stage stale) hydro_w
would pass it. With the limiter off nothing fills a NaN before the detector, and a NaN
fails every comparison, so a second arm plants one. A third arm runs a six-block Mesh with the floor planted in ONE block:
the decision is per process, so every block must roll back and the time step must halve.

  python test_check_redo_floor.py [--device cuda]
"""
import argparse
import math
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def block_arm(device, yaml_file):
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

    block = MeshBlock(MeshBlockOptions.from_yaml(yaml_file))
    block.to(torch.device(device))
    eos = next(m for _, m in block.named_modules() if type(m).__name__ == "IdealMoist")
    density_floor = eos.options.density_floor()

    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    w[kIDN] = 1.0
    w[kIPR] = 1.0e5
    w[kICY] = 1.0e-3
    block_vars, _ = block.initialize({"hydro_w": w})

    u0 = block_vars["hydro_u"].clone()
    dt = block.max_time_step(block_vars)
    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt, stage)
    if block.check_redo(block_vars) != 0:
        return "healthy step was rejected"

    k, j, i = u0.size(1) // 2, u0.size(2) // 2, u0.size(3) // 2
    block_vars["hydro_u"][kIDN, k, j, i] = density_floor  # at the floor, not below it
    block_vars["hydro_u"][kICY:, k, j, i] = 0.0
    w_stale_min = block_vars["hydro_w"][kIDN].min().item()

    err = block.check_redo(block_vars)
    if err != 1:
        return "floored cell not rejected (err=%d; stale hydro_w min rho=%g)" % (err, w_stale_min)
    if not torch.equal(block_vars["hydro_u"], u0):
        return "hydro_u not restored to the pre-step state"
    w_expect = eos.compute("U->W", [u0.clone()])
    if not torch.equal(block_vars["hydro_w"], w_expect):
        return "hydro_w not recomputed from the restored hydro_u"
    if block.check_redo(block_vars) != 0:
        return "restored state was rejected"
    return None


def nan_arm(device, yaml_file):
    """Limiter off, so nothing fills a NaN before the detector sees it."""
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    cfg["dynamics"]["equation-of-state"]["limiter"] = False
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))

    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    w[kIDN] = 1.0
    w[kIPR] = 1.0e5
    w[kICY] = 1.0e-3
    block_vars, _ = block.initialize({"hydro_w": w})
    u0 = block_vars["hydro_u"].clone()
    dt = block.max_time_step(block_vars)
    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt, stage)
    if block.check_redo(block_vars) != 0:
        return "healthy step was rejected with the limiter off"

    k, j, i = u0.size(1) // 2, u0.size(2) // 2, u0.size(3) // 2
    block_vars["hydro_u"][kIDN, k, j, i] = float("nan")
    err = block.check_redo(block_vars)
    if err != 1:
        return "a NaN density was not rejected (err=%d)" % err
    if not torch.equal(block_vars["hydro_u"], u0):
        return "hydro_u not restored after the NaN"
    return None


def mesh_arm(device, yaml_file, nan=False):
    """Six cubed-sphere panels in one process, solid-body wind, floor (or a NaN velocity) planted in block 3 only."""
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1, kIV2, kIV3

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("scalar", None)
    density_floor = float(cfg["dynamics"]["equation-of-state"]["density-floor"])
    nx = cfg["geometry"]["cells"]["nx2"]
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(nx, nx)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    for block in blocks:
        block.to(torch.device(device))

    mesh_vars = []
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        beta, alpha = torch.meshgrid(coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij")
        face_id = int(block.get_layout().loc_of(ib)[2])
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face_id), alpha, beta)
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64, device=alpha.device)
        vel[2] = 10.0 * torch.cos(lat)
        snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face_id)
        w[kIDN] = 1.0
        w[kIPR] = 1.0e5
        w[kIV1] = vel[0].unsqueeze(-1).to(w.device)
        w[kIV2] = vel[1].unsqueeze(-1).to(w.device)
        w[kIV3] = vel[2].unsqueeze(-1).to(w.device)
        mesh_vars.append({"hydro_w": w})
    mesh_vars, _ = mesh.initialize(mesh_vars)

    u0 = [mv["hydro_u"].clone() for mv in mesh_vars]
    intg = blocks[0].module("intg")
    mesh.set_cycle(1)
    dt0 = mesh.max_time_step(mesh_vars)
    for stage in range(len(intg.stages)):
        mesh.forward(mesh_vars, dt0, stage)
    if mesh.check_redo(mesh_vars) != 0:
        return "healthy mesh step was rejected"
    moved = sum(int(not torch.equal(mv["hydro_u"], u)) for mv, u in zip(mesh_vars, u0))
    if moved != len(blocks):
        return "the step left %d of %d blocks unchanged; a restore would be vacuous" % (len(blocks) - moved, len(blocks))

    ub = mesh_vars[3]["hydro_u"]
    k, j, i = ub.size(1) // 2, ub.size(2) // 2, ub.size(3) // 2
    if nan:  # zeroed by the limiter in floor_hit's conversion: only its NaN mark rejects the step
        ub[kIV1, k, j, i] = float("nan")
    else:
        ub[kIDN, k, j, i] = density_floor
    err = mesh.check_redo(mesh_vars)
    if err != 1:
        what = "a NaN velocity" if nan else "a floor"
        return "%s in one block of six not rejected (err=%d)" % (what, err)
    for ib, (mv, u) in enumerate(zip(mesh_vars, u0)):
        if not torch.equal(mv["hydro_u"], u):
            return "block %d not restored after a repair in block 3" % ib
    dt1 = mesh.max_time_step(mesh_vars)
    if abs(dt1 - 0.5 * dt0) > 1.0e-12 * dt0:
        return "time step after the redo is %g, expected half of %g" % (dt1, dt0)
    if mesh.check_redo(mesh_vars) != 0:
        return "restored mesh was rejected"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--yaml", default=os.path.join(HERE, "test_fix_vapor_reports_failure.yaml"))
    ap.add_argument("--mesh-yaml", default=os.path.join(HERE, "test_flux_positivity_cubedsphere.yaml"))
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0

    failures = []
    for name, arm, yf in (("block", block_arm, args.yaml), ("nan", nan_arm, args.yaml),
                          ("mesh", mesh_arm, args.mesh_yaml),
                          ("meshnan", lambda d, y: mesh_arm(d, y, nan=True), args.mesh_yaml)):
        msg = arm(args.device, yf)
        print("%-6s %s" % (name, "PASS" if msg is None else "FAIL: " + msg))
        if msg is not None:
            failures.append(name)
    if failures:
        print("FAIL (%s): %s" % (args.device, ", ".join(failures)))
        return 1
    print("PASS (%s): floored step rejected, every block restored, time step halved" % args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
