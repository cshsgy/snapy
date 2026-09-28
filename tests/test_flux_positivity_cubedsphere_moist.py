#!/usr/bin/env python3
"""Hydro SPECIES flux-positivity limiter across a cubed-sphere PANEL SEAM.

Companion of test_flux_positivity_cubedsphere.py, on the same six-panel deck, with the dry
EOS swapped for ideal-moist and the vapor <=> cloud pair of test_flux_positivity.yaml added
(ny = 2). The scalar seam test cannot see the hydro channel: with ny = 0 the hydro limiter
(hydro_forward.cpp 4.C) never runs, so its theta ghost fill is untested there. Here the vapor
carries the complement 1 - hat of the scalar test's hat, so going downwind it steps from empty
to full on the +X/+Y seam: the empty edge cells just upwind of the seam are drained by the
reconstruction undershoot and get theta = 0 inside the hat's latitude band, and the neighbour
panel takes that theta from its ghost as the donor of the seam face. A ghost theta that is not
the neighbour's edge-cell value (an interpolated fill) leaks species mass at the seam. The
pressure is raised so the column is warm (~313 K) and everywhere unsaturated, the cloud
channel stays zero and
the saturation adjustment moves nothing; the vapor+cloud total must hold to round-off in both
arms, and the limited arm must fire and keep every species >= 0.

  python test_flux_positivity_cubedsphere_moist.py [--device cuda] [--yaml PATH]
"""
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import torch
import yaml

DRIFT_TOL = 1e-13
SPECIES_MIN = -1e-15
QMAX = 0.02  # species mass fraction inside the hat

PRES = 1.1e6  # with the deck's H/He air at rho = 1: T ~ 313 K, vapor well below saturation
MOIST_SPECIES = [
    {"name": "vapor", "composition": {"H": 2, "O": 1}, "cv_R": 3.5},
    {"name": "cloud", "composition": {"H": 2, "O": 1}, "cv_R": 9.0, "u0_R": -3430.0},
]
MOIST_REACTIONS = [
    {"equation": "vapor <=> cloud", "type": "nucleation",
     "rate-constant": {"formula": "h2o_ideal"}},
]


def run_arm(yaml_file: str, limiter: bool, device: str):
    import snapy
    from snapy import Mesh, MeshOptions, kIDN, kIV1, kIV2, kIV3, kIPR, kICY

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    assert cfg["distribute"]["layout"] == "cubed-sphere", "deck must be the six-panel one"
    cfg["species"] = cfg["species"][:1] + MOIST_SPECIES
    cfg["reactions"] = MOIST_REACTIONS
    cfg["dynamics"]["equation-of-state"]["type"] = "ideal-moist"
    cfg["dynamics"]["equation-of-state"]["limiter"] = limiter
    if not limiter:
        cfg["scalar"].pop("upper-bound", None)
    nx = cfg["geometry"]["cells"]["nx2"]
    nlim = cfg["integration"]["nlim"]

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
    assert len(blocks) == 6, "expected six panels, got %d" % len(blocks)
    for block in blocks:
        block.to(torch.device(device))

    lon0, half, band = math.radians(15.0), math.radians(30.0), math.radians(30.0)
    mesh_vars, faces = [], set()
    for ib, block in enumerate(blocks):
        coord = block.module("coord")
        beta, alpha = torch.meshgrid(coord.buffer("x3v"), coord.buffer("x2v"), indexing="ij")
        face_id = int(block.get_layout().loc_of(ib)[2])
        faces.add(face_id)
        lon, lat = snapy.coord.cs_ab_to_lonlat(snapy.coord.get_cs_face_name(face_id), alpha, beta)
        bufs = dict(block.named_buffers())
        w = bufs["hydro.D"].clone().zero_()
        r = bufs["scalar.D"].clone().zero_()
        ny = w.size(0) - kICY
        assert ny == 2, "expected vapor+cloud, got ny=%d" % ny
        vel = torch.zeros((3,) + tuple(alpha.shape), dtype=torch.float64, device=alpha.device)
        vel[2] = 10.0 * torch.cos(lat)  # (v_r, v_theta, v_phi) -> contravariant, in place
        snapy.coord.cs_sph_to_contra_(vel, alpha, beta, face_id)
        w[kIDN] = 1.0
        w[kIPR] = PRES
        w[kIV1] = vel[0].unsqueeze(-1)
        w[kIV2] = vel[1].unsqueeze(-1)
        w[kIV3] = vel[2].unsqueeze(-1)
        dlon = torch.remainder(lon - lon0 + math.pi, 2 * math.pi) - math.pi
        hat = ((dlon.abs() < half) & (lat.abs() < band)).to(torch.float64).unsqueeze(-1)
        w[kICY] = QMAX * (1.0 - hat)  # the cloud channel stays identically zero
        r[0] = hat
        r[1] = 1.0 - hat
        mesh_vars.append({"hydro_w": w, "scalar_r": r})
    assert faces == set(range(6)), "panels do not cover the six faces: %s" % sorted(faces)
    assert str(mesh_vars[0]["hydro_w"].device).startswith(device)
    mesh_vars, t = mesh.initialize(mesh_vars)

    def species():
        """Per-panel vapor+cloud totals (6,) and the minimum species partial density."""
        tot = torch.zeros(6, dtype=torch.float64)
        smin = 1e300
        for ib, block in enumerate(blocks):
            sl = block.part((0, 0, 0), False)[1:]
            vol = block.module("coord").cell_volume()[sl]
            u = mesh_vars[ib]["hydro_u"][(slice(kICY, kICY + 2),) + tuple(sl)]
            tot[ib] = (u * vol).sum().cpu()
            smin = min(smin, float(u.min()))
        return tot, smin

    panel0, smin = species()
    tot0 = panel0.sum(0)
    intg = blocks[0].module("intg")
    for cycle in range(1, nlim + 1):
        mesh.set_cycle(cycle)
        dt = mesh.max_time_step(mesh_vars)
        for stage in range(len(intg.stages)):
            mesh.forward(mesh_vars, dt, stage)
            smin = min(smin, species()[1])
        assert mesh.check_redo(mesh_vars) == 0, "step rejected at cycle %d" % cycle
    panel1, _ = species()
    tot1 = panel1.sum(0)
    hits = sum(int(dict(b.named_buffers())["hydro.positivity_hits"].item()) for b in blocks)
    drift = abs((tot1 - tot0) / tot0).item()
    # species mass that changed panel: the transport really crossed the seams
    moved = ((panel1 - panel0).abs().sum() / tot0).item()
    return {"drift": drift, "min": smin, "hits": hits, "moved": moved}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument(
        "--yaml",
        default=str(Path(__file__).resolve().parent / "test_flux_positivity_cubedsphere.yaml"),
    )
    args = ap.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("SKIP: cuda requested but not available")
        return 125

    base = run_arm(args.yaml, limiter=False, device=args.device)
    lim = run_arm(args.yaml, limiter=True, device=args.device)
    for name, arm in (("base", base), ("limited", lim)):
        print("%-8s drift=%.3e min=%+.3e hits=%d moved=%.3e"
              % (name, arm["drift"], arm["min"], arm["hits"], arm["moved"]))

    failures = []
    if not base["drift"] < DRIFT_TOL:
        failures.append("base arm does not conserve species across the seam: %g" % base["drift"])
    if not lim["drift"] < DRIFT_TOL:
        failures.append(
            "limited arm leaks species across the seam: %g"
            " -- theta ghost not the donor's?" % lim["drift"])
    if not lim["hits"] > 0:
        failures.append("hydro limiter never fired: the test no longer bites")
    if not lim["min"] >= SPECIES_MIN:
        failures.append("limited arm left a species negative: min=%g" % lim["min"])
    if not lim["moved"] > 1e-6:
        failures.append(
            "no species mass changed panel (moved=%g): the seams are not crossed" % lim["moved"])
    for msg in failures:
        print("FAIL:", msg)
    if failures:
        sys.exit(1)
    print("### flux positivity cubed-sphere moist seam test passed. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
