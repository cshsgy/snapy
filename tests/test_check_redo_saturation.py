#!/usr/bin/env python3
"""check_redo must redo a step whose saturation adjustment left a cell unadjusted.

The step starts from one strongly supersaturated NH4SH cell (the cell of kintera's
test_saturation_failure_reported.py); with max-iter 1 the adjustment leaves cells unadjusted (at
least one; 11 on the reference CPU run). kintera only counts the failures, so check_redo must read
the count, redo the step with the saturation cause alone, and drain it so that the restored
state is accepted. A control step with the default max-iter must be accepted: no other cause
fires in this scenario. A failure counted between steps (a ThermoY call from Python) is not the
step's: a clean step after it must be accepted. A last arm runs a two-block Mesh with the cell in
block 1 only, so Mesh::check_redo must read the count of a block other than the first.

  python test_check_redo_saturation.py [--device cuda]
"""
import argparse
import ctypes
import os
import re
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# NH4SH 150x supersaturated, no cloud yet: T, rho, then NH3, H2S, NH3(s), NH4SH(s) mass fractions
CELL = (208.72174072916363, 0.8801790168247441, 5.279766354738258e-04, 1.428311049569326e-02,
        1.0099328092205351e-07, 0.)


def make_block(device, yaml_file, max_iter):
    from snapy import MeshBlock, MeshBlockOptions

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    eos = cfg["dynamics"]["equation-of-state"]
    if max_iter is None:
        eos.pop("max-iter", None)
    else:
        eos["max-iter"] = max_iter
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    block.to(torch.device(device))
    return block


def modules(block):
    mods = dict(block.named_modules())
    eos = next(m for m in mods.values() if type(m).__name__ == "IdealMoist")
    return eos, mods["hydro.eos.thermo"]


def fields(block, cell=True):
    """T, rho uniform, H2S everywhere, NH3 in one interior cell only (cell=False: none)."""
    from snapy import kICY, kIDN

    u = dict(block.named_buffers())["hydro.D"].clone().zero_()
    temp = torch.full_like(u[kIDN], CELL[0])
    rho = torch.full_like(u[kIDN], CELL[1])
    y = torch.zeros_like(u[kICY:])
    y[1] = CELL[3]
    if cell:
        k, j, i = u.size(1) // 2, u.size(2) // 2, u.size(3) // 2
        y[:, k, j, i] = torch.tensor(CELL[2:], dtype=y.dtype, device=y.device)
    return u, temp, rho, y


def initial_state(block, cell=True):
    """At rest, with fields(block, cell)."""
    from snapy import kICY, kIDN, kIPR

    eos, thermo = modules(block)
    u, temp, rho, y = fields(block, cell)
    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    u[kIDN] = rho * (1.0 - y.sum(0))
    u[kICY:] = rho * y
    u[kIPR] = ie
    return {"hydro_w": eos.compute("U->W", [u])}


def initialize(block, cell=True):
    block_vars, _ = block.initialize(initial_state(block, cell))
    modules(block)[1].take_saturation_adjustment_failures()  # nothing before the step counts
    return block_vars


def step(block, block_vars):
    dt = block.max_time_step(block_vars)
    for stage in range(len(block.intg.stages)):
        block.forward(block_vars, dt, stage)


def check_redo_causes(block, block_vars):
    """check_redo's return value and the causes its redo message names (None: no message).

    block may be a MeshBlock or a Mesh (then block_vars is the list of block variables)."""
    sys.stdout.flush()
    ctypes.CDLL(None).fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile("w+") as f:
        os.dup2(f.fileno(), 1)
        try:
            err = block.check_redo(block_vars)
        finally:
            sys.stdout.flush()
            ctypes.CDLL(None).fflush(None)  # snapy's log writes std::cout, unflushed
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        out = f.read()
    sys.stdout.write(out)
    m = re.search(r"\(causes:([a-z ]*)\)", out)
    return err, (m.group(1).split() if m else None)


def run(device, yaml_file):
    failures = []

    # control: the adjustment converges, so nothing may redo this step
    block = make_block(device, yaml_file, None)
    block_vars = initialize(block)
    step(block, block_vars)
    n = modules(block)[1].take_saturation_adjustment_failures()
    err, causes = check_redo_causes(block, block_vars)
    print("control  failed adjustments %d, check_redo %d, causes %s" % (n, err, causes))
    if n != 0 or err != 0:
        failures.append("control step (default max-iter) not clean: %d failures, check_redo %d, "
                        "causes %s" % (n, err, causes))

    # the same step with max-iter 1 leaves the NH4SH cell unadjusted
    block = make_block(device, yaml_file, 1)
    block_vars = initialize(block)
    step(block, block_vars)
    n = modules(block)[1].take_saturation_adjustment_failures()
    print("max-iter 1: failed adjustments in the step %d" % n)
    if n < 1:
        failures.append("max-iter 1 left no cell unadjusted; the scenario is vacuous")

    # the same again, read only by check_redo
    block = make_block(device, yaml_file, 1)
    block_vars = initialize(block)
    u0 = block_vars["hydro_u"].clone()
    step(block, block_vars)
    err, causes = check_redo_causes(block, block_vars)
    left = modules(block)[1].take_saturation_adjustment_failures() if err != 1 else 0
    print("redo     check_redo %d, causes %s, count left undrained %d" % (err, causes, left))
    if err != 1:
        failures.append("check_redo == %d despite %d failed saturation adjustment(s) in the step"
                        % (err, max(n, left)))
    elif causes != ["saturation"]:
        failures.append("redo causes %s, expected saturation alone" % causes)
    elif not torch.equal(block_vars["hydro_u"], u0):
        failures.append("hydro_u not restored to the pre-step state")
    else:
        err, causes = check_redo_causes(block, block_vars)
        print("again    check_redo %d, causes %s" % (err, causes))
        if err != 0:
            failures.append("a drained count redid the step again (check_redo %d, causes %s)"
                            % (err, causes))
    failures += between_steps_arm(device, yaml_file)
    failures += mesh_arm(device, yaml_file)
    return failures


def between_steps_arm(device, yaml_file):
    """A failed adjustment of the block's own ThermoY between steps, then a step with no NH3 (so
    nothing to adjust): the step must be accepted, the stale count discarded at its start. The
    call takes the interior with the NH3 cell, the shape advance_local passes: kintera reuses its
    warm-start buffers without checking their shape."""
    block = make_block(device, yaml_file, 1)
    block_vars = initialize(block, cell=False)
    thermo = modules(block)[1]
    _, temp, rho, y = fields(block)
    sl = block.part((0, 0, 0), False)[1:]
    temp, rho, y = temp[sl], rho[sl].contiguous(), y[(slice(None),) + sl].contiguous()
    ie = thermo.compute("VT->U", [thermo.compute("DY->V", [rho, y]), temp])
    diag = torch.zeros(rho.shape + (1,), dtype=rho.dtype, device=rho.device)
    thermo.forward(rho, ie, y, False, diag)
    nbad = int((diag < 0).sum())
    print("between  adjustment before the step: %d cell(s) failed" % nbad)
    if nbad < 1:
        return ["max-iter 1 left the NH3 cell adjusted; the between-steps arm is vacuous"]
    step(block, block_vars)
    err, causes = check_redo_causes(block, block_vars)
    print("between  clean step after it: check_redo %d, causes %s" % (err, causes))
    if err != 0:
        return ["a failure counted before the step redid a clean step (check_redo %d, causes %s)"
                % (err, causes)]
    return []


def mesh_arm(device, yaml_file):
    """Two blocks in one process (nb2 = 2), the NH4SH cell in block 1 only: Mesh::check_redo must
    redo both blocks with the saturation cause alone, then accept the restored state."""
    from snapy import Mesh, MeshOptions

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    cfg["geometry"]["cells"]["nx2"] = 8
    cfg["distribute"]["nb2"] = 2
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(cfg, f)
        tmp = f.name
    try:
        options = MeshOptions.from_yaml(tmp)
        options.blocks_per_process(2)
        mesh = Mesh(options)
    finally:
        os.unlink(tmp)
    blocks = list(mesh.blocks)
    for block in blocks:
        block.to(torch.device(device))
    mesh_vars, _ = mesh.initialize([initial_state(b, cell=(ib == 1)) for ib, b in enumerate(blocks)])
    for block in blocks:
        modules(block)[1].take_saturation_adjustment_failures()
    u0 = [mv["hydro_u"].clone() for mv in mesh_vars]

    mesh.set_cycle(1)
    dt = mesh.max_time_step(mesh_vars)
    for stage in range(len(blocks[0].module("intg").stages)):
        mesh.forward(mesh_vars, dt, stage)
    err, causes = check_redo_causes(mesh, mesh_vars)
    print("mesh     check_redo %d, causes %s (2 blocks, the cell in block 1)" % (err, causes))
    if err != 1:
        return ["Mesh::check_redo == %d with an unadjusted cell in block 1" % err]
    if causes != ["saturation"]:
        return ["Mesh redo causes %s, expected saturation alone" % causes]
    for ib, (mv, u) in enumerate(zip(mesh_vars, u0)):
        if not torch.equal(mv["hydro_u"], u):
            return ["block %d not restored after the Mesh redo" % ib]
    err, causes = check_redo_causes(mesh, mesh_vars)
    print("mesh     again check_redo %d, causes %s" % (err, causes))
    if err != 0:
        return ["a drained count redid the Mesh step again (check_redo %d, causes %s)" % (err, causes)]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--yaml", default=os.path.join(HERE, "test_check_redo_saturation.yaml"))
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125

    failures = run(args.device, args.yaml)
    for msg in failures:
        print("FAIL (%s): %s" % (args.device, msg))
    if failures:
        return 1
    print("PASS (%s): an unadjusted cell redoes the step once, with the saturation cause alone"
          " (MeshBlock and Mesh)"
          % args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
