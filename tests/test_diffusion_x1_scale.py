#!/usr/bin/env python3
"""An x1 profile of the kinematic diffusion coefficients, from Python and from YAML.

forcing.diffusion takes nu_scale_x1 / kappa_scale_x1 two ways: a Python tensor over the
block's x1 cell centres (ghosts included) set on DiffusionOptions before the MeshBlock is
built, or a YAML table {x1: [...], scale: [...]} in the x1 coordinate, interpolated onto
the cell centres. Each arm runs one full step of test_diffusion.yaml (nx1 = 16, reflecting
x1 walls) from a state with a velocity shear and a temperature gradient:
  1. a profile of ones, from Python and from YAML, gives the step of no profile bit for bit;
  2. a YAML table whose knots are the cell centres gives the step of the same values set
     from Python, bit for bit;
  3. a non-uniform profile changes the step (the profile is not ignored).

  python test_diffusion_x1_scale.py [--device cuda]
"""
import argparse
import math
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NX1 = 16


def config(table=None):
    with open(os.path.join(HERE, "test_diffusion.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["geometry"]["cells"]["nx1"] = NX1
    cfg["geometry"]["bounds"]["x1max"] = float(NX1)
    if table is not None:
        cfg["forcing"]["diffusion"]["nu_scale_x1"] = table
        cfg["forcing"]["diffusion"]["kappa_scale_x1"] = table
    return cfg


def make_block(device, table=None, cells=None):
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(table), f)
        tmp = f.name
    try:
        options = MeshBlockOptions.from_yaml(tmp)
    finally:
        os.unlink(tmp)
    if cells is not None:
        diffusion = options.hydro().diffusion()
        diffusion.nu_scale_x1(cells)
        diffusion.kappa_scale_x1(cells)
    block = MeshBlock(options)
    block.to(torch.device(device))
    return block


def cell_centres():
    """x1 cell centres of the block, ghosts included (as the Python profile is indexed)."""
    from snapy import MeshBlock, MeshBlockOptions

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config(), f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
    finally:
        os.unlink(tmp)
    return dict(block.named_buffers())["coord.x1v"].to(torch.float64).cpu()


def step(block):
    """One full step from a sheared, conducting state; returns hydro_u after it."""
    from snapy import kIDN, kIPR, kIV2

    w = dict(block.named_buffers())["hydro.D"].clone().zero_()
    x = dict(block.named_buffers())["coord.x1v"].to(w).view(1, 1, -1)
    w[kIDN] = 1.0 + 0.05 * x
    w[kIPR] = 1.0e5 * (1.0 + 0.01 * torch.sin(2.0 * math.pi * x / NX1))
    w[kIV2] = torch.sin(2.0 * math.pi * x / NX1)
    block_vars, _ = block.initialize({"hydro_w": w})
    dt = block.max_time_step(block_vars)
    for stage in range(len(block.module("intg").stages)):
        block.forward(block_vars, dt, stage)
    return block_vars["hydro_u"].clone()


def arm(failures, name, fn):
    try:
        msg = fn()
    except Exception as err:  # the feature missing is a failure, not a crash
        msg = "%s: %s" % (type(err).__name__, str(err).splitlines()[0] if str(err) else "")
    print("%-8s %s" % (name, "ok" if msg is None else "FAIL " + msg))
    if msg is not None:
        failures.append("%s: %s" % (name, msg))


def run(device):
    failures = []
    x1v = cell_centres()
    ones = torch.ones_like(x1v)
    ref = {}

    def unity():
        ref["none"] = step(make_block(device))
        by_api = step(make_block(device, cells=ones))
        by_yaml = step(make_block(device, table={"x1": [0.0, float(NX1)], "scale": [1.0, 1.0]}))
        if not torch.equal(by_api, ref["none"]):
            return "a profile of ones set from Python changed the step"
        if not torch.equal(by_yaml, ref["none"]):
            return "a YAML profile of ones changed the step"
        return None

    def yaml_equals_api():
        values = 1.0 + 0.5 * torch.cos(x1v)  # smooth, positive, on every centre
        by_api = step(make_block(device, cells=values))
        table = {"x1": [float(v) for v in x1v], "scale": [float(v) for v in values]}
        by_yaml = step(make_block(device, table=table))
        if not torch.equal(by_yaml, by_api):
            return "YAML table on the cell centres != the same profile from Python (max |d| %g)" % (
                (by_yaml - by_api).abs().max().item())
        if "none" in ref and torch.equal(by_api, ref["none"]):
            return "a non-uniform profile left the step unchanged"
        return None

    def inference():
        # an inference tensor keeps no version counter, so a later in-place
        # write under torch.inference_mode() could not be detected: refused
        with torch.inference_mode():
            cells = torch.ones_like(x1v)
        try:
            make_block(device, cells=cells)
        except RuntimeError as err:
            if "inference tensor" not in str(err):
                return "refused for another reason: %s" % str(err).splitlines()[0]
        else:
            return "an inference tensor was accepted as a profile"
        # a YAML table parsed under inference mode is an ordinary tensor
        with torch.inference_mode():
            make_block(device, table={"x1": [0.0, float(NX1)], "scale": [1.0, 2.0]})
        return None

    def outside_writes():
        # writes that bump no version counter: through the NumPy array behind
        # torch.from_numpy, and through the .data alias; step() calls
        # max_time_step before forward
        import numpy as np

        for how in ("from_numpy", ".data"):
            array = np.ones(x1v.numel())
            cells = torch.from_numpy(array) if how == "from_numpy" else torch.ones_like(x1v)
            block = make_block(device, cells=cells)
            step(block)
            if how == "from_numpy":
                array[3] = 4.0
            else:
                cells.data[3] = 4.0
            try:
                step(block)
            except RuntimeError as err:
                if "profile" not in str(err):
                    return "%s: refused for another reason: %s" % (how, str(err).splitlines()[0])
                continue
            return "%s: a write after the build was not refused" % how
        return None

    arm(failures, "unity", unity)
    arm(failures, "yaml-api", yaml_equals_api)
    arm(failures, "inference", inference)
    arm(failures, "outside", outside_writes)
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125
    failures = run(args.device)
    for msg in failures:
        print("FAIL (%s): %s" % (args.device, msg))
    if failures:
        return 1
    print("PASS (%s): x1 diffusion profiles from Python and YAML" % args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
