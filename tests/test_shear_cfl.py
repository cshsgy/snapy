#!/usr/bin/env python3
"""A horizontal wind that jumps by more than cs across an x1 face bounds dt at
shear_cfl * cs_f * dx_h / (|v_h(i)| |v_h(i+1)|); no such face, or shear-cfl 0, leaves dt alone.

  python test_shear_cfl.py [--device cuda]
"""
import argparse
import os
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NX1, NX2, L1, L2 = 8, 8, 0.08, 8.0  # dx1 = 0.01, dx2 = 1


def time_step(base_yaml, device, shear_cfl, v_lo, v_hi):
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1

    with open(base_yaml) as f:
        config = yaml.safe_load(f)
    config["geometry"]["bounds"].update({"x1max": L1, "x2max": L2})
    config["geometry"]["cells"].update({"nx1": NX1, "nx2": NX2, "nx3": 1})
    config["integration"].update({"cfl": 0.5, "implicit-scheme": 9, "shear-cfl": shear_cfl})
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir=os.getcwd()) as f:
        yaml.safe_dump(config, f)
        tmp = f.name
    try:
        block = MeshBlock(MeshBlockOptions.from_yaml(tmp))
        block.to(torch.device(device))
        eos = next(m for _, m in block.named_modules() if type(m).__name__ == "IdealMoist")
        w = dict(block.named_buffers())["hydro.D"].clone().zero_()
        w[kIDN] = 1.0
        w[kIPR] = 1.0e5
        w[kICY] = 1.0e-3
        half = w.shape[-1] // 2
        w[kIV1 + 1, ..., :half] = v_lo   # x2-wind below the mid-column face
        w[kIV1 + 1, ..., half:] = v_hi   # and above it
        cs = eos.compute("WA->L", [w, eos.compute("W->A", [w])]).max().item()
        block_vars, _ = block.initialize({"hydro_w": w})
        return block.max_time_step(block_vars), cs
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--yaml", default=os.path.join(HERE, "test_fix_vapor_reports_failure.yaml"))
    args = ap.parse_args()
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 0

    cfl, dx2 = 0.5, L2 / NX2
    failures = []

    def check(name, got, want):
        print("%-40s dt=%.9e  expected=%.9e" % (name, got, want))
        if abs(got - want) > 1.0e-9 * want:
            failures.append(name)

    _, cs = time_step(args.yaml, args.device, 0.0, 0.0, 0.0)
    v = 3.0 * cs                       # a supersonic reversal across the mid face: jump 6 cs
    # the bound binds at exactly shear_cfl * cs * dx2 / v^2 and scales with the knob
    dt, _ = time_step(args.yaml, args.device, 0.2, v, -v)
    check("jump 6cs, shear_cfl 0.2", dt, 0.2 * cs * dx2 / (v * v))
    dt, _ = time_step(args.yaml, args.device, 0.1, v, -v)
    check("jump 6cs, shear_cfl 0.1", dt, 0.1 * cs * dx2 / (v * v))
    # switched off: the explicit x2 bound is what it was
    dt, _ = time_step(args.yaml, args.device, 0.0, v, -v)
    check("jump 6cs, shear_cfl 0 (off)", dt, cfl * dx2 / (v + cs))
    # a uniform supersonic wind has no jump: untouched
    dt, _ = time_step(args.yaml, args.device, 0.2, v, v)
    check("uniform 3cs, shear_cfl 0.2", dt, cfl * dx2 / (v + cs))
    # a subsonic jump (0.6 cs) does not qualify
    dt, _ = time_step(args.yaml, args.device, 0.2, 0.3 * cs, -0.3 * cs)
    check("jump 0.6cs, shear_cfl 0.2", dt, cfl * dx2 / (0.3 * cs + cs))

    if failures:
        print("FAIL (%s): %s" % (args.device, ", ".join(failures)))
        return 1
    print("PASS (%s): supersonic shear faces bounded, everything else untouched" % args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
