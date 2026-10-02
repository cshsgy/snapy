#!/usr/bin/env python3
"""An implicit direction keeps its ADVECTIVE time-step bound.

  python test_implicit_advection_cfl.py [--device cuda]
"""
import argparse
import os
import sys
import tempfile

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
NX1, NX2, L1, L2 = 8, 8, 0.08, 8.0  # dx1 = 0.01, dx2 = 1


def time_step(base_yaml, device, scheme, advection_cfl, v1):
    from snapy import MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR, kIV1

    with open(base_yaml) as f:
        config = yaml.safe_load(f)
    config["geometry"]["bounds"].update({"x1max": L1, "x2max": L2})
    config["geometry"]["cells"].update({"nx1": NX1, "nx2": NX2, "nx3": 1})
    config["integration"].update({"cfl": 0.5, "implicit-scheme": scheme})
    if advection_cfl is not None:
        config["integration"]["implicit-advection-cfl"] = advection_cfl
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
        w[kIV1] = v1
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

    cfl, dx1, dx2, v1 = 0.5, L1 / NX1, L2 / NX2, 100.0
    failures = []

    def check(name, got, want):
        print("%-34s dt=%.9e  expected=%.9e" % (name, got, want))
        if abs(got - want) > 1.0e-12 * want:
            failures.append(name)

    # x1 implicit: the advective bound binds at exactly advection_cfl * dx1 / |v1|
    dt, cs = time_step(args.yaml, args.device, 9, 1.0, v1)
    check("implicit x1, advection_cfl 1.0", dt, 1.0 * dx1 / v1)
    dt, _ = time_step(args.yaml, args.device, 9, 0.25, v1)
    check("implicit x1, advection_cfl 0.25", dt, 0.25 * dx1 / v1)
    # at rest the implicit direction is unbounded and the explicit x2 bound is unchanged
    dt, cs = time_step(args.yaml, args.device, 9, 1.0, 0.0)
    check("implicit x1 at rest -> explicit x2", dt, cfl * dx2 / cs)
    # explicit everywhere: the acoustic x1 bound is back
    dt, cs = time_step(args.yaml, args.device, 0, None, v1)
    check("explicit x1", dt, cfl * dx1 / (v1 + cs))
    # the knob without an implicit direction is a config error, not a silent no-op
    try:
        time_step(args.yaml, args.device, 0, 1.0, v1)
        failures.append("implicit-advection-cfl accepted with implicit-scheme 0")
    except RuntimeError:
        print("%-34s raises, as it must" % "explicit x1 + advection_cfl key")

    if failures:
        print("FAIL (%s): %s" % (args.device, ", ".join(failures)))
        return 1
    print("PASS (%s): implicit direction bounded on advection, explicit ones untouched" % args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
