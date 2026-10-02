#!/usr/bin/env python3
"""One moist step of the VIC on CPU and CUDA.

implicit-scheme 1 and 9 each differ from scheme 0 (so the correction ran),
and the two devices agree. Exits 125 when CUDA is not available.

  python test_vic_moist_device.py
"""
import os
import sys
from pathlib import Path

import torch
import yaml

import test_tracer_dry_convention as tracer

# Measured on sm_90 (H100, CUDA_VISIBLE_DEVICES=0), one step, float64:
# max|cpu - cuda| = 5.821e-11 for both scheme 1 and scheme 9. Bound is ~17x that.
DEVICE_TOL = 1.0e-9
# Scheme 0 leaves the implicit correction off. The correction has to move
# the state by more than this or the comparison is vacuous.
VIC_FLOOR = 1.0e-8


def hydro(scheme, device, base):
    cfg = yaml.safe_load(yaml.safe_dump(base))
    cfg["integration"]["implicit-scheme"] = scheme
    state, _, _ = tracer.run(cfg, device, moving=True, nlim=1)
    return state["hydro_u"].detach().to(dtype=torch.float64).cpu()


def main():
    if os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available():
        print("SKIP: cuda requested but not available")
        return 125

    with open(Path(__file__).resolve().parent / "test_tracer_dry_convention.yaml") as f:
        base = yaml.safe_load(f)

    got = {}
    for scheme in (0, 1, 9):
        for device in ("cpu", "cuda"):
            got[(scheme, device)] = hydro(scheme, device, base)

    failures = []
    for scheme in (1, 9):
        for device in ("cpu", "cuda"):
            moved = float((got[(scheme, device)] - got[(0, device)]).abs().max())
            print(f"scheme {scheme} {device}: max|u - scheme0|={moved:.3e}")
            if not moved > VIC_FLOOR:
                failures.append(
                    f"scheme {scheme} on {device} did not leave scheme 0 (max {moved:.3e})"
                )
        agree = float((got[(scheme, "cpu")] - got[(scheme, "cuda")]).abs().max())
        print(f"scheme {scheme}: max|cpu - cuda|={agree:.3e}")
        if not agree < DEVICE_TOL:
            failures.append(
                f"scheme {scheme}: cpu and cuda differ by {agree:.3e}, bound {DEVICE_TOL:.3e}"
            )

    for msg in failures:
        print("FAIL:", msg)
    if failures:
        return 1
    print("### VIC moist CPU/CUDA step agrees. ###")
    return 0


if __name__ == "__main__":
    sys.exit(main())
