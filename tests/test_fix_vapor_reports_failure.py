#!/usr/bin/env python3
"""An unrepairable vapour column must raise from eos.compute("U->W") on every device.

  python test_fix_vapor_reports_failure.py [--device cuda]
"""
import argparse
import os
import sys

import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--yaml", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "test_fix_vapor_reports_failure.yaml"))
    args = ap.parse_args(argv)
    if args.device.startswith("cuda") and (os.environ.get("SNAPY_BUILD_CUDA", "1") == "0" or not torch.cuda.is_available()):
        print("SKIP: cuda requested but not available")
        return 125

    from snapy import MeshBlock, MeshBlockOptions, kICY

    op = MeshBlockOptions.from_yaml(args.yaml)
    block = MeshBlock(op)
    block.to(torch.device(args.device))
    eos = next(m for _, m in block.named_modules() if type(m).__name__ == "IdealMoist")

    bufs = dict(block.named_buffers())
    w = bufs["hydro.D"].clone().zero_()  # (nvar, nc3, nc2, nc1)
    w[0] = 1.0
    w[4] = 1.0e5
    w[kICY] = 1.0e-3        # vapour
    u = eos.compute("W->U", [w])

    # a benign state must pass
    eos.compute("U->W", [u.clone()])

    # every cell of ONE interior column goes vapour-negative: net deficit, unrepairable
    u_bad = u.clone()
    k, j = u.size(1) // 2, u.size(2) // 2   # an interior column (a dim of size 1 has no ghosts)
    u_bad[kICY, k, j, :] = -1.0e-6
    try:
        eos.compute("U->W", [u_bad])
    except RuntimeError as e:
        assert "Failed to fix vapor" in str(e), str(e)
        print("PASS (%s): unrepairable column reported" % args.device)
        return 0
    print("FAIL (%s): unrepairable vapour column was NOT reported" % args.device)
    return 1


if __name__ == "__main__":
    sys.exit(main())
