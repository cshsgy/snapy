#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


SKIP_CODE = 125


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-type", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args()

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        print("Skipping exchange test: torchrun not found")
        return SKIP_CODE

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print("Skipping exchange test: CUDA runtime is unavailable")
            return SKIP_CODE

    build_dir = Path(args.build_dir).resolve()
    executable = build_dir / "tests" / f"test_exchange.{args.build_type}"
    env = os.environ.copy()
    env.update(
        {
            "BACKEND": args.backend,
            "DEVICE": args.device,
            "BLOCKS_PER_PROCESS": "3",
            "EXPECT_LOCAL_NEIGHBOR": "1",
            "EXPECT_REMOTE_NEIGHBOR": "1",
        }
    )
    subprocess.run(
        [torchrun, "--no-python", "--nproc-per-node=2", str(executable)],
        cwd=build_dir / "tests",
        env=env,
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
