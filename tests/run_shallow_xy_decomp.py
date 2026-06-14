#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from netCDF4 import Dataset
import numpy as np


FIELDS = ("x1", "x2", "x3", "rho", "vel1", "vel2", "vel3")
SKIP_CODE = 125


def skip(msg: str) -> int:
    print(f"Skipping test_shallow_xy_decomp: {msg}")
    return SKIP_CODE


def run(cmd, cwd: Path, env=None) -> None:
    print(f"+ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def prepare_case(case_dir: Path, yaml_src: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    target = case_dir / "shallow_xy.yaml"
    try:
        target.symlink_to(yaml_src)
    except OSError:
        shutil.copy2(yaml_src, target)


def compare_pair(path_a: Path, path_b: Path, label: str) -> None:
    with Dataset(path_a, "r") as data_a, Dataset(path_b, "r") as data_b:
        for field in FIELDS:
            arr_a = np.asarray(data_a[field][:])
            arr_b = np.asarray(data_b[field][:])
            if arr_a.shape != arr_b.shape:
                raise ValueError(
                    f"{label}: field {field} shape mismatch {arr_a.shape} != {arr_b.shape}"
                )

            diff = np.abs(arr_a - arr_b)
            max_abs = float(diff.max(initial=0.0))
            rms = float(np.sqrt(np.mean(diff * diff)))
            print(f"{label}: {field} max_abs={max_abs} rms={rms}")
            if max_abs != 0.0:
                raise ValueError(
                    f"{label}: field {field} differs, max_abs={max_abs}, rms={rms}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-type", required=True)
    parser.add_argument("--backend", default="gloo", choices=("gloo", "ucx"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if args.backend == "ucx" and args.device == "cuda":
        try:
            import torch
        except Exception as exc:
            return skip(f"torch import failed: {exc}")

        if not torch.cuda.is_available():
            return skip("CUDA runtime is unavailable")

        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            return skip(f"need at least 2 GPUs, found {gpu_count}")
    else:
        gpu_count = 0

    build_dir = Path(args.build_dir).resolve()
    tests_dir = build_dir / "tests"
    bin_dir = build_dir / "bin"
    exe = bin_dir / f"shallow_xy.{args.build_type}"
    if not exe.exists():
        raise FileNotFoundError(f"missing executable {exe}")

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise FileNotFoundError("torchrun not found in PATH")

    pd_combine = shutil.which("pd-combine")
    if pd_combine is None:
        raise FileNotFoundError("pd-combine not found in PATH")

    cases = [("single", 1, "", Path(os.path.abspath(bin_dir / "shallow_xy_single.yaml")))]
    if args.backend == "gloo" or gpu_count >= 1:
        cases.append(("mesh4", 1, "0", Path(os.path.abspath(bin_dir / "shallow_xy_mesh4.yaml"))))
    cases.append(("proc2_mesh2", 2, "0,1", Path(os.path.abspath(bin_dir / "shallow_xy_proc2_mesh2.yaml"))))
    if args.backend == "gloo" or gpu_count >= 4:
        cases.append(("proc4", 4, "0,1,2,3", Path(os.path.abspath(bin_dir / "shallow_xy_proc4.yaml"))))

    outputs = {}
    for name, ranks, visible_devices, yaml_src in cases:
        if not yaml_src.exists():
            raise FileNotFoundError(f"missing input file {yaml_src}")

        case_dir = tests_dir / f"shallow_xy_{args.backend}_{name}"
        prepare_case(case_dir, yaml_src)

        env = os.environ.copy()
        env["BACKEND"] = args.backend
        env["DEVICE"] = args.device
        if args.device == "cuda" and visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
        run(
            [
                torchrun,
                "--no-python",
                f"--nproc-per-node={ranks}",
                str(exe),
                "shallow_xy.yaml",
            ],
            cwd=case_dir,
            env=env,
        )
        run([pd_combine, "0", "-o", "main"], cwd=case_dir, env=env)

        output = case_dir / "shallow_xy-main.nc"
        if not output.exists():
            raise FileNotFoundError(f"missing output file {output}")
        outputs[name] = output

    compare_pair(outputs["single"], outputs["mesh4"], "single vs mesh4")
    compare_pair(outputs["single"], outputs["proc2_mesh2"], "single vs proc2_mesh2")
    if "proc4" in outputs:
        compare_pair(outputs["single"], outputs["proc4"], "single vs proc4")
    compare_pair(outputs["mesh4"], outputs["proc2_mesh2"], "mesh4 vs proc2_mesh2")
    if "proc4" in outputs:
        compare_pair(outputs["mesh4"], outputs["proc4"], "mesh4 vs proc4")
        compare_pair(outputs["proc2_mesh2"], outputs["proc4"], "proc2_mesh2 vs proc4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
