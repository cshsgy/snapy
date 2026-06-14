#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


SKIP_CODE = 125


def skip(msg: str) -> int:
    print(f"Skipping test_shallow_splash_decomp: {msg}")
    return SKIP_CODE


def run(cmd, cwd: Path, env=None) -> None:
    print(f"+ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def prepare_case(case_dir: Path, yaml_src: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    target = case_dir / "shallow_splash.yaml"
    try:
        target.symlink_to(yaml_src)
    except OSError:
        shutil.copy2(yaml_src, target)


def ensure_reference(tests_dir: Path) -> Path:
    ref = tests_dir / "shallow_splash-ref.nc"
    if ref.exists():
        return ref

    url = "https://zenodo.org/records/18121953/files/shallow_splash-ref.nc"
    print(f"+ downloading {url}")
    urllib.request.urlretrieve(url, ref)
    return ref


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-type", required=True)
    parser.add_argument("--backend", required=True, choices=("gloo", "ucx"))
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
    test_script = Path(__file__).with_name("test_shallow_splash.py").resolve()
    exe = bin_dir / f"shallow_splash.{args.build_type}"
    if not exe.exists():
        return skip(f"missing executable {exe}")
    if not test_script.exists():
        return skip(f"missing comparison script {test_script}")

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        return skip("torchrun not found in PATH")

    pd_combine = shutil.which("pd-combine")
    if pd_combine is None:
        return skip("pd-combine not found in PATH")

    try:
        reference = ensure_reference(tests_dir)
    except Exception as exc:
        return skip(f"reference download failed: {exc}")

    cases = [("mesh6", 1, "0", Path(os.path.abspath(bin_dir / "shallow_splash_mesh6.yaml")))]
    if args.backend == "gloo" or gpu_count >= 6:
        cases.append(("proc6", 6, "0,1,2,3,4,5", Path(os.path.abspath(bin_dir / "shallow_splash.yaml"))))
    cases.append(
        ("proc2_mesh3", 2, "0,1", Path(os.path.abspath(bin_dir / "shallow_splash_proc2_mesh3.yaml")))
    )
    if args.backend == "gloo" or gpu_count >= 3:
        cases.append(
            ("proc3_mesh2", 3, "0,1,2", Path(os.path.abspath(bin_dir / "shallow_splash_proc3_mesh2.yaml")))
        )

    for name, ranks, visible_devices, yaml_src in cases:
        if not yaml_src.exists():
            return skip(f"missing input file {yaml_src}")

        case_dir = tests_dir / f"shallow_splash_{args.backend}_{name}"
        prepare_case(case_dir, yaml_src)

        env = os.environ.copy()
        env["BACKEND"] = args.backend
        env["DEVICE"] = args.device
        if args.device == "cuda":
            env["CUDA_VISIBLE_DEVICES"] = visible_devices

        run(
            [
                torchrun,
                "--no-python",
                f"--nproc-per-node={ranks}",
                str(exe),
                "shallow_splash.yaml",
            ],
            cwd=case_dir,
            env=env,
        )
        run([pd_combine, "0", "-o", "main"], cwd=case_dir, env=env)
        run(
            [
                sys.executable,
                str(test_script),
                "shallow_splash-main.nc",
                str(reference),
            ],
            cwd=case_dir,
            env=env,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
