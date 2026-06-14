#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SKIP_CODE = 125
MASS_PATTERN = re.compile(r"mass0=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def skip(msg: str) -> int:
    print(f"Skipping example mass check: {msg}")
    return SKIP_CODE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--build-type", required=True)
    parser.add_argument("--example", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--backend", required=True, choices=("gloo", "ucx"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--mass-rtol", type=float, default=1.0e-8)
    return parser.parse_args()


def link_resource(target: Path, src: Path) -> None:
    try:
        target.symlink_to(src)
    except OSError:
        shutil.copy2(src, target)


def kintera_data_dir() -> Path | None:
    try:
        import kintera
    except Exception:
        return None

    data_dir = Path(kintera.__file__).resolve().parent / "data"
    if data_dir.is_dir():
        return data_dir
    return None


def prepare_case(case_dir: Path, yaml_src: Path, yaml_name: str) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    link_resource(case_dir / yaml_name, yaml_src)

    data_dir = kintera_data_dir()
    if data_dir is not None:
        for resource in data_dir.iterdir():
            if resource.is_file():
                link_resource(case_dir / resource.name, resource)


def torchrun_path() -> str | None:
    return shutil.which("torchrun")


def ensure_cuda() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        return False, f"torch import failed: {exc}"

    if not torch.cuda.is_available():
        return False, "CUDA runtime is unavailable"
    if torch.cuda.device_count() < 1:
        return False, "need at least 1 GPU, found 0"
    return True, ""


def run_case(
    torchrun: str | None,
    exe: Path,
    example: str,
    yaml_name: str,
    case_dir: Path,
    env: dict[str, str],
    nproc: int,
) -> Path:
    log_path = case_dir / "run.log"
    if nproc == 1:
        cmd = [str(exe)]
    else:
        if torchrun is None:
            raise FileNotFoundError("torchrun not found in PATH")
        cmd = [torchrun, "--no-python", f"--nproc-per-node={nproc}", str(exe)]
    if example == "run_hydro":
        cmd.extend(["-i", yaml_name])
    else:
        cmd.append(yaml_name)
    print(f"+ (cd {case_dir} && {' '.join(cmd)})")
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            cwd=case_dir,
            env=env,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return log_path


def print_log_on_failure(log_path: Path) -> None:
    if not log_path.exists():
        return
    print(f"--- begin {log_path} ---")
    try:
        print(log_path.read_text(encoding="utf-8"), end="")
    except Exception as exc:
        print(f"<failed to read log: {exc}>")
    print(f"--- end {log_path} ---")


def load_masses(log_path: Path) -> list[float]:
    text = log_path.read_text(encoding="utf-8")
    return [float(match.group(1)) for match in MASS_PATTERN.finditer(text)]


def check_mass(log_path: Path, rtol: float) -> None:
    masses = load_masses(log_path)
    if len(masses) < 2:
        raise ValueError(f"Expected at least 2 mass0 samples in {log_path}, found {len(masses)}")

    start = masses[0]
    end = masses[-1]
    scale = max(abs(start), 1.0)
    rel = abs(end - start) / scale
    print(f"mass drift start={start:.16e} end={end:.16e} relative={rel:.3e}")
    if rel > rtol:
        raise ValueError(f"Mass drift {rel:.3e} exceeds tolerance {rtol:.3e}")


def main() -> int:
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    bin_dir = build_dir / "bin"
    tests_dir = build_dir / "tests"
    exe = bin_dir / f"{args.example}.{args.build_type}"
    yaml_src = bin_dir / args.input
    yaml_name = f"{args.example}.yaml"

    if not exe.exists():
        raise FileNotFoundError(f"missing executable {exe}")
    if not yaml_src.exists():
        raise FileNotFoundError(f"missing input file {yaml_src}")

    torchrun = torchrun_path()
    if args.nproc > 1 and torchrun is None:
        return skip("torchrun not found in PATH")

    env = os.environ.copy()
    env["BACKEND"] = args.backend
    device = args.device or "cpu"
    env["DEVICE"] = device
    if device == "cuda":
        ok, msg = ensure_cuda()
        if not ok:
            return skip(msg)
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    case_dir = tests_dir / f"{args.example}_{args.backend}"
    prepare_case(case_dir, yaml_src, yaml_name)
    try:
        log_path = run_case(
            torchrun, exe, args.example, yaml_name, case_dir, env, args.nproc
        )
    except subprocess.CalledProcessError as exc:
        print_log_on_failure(case_dir / "run.log")
        raise exc
    check_mass(log_path, args.mass_rtol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
