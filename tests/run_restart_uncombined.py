#!/usr/bin/env python3
"""Verify file-per-rank (uncombined) restart reproduces an uninterrupted run.

This exercises the ``combine: false`` restart path, where every rank writes its
own ``<basename>.block<rank>.<num>.restart`` file with no cross-rank barrier or
serial bundling. On restart, each rank loads its own file (the block id in the
path is rewritten to the local rank), so passing any one of the per-rank files
to ``--restart`` resumes all blocks.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np

SKIP_CODE = 125

try:
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_restart_uncombined: yaml import failed: {exc}")
  sys.exit(SKIP_CODE)


def run(cmd, cwd: Path, env=None) -> None:
  print(f"+ (cd {cwd} && {' '.join(cmd)})")
  subprocess.run(cmd, cwd=cwd, env=env, check=True)


def prepare_case(base_yaml: Path, case_dir: Path, blocks_per_process: int) -> Path:
  if case_dir.exists():
    shutil.rmtree(case_dir)
  case_dir.mkdir(parents=True)

  with base_yaml.open("r") as f:
    config = yaml.safe_load(f)

  dist = config.setdefault("distribute", {})
  dist["backend"] = "gloo"
  dist["blocks_per_process"] = blocks_per_process

  integration = config.setdefault("integration", {})
  integration["tlim"] = 0.0
  integration["nlim"] = 0
  integration["ncycle_out"] = 0

  config["output_dir"] = "."
  config["outputs"] = [
      # File-per-rank restart: no root-rank bundling.
      {"type": "restart", "dt": 0.0, "combine": False},
      {"type": "netcdf", "variables": ["prim"], "dt": 0.0},
  ]

  target_yaml = case_dir / base_yaml.name
  with target_yaml.open("w") as f:
    yaml.safe_dump(config, f)
  return target_yaml


def combine_output(case_dir: Path) -> Path:
  nc_files = sorted(case_dir.glob("*.nc"))
  if not nc_files:
    raise FileNotFoundError(f"No NetCDF output found in {case_dir}")
  return nc_files[-1]


def find_per_rank_restart(case_dir: Path) -> Path:
  # Uncombined dumps are named <basename>.block<rank>.<num>.restart. Confirm a
  # per-rank file exists for each block, then return one for --restart (each
  # rank rewrites the block id to its own rank on load).
  restart_files = sorted(case_dir.glob("*.block*.restart"))
  if not restart_files:
    raise FileNotFoundError(f"No per-rank restart file found in {case_dir}")
  return restart_files[0]


def compare_netcdf(path_a: Path, path_b: Path) -> None:
  with netCDF4.Dataset(path_a, "r") as data_a, netCDF4.Dataset(path_b, "r") as data_b:
    vars_a = {k: np.asarray(v[:]) for k, v in data_a.variables.items() if v.ndim > 0}
    vars_b = {k: np.asarray(v[:]) for k, v in data_b.variables.items() if v.ndim > 0}

    if vars_a.keys() != vars_b.keys():
      raise ValueError(f"Variable mismatch: {vars_a.keys()} vs {vars_b.keys()}")

    for name in vars_a:
      diff = np.abs(vars_a[name] - vars_b[name])
      max_abs = float(diff.max(initial=0.0))
      if max_abs != 0.0:
        raise ValueError(f"{name} differs after restart (max abs diff {max_abs})")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  bin_dir = build_dir / "bin"
  tests_dir = build_dir / "tests"
  repo_root = Path(__file__).resolve().parent.parent

  torchrun = shutil.which("torchrun")
  if torchrun is None:
    print("Skipping test_restart_uncombined: torchrun not found")
    return SKIP_CODE

  name = "straka"
  blocks_per_process = 2
  exe = bin_dir / f"{name}.{args.build_type}"
  if not exe.exists():
    raise FileNotFoundError(f"missing executable {exe}")

  base_yaml = repo_root / "examples" / "straka.yaml"
  if not base_yaml.exists():
    raise FileNotFoundError(f"missing input file {base_yaml}")

  env = os.environ.copy()
  env["BACKEND"] = "gloo"
  py_paths = [str(repo_root / "python"), str(repo_root)]
  existing = env.get("PYTHONPATH")
  if existing:
    py_paths.append(existing)
  env["PYTHONPATH"] = ":".join(py_paths)

  case_dir = tests_dir / f"restart_uncombined_{name}"
  yaml_path = prepare_case(base_yaml, case_dir, blocks_per_process)

  run(
      [
          torchrun,
          "--no-python",
          "--nproc-per-node=1",
          str(exe),
          str(yaml_path),
      ],
      cwd=case_dir,
      env=env,
  )
  base_nc = combine_output(case_dir)
  restart_file = find_per_rank_restart(case_dir)

  restart_dir = tests_dir / f"restart_uncombined_{name}_from_restart"
  if restart_dir.exists():
    shutil.rmtree(restart_dir)
  restart_dir.mkdir(parents=True)
  restart_yaml = restart_dir / yaml_path.name
  shutil.copy2(yaml_path, restart_yaml)
  # Copy every per-rank restart file so each block can load its own.
  for part in case_dir.glob("*.block*.restart"):
    shutil.copy2(part, restart_dir / part.name)

  run(
      [
          torchrun,
          "--no-python",
          "--nproc-per-node=1",
          str(exe),
          str(restart_yaml),
          "--restart",
          str((restart_dir / restart_file.name).resolve()),
      ],
      cwd=restart_dir,
      env=env,
  )
  restarted_nc = combine_output(restart_dir)
  compare_netcdf(base_nc, restarted_nc)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
