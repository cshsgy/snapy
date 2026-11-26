#!/usr/bin/env python3
"""A failed combine on the root rank must stop every rank.

Two ranks run examples/shallow_xy_proc2_mesh2.yaml with one netcdf output
(combine on) and no step: the frame written after initialization is the only
one. The combined target `<basename>.out0.00000.nc` is pre-created as a
non-empty directory, so the root's combine fails (remove() cannot delete it and
the combined file cannot be created in its place). Expected: the run exits
non-zero and EVERY rank's stream carries the combine failure, because the root's
failure is shared with the other ranks. Without that, only the root raises and
rank 1 leaves combine_on_root as if the combined file existed, carries on into
the next collective and is torn down by torchrun without a word of its own.

  python run_combine_root_throw.py --build-dir <build> --build-type release

Exit 125 (skip) when torchrun is missing or when the build has no NetCDF, in
which case no part file is written and there is nothing to combine.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SKIP_CODE = 125
MESSAGE = "combine failed on the root rank"

try:
  import yaml
except Exception as exc:  # pragma: no cover - dependency guard
  print(f"Skipping test_combine_root_throw: yaml import failed: {exc}")
  sys.exit(SKIP_CODE)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  tests_dir = build_dir / "tests"
  repo_root = Path(__file__).resolve().parent.parent
  exe = build_dir / "bin" / f"shallow_xy.{args.build_type}"
  if not exe.exists():
    raise FileNotFoundError(f"missing executable {exe}")
  torchrun = shutil.which("torchrun")
  if torchrun is None:
    print("Skipping test_combine_root_throw: torchrun not found")
    return SKIP_CODE

  case_dir = tests_dir / "combine_root_throw"
  if case_dir.exists():
    shutil.rmtree(case_dir)
  case_dir.mkdir(parents=True)
  base_yaml = repo_root / "examples" / "shallow_xy_proc2_mesh2.yaml"
  with base_yaml.open() as f:
    config = yaml.safe_load(f)
  config["integration"]["tlim"] = 0.0
  config["integration"]["nlim"] = 0
  config["output_dir"] = "."
  config["outputs"] = [{"type": "netcdf", "variables": ["prim"], "dt": 0.0}]
  yaml_path = case_dir / "combine_root.yaml"
  with yaml_path.open("w") as f:
    yaml.safe_dump(config, f)
  # the combined target, as a directory the root cannot remove or replace
  target = case_dir / "combine_root.out0.00000.nc"
  target.mkdir()
  (target / "placeholder").write_text("not a file the combine may replace\n")

  env = os.environ.copy()
  env["BACKEND"] = "gloo"
  py_paths = [str(repo_root / "python"), str(repo_root)]
  if env.get("PYTHONPATH"):
    py_paths.append(env["PYTHONPATH"])
  env["PYTHONPATH"] = ":".join(py_paths)

  rank_dir = case_dir / "ranks"
  cmd = [torchrun, "--no-python", "--nproc-per-node=2", "--log-dir",
         str(rank_dir), "--redirects", "3", str(exe), str(yaml_path)]
  print(f"+ (cd {case_dir} && {' '.join(cmd)})")
  proc = subprocess.run(cmd, cwd=case_dir, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True)
  print(proc.stdout)
  streams = {}
  for path in sorted(rank_dir.glob("*/attempt_*/*/std*.log")):
    rank = path.parent.name
    streams.setdefault(rank, "")
    streams[rank] += path.read_text(errors="replace")
  for rank, text in sorted(streams.items()):
    print(f"==== rank {rank} ====")
    print(text)

  parts = sorted(case_dir.glob("combine_root.block*.out0.00000.nc"))
  if not parts and not any(MESSAGE in t for t in streams.values()):
    print("Skipping test_combine_root_throw: this build writes no netcdf")
    return SKIP_CODE
  if proc.returncode == 0:
    print("FAIL: the run finished although the root's combine failed")
    return 1
  missing = [r for r, t in sorted(streams.items()) if MESSAGE not in t]
  if len(streams) < 2:
    print(f"FAIL: expected two rank streams, found {sorted(streams)}")
    return 1
  if missing:
    print(f"FAIL: rank(s) {missing} did not report the root's combine failure")
    return 1
  print("PASS: every rank stopped on the root's combine failure")
  return 0


if __name__ == "__main__":
  sys.exit(main())
