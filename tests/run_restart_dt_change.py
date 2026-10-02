#!/usr/bin/env python3
"""A restart output whose dt shrinks must not keep a next_time left over from the old
interval. The reproduction disables prim with dt 1e30, then resumes it at dt 7: before
the fix the saved next_time is ~1e30, so the resumed run writes no prim frames."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_restart_new_output import (BASE_TLIM, FRAME_DT, RESTART_DT, RESUME_TLIM,
                                    restart_schedule, run, stream_times, write_case)

import shutil

SKIP_CODE = 125
DISABLED_DT = 1.0e30
# A hydro step is a fraction of a second on this mesh. A frame that belongs to
# schedule point T lands within this window of T, and nowhere near the resume instant.
STEP_SLOP = 1.0


def main() -> int:
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--build-dir", required=True)
  parser.add_argument("--build-type", required=True)
  args = parser.parse_args()

  build_dir = Path(args.build_dir).resolve()
  tests_dir = build_dir / "tests"
  repo_root = Path(__file__).resolve().parent.parent
  exe = build_dir / "bin" / f"straka.{args.build_type}"
  if not exe.exists():
    raise FileNotFoundError(f"missing executable {exe}")
  base_yaml = repo_root / "examples" / "straka.yaml"
  torchrun = shutil.which("torchrun")
  if torchrun is None:
    print("Skipping test_restart_dt_change: torchrun not found")
    return SKIP_CODE

  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = ""
  env["OMP_NUM_THREADS"] = "1"
  env["BACKEND"] = "gloo"
  py_paths = [str(repo_root / "python"), str(repo_root)]
  if env.get("PYTHONPATH"):
    py_paths.append(env["PYTHONPATH"])
  env["PYTHONPATH"] = ":".join(py_paths)

  restart = {"type": "restart", "dt": RESTART_DT}
  disabled = {"type": "netcdf", "variables": ["prim"], "dt": DISABLED_DT}
  enabled = {"type": "netcdf", "variables": ["prim"], "dt": FRAME_DT}
  launch = [torchrun, "--no-python", "--nproc-per-node=1", str(exe)]

  base_dir = tests_dir / "restart_dt_change_base"
  base_yaml_run = write_case(base_yaml, base_dir, BASE_TLIM, [restart, disabled])
  run(launch + [str(base_yaml_run)], base_dir, env)
  # The final restart is at tlim. The case in the issue resumes from the dt=23
  # cadence file, whose prim next_time is still the disabled interval.
  candidates = []
  for path in sorted(base_dir.glob("*.restart")):
    t, nxt, _ = restart_schedule(path)
    if len(nxt) > 1 and nxt[1] > t + FRAME_DT:
      candidates.append((t, path, nxt))
  if not candidates:
    raise AssertionError("base run never left prim next_time more than one new dt ahead")
  resume_t, restart_file, saved_next = min(candidates, key=lambda item: item[0])

  resumed_dir = tests_dir / "restart_dt_change_resumed"
  resumed_yaml = write_case(base_yaml, resumed_dir, RESUME_TLIM, [restart, enabled])
  run(launch + [str(resumed_yaml), "--restart", str(restart_file.resolve())],
      resumed_dir, env)

  times = stream_times(resumed_dir).get("out1", [])
  expected = []
  t = resume_t + FRAME_DT
  while t <= RESUME_TLIM + 1e-9:
    expected.append(t)
    t += FRAME_DT
  if len(times) != len(expected):
    raise AssertionError(
        f"prim frames {times}, expected one near each of {expected} "
        f"(resume {resume_t}, dt {FRAME_DT}, tlim {RESUME_TLIM})")
  for got, want in zip(times, expected):
    if abs(got - want) > STEP_SLOP:
      raise AssertionError(
          f"prim frame at {got} is not the scheduled {want} (slop {STEP_SLOP})")
  print(f"ok: resume {resume_t:.3f}; prim frames {times}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
