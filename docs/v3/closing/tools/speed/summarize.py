#!/usr/bin/env python3
"""Collect aam/speed/runs/*.rank0.json into per-cell medians: for each cell and tree, the median
per-step time of each of the 3 runs; reported value = median of the 3 run medians, spread = min..max
of the run medians. Prints a markdown table and writes summary.json."""
import json, glob, os, statistics, sys
R = os.environ.get("SPEED_RUNS") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
cells = {}
for f in sorted(glob.glob(os.path.join(R, "*.rank0.json"))):
    base = os.path.basename(f)[:-len(".rank0.json")]
    name, tree, rep = base.rsplit(".", 2)
    d = json.load(open(f))
    cells.setdefault(name, {}).setdefault(tree, []).append(d)
out = {}
for name, trees in cells.items():
    out[name] = {}
    for tree, runs in trees.items():
        meds = sorted(r["median_s"] for r in runs)
        rep = runs[0]
        out[name][tree] = {"runs": len(runs), "median_ms": 1e3 * meds[len(meds) // 2], "spread_ms": [1e3 * meds[0], 1e3 * meds[-1]],
                           "per_run_median_ms": [1e3 * m for m in meds], "steps": rep["steps"], "execution_report": rep.get("execution_report"),
                           "capability": rep.get("capability"), "dt0": rep["dt0"], "world_size": rep["world_size"], "blocks": rep["blocks_in_process"],
                           "driver": rep.get("driver"), "threads": rep.get("threads"), "gpu": rep.get("gpu"), "finite": all(r["finite"] for r in runs)}
json.dump(out, open(os.environ.get("SPEED_SUMMARY") or os.path.join(os.path.dirname(R), "summary.json"), "w"), indent=1)
def fmt(c):
    return f"{c['median_ms']:.2f} ms ({c['spread_ms'][0]:.2f}–{c['spread_ms'][1]:.2f})"
print("| cell | before 730b3500 | after d3178f20 | speed-up (before/after) |")
print("|---|---|---|---|")
for name, trees in out.items():
    b = trees.get("before"); a = trees.get("after")
    su = f"{b['median_ms']/a['median_ms']:.2f}x" if (a and b) else "—"
    print(f"| {name} | {fmt(b) if b else 'n/a'} | {fmt(a) if a else 'n/a'} | {su} |")
print()
print("| eager vs graph (after) | eager | cuda-graph | graph speed-up |")
print("|---|---|---|---|")
for name, trees in out.items():
    if name.endswith("_eager") and name[:-6] + "_graph" in out:
        eb = out.get(name + "B", {}).get("after")  # the eager runs taken back to back with the graph runs
        e = eb or trees.get("after"); g = out[name[:-6] + "_graph"].get("after")
        if e and g:
            print(f"| {name[:-6]} | {fmt(e)} | {fmt(g)} | {e['median_ms']/g['median_ms']:.2f}x |")
