"""Relocation credit for the baseline rise checks (tools/ratchet_gate.py off-pin, tools/check_provenance.py).

The baselines are per chapter, and a chapter's baseline may rise against the target branch's copy only when the code
pin moves. Moving a file to another chapter (a renumbering, a chapter taking another slot) moves its findings with
it, so the target chapter's baseline rises although nothing new was written. This module gives such a move credit,
under one narrow rule shared by both gates:

  * a file is relocated when `git diff -M` between the target ref and the working tree reports it as renamed, the
    old and new paths belong to different chapters, and the gate counts the file at both paths (the gate's own
    source filter: a file the gate never read has no finding in any baseline, so it can carry no credit);
  * a finding of the relocated file is credited only when it matches, finding by finding, a finding of the same
    file at the ref (the same identity, as a multiset: a finding present twice at the ref credits at most two);
  * the credited findings come off the source chapter's baseline at the ref and are added to the target chapter's,
    and the rise check runs against that adjusted copy. The move is zero-sum: the source's adjusted count is not
    clamped at zero, so a credit larger than what the source counted leaves it negative and its rise check fails.

Every other finding still counts: a new finding in a relocated file, a finding of a file that git does not see as
renamed, and any finding elsewhere in the target chapter must stay within the target's baseline at the ref. No
limit is raised and no baseline number is edited; only the copy at the ref that the rise check compares with is
adjusted, by the matched findings.
"""
from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path
from typing import Callable, Hashable, Iterable


def relocated_files(root: Path, ref: str, chapter: Callable[[str], str],
                    counted: Callable[[str], bool]) -> list[tuple[str, str, str, str]]:
    """(old, new, source chapter, target chapter) for every file under book/ that git sees as renamed between `ref`
    and the working tree into another chapter, and that the gate counts at both paths. Paths are relative to book/;
    `chapter` maps such a path to its chapter id, `counted` says whether the gate reads it."""
    r = subprocess.run(["git", "-C", str(root), "diff", "-M", "--name-status", "--relative", ref, "--", "book"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].startswith("R"):
            continue
        old, new = (p[len("book/"):] if p.startswith("book/") else p for p in parts[1:])
        if not (counted(old) and counted(new)):
            continue
        src, dst = chapter(old), chapter(new)
        if src != dst:
            out.append((old, new, src, dst))
    return out


def credit(pairs: Iterable[tuple[str, str, str, str]], old_findings: Callable[[str], list[Hashable]],
           new_findings: Callable[[str], list[Hashable]]) -> dict[tuple[str, str], int]:
    """{(source, target): n}: the findings of each relocated file that match, one for one, a finding of the same
    file at the ref. `old_findings(old)` lists the file's finding identities at the ref, `new_findings(new)` in the
    working tree."""
    out: dict[tuple[str, str], int] = {}
    for old, new, src, dst in pairs:
        n = sum((Counter(old_findings(old)) & Counter(new_findings(new))).values())
        if n:
            out[(src, dst)] = out.get((src, dst), 0) + n
    return out


def adjusted(old_counts: dict[str, int], credits: dict[tuple[str, str], int]) -> dict[str, int]:
    """The ref's per-chapter counts with each credit moved from its source chapter to its target chapter."""
    out = dict(old_counts)
    for (src, dst), n in credits.items():
        out[src] = out.get(src, 0) - n   # zero-sum: never clamped, an over-credit fails the source's check
        out[dst] = out.get(dst, 0) + n
    return out


def describe(credits: dict[tuple[str, str], int]) -> str:
    return ", ".join(f"{src} -> {dst}: {n}" for (src, dst), n in sorted(credits.items()))
