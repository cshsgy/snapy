"""Provenance of quoted results (N7): every provenance tag of the book names a resolvable code sha and a command.

A quoted result (a measured number, a timing, a test outcome, an error norm) is marked by the provenance tag of
STYLE section 4, in the text or in a footnote definition:

    [sha: <40-hex commit of chengcli/snapy>; deck: <input>; run: `<exact command>`, <device, precision, ranks>;
     <versions>]

N7 sharpens it. A tag fails when
  - its `sha` field holds no full 40-hex commit (as `<sha>`, `snapy@<sha>` or `chengcli/snapy@<sha>`); a phrase
    such as "the code pin" or an abbreviated sha does not count;
  - the sha does not resolve in chengcli/snapy, or is not the code pin of code_pin.yml or a later commit;
  - it has no `run:` field with a backticked command;
  - it quotes a timing (a number of seconds or milliseconds, or a wall-clock time, in the tag or the rest of its
    footnote) and has no `scope:` field (what the time covers) or no `reps:` field (how many repetitions).

The count of failing tags per chapter may only fall. tools/provenance-baseline.json records it, and the check fails
when the file is missing, when a chapter's count differs from its baseline (a rise, or a fall not yet recorded), and,
given the target branch (`--base-ref`), when the baseline rises against the target's copy: a new chapter entry is a
rise from 0. The comparison reads the target's copy as tools/ratchet_gate.py does, and allows a rise only when the
code pin moves; tag problems of a file that git sees as renamed from another chapter, matched one for one with that
file's problems at the target, count against the source chapter there (tools/relocation_credit.py). `--update`
writes the current counts.

Shas are resolved like tools/code_anchors.py resolves the pin: a local clone (`SNAPY_REPO`, then the repository
holding this report), else a cache under `SNAPY_REPORT_CACHE` fetched from GitHub with its history, so that "the pin
or later" can be decided.

usage: python3 tools/check_provenance.py [report_dir] [--update] [--list] [--base-ref REF]   (exit 0: at the baseline)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = "chengcli/snapy"
BASELINE = "tools/provenance-baseline.json"
TARGET = "tech-report"
_TAG = re.compile(r"\[sha:\s*((?:[^\[\]]|\[[^\[\]]*\])*)\]", re.S)
_HEX = re.compile(r"(?:([\w./-]+)@)?\b([0-9a-f]{7,40})\b")
_TIMING = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|min|seconds?)\b(?!\w)|\bwall[- ](?:clock|time)\b", re.I)
_CODE = re.compile(r"`[^`]+`")


# --- the book -----------------------------------------------------------------------------------------------------

def _counted(rel):
    """Whether this gate reads the file at `rel` (relative to book/): the filter of _sources()."""
    return rel.endswith(".qmd") and not rel.startswith(("_", "."))


def _sources(book):
    for q in sorted(Path(book).rglob("*.qmd")):
        rel = q.relative_to(book).as_posix()
        if rel.startswith(("_", ".")):
            continue
        yield rel, q.read_text(encoding="utf-8")


def _chapter(rel):
    parts = rel.split("/")
    if rel == "index.qmd":
        return "preface"
    if len(parts) < 2 or parts[0] != "chapters":
        return "book"
    m = re.match(r"(\d\d)", parts[1])
    if m:
        return f"ch{m.group(1)}"
    m = re.match(r"_?(app[a-z])", parts[-1]) or re.match(r"(app[a-z])", parts[1])
    return m.group(1) if m else parts[1]


def _fields(body):
    """The `name: value` fields of a tag body, split at top-level semicolons; the first is the sha field."""
    parts, depth, cur = [], 0, ""
    for ch in body:
        if ch == "`":
            depth ^= 1
        if ch == ";" and not depth:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = {"sha": parts[0].strip()}
    for p in parts[1:]:
        m = re.match(r"\s*([a-z][a-z ]*?):\s*(.*)", p, re.S)
        if m:
            out.setdefault(m.group(1).strip(), m.group(2).strip())
    return out


def _context(text, end):
    """The rest of the paragraph after a tag: for a footnote, its continuation lines, up to a blank line."""
    rest = text[end:]
    stop = re.search(r"\n\s*\n", rest)
    return rest[:stop.start()] if stop else rest


def tags(book):
    """Every provenance tag: dict(file, line, chapter, fields, text)."""
    out = []
    for rel, text in _sources(book):
        for m in _TAG.finditer(text):
            out.append(dict(file=rel, line=text.count("\n", 0, m.start()) + 1, chapter=_chapter(rel),
                            fields=_fields(m.group(1)), text=m.group(1) + _context(text, m.end())))
    return out


# --- shas ---------------------------------------------------------------------------------------------------------

def _git(gitdir, *args):
    return subprocess.run(["git", "-C", str(gitdir), *args], capture_output=True, text=True)


def _has(gitdir, sha):
    return _git(gitdir, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


class Resolver:
    """Decide whether a sha is a commit of chengcli/snapy at or after the pin."""

    def __init__(self, report, pin, fetch=True, local=None):
        self.pin, self.fetch, self.memo = pin, fetch, {}
        cands = [Path(p) for p in ([os.environ.get("SNAPY_REPO", "")] if local is None else local) if p]
        if local is None:
            top = _git(report, "rev-parse", "--show-toplevel")
            if top.returncode == 0:
                cands.append(Path(top.stdout.strip()))
        self.local = cands
        cache = Path(os.environ.get("SNAPY_REPORT_CACHE", Path.home() / ".cache" / "snapy-report"))
        self.bare = cache / (REPO.replace("/", "__") + "-history.git")

    def _dir(self, sha):
        for d in self.local:
            if _has(d, sha) and _has(d, self.pin) and _git(d, "rev-parse", "--is-shallow-repository").stdout.strip() != "true":
                return d
        if not self.fetch:
            return None
        if not (self.bare / "HEAD").exists():
            try:
                self.bare.mkdir(parents=True, exist_ok=True)
            except OSError:
                return None
            _git(self.bare, "init", "--quiet", "--bare")
        for s in {sha, self.pin}:
            if not _has(self.bare, s):
                _git(self.bare, "fetch", "--quiet", f"https://github.com/{REPO}.git", s)
        return self.bare if _has(self.bare, sha) and _has(self.bare, self.pin) else None

    def __call__(self, sha):
        """None when the sha is at or after the pin, else the reason it fails."""
        if sha not in self.memo:
            d = self._dir(sha)
            if d is None:
                self.memo[sha] = f"sha {sha[:12]} does not resolve in {REPO}"
            elif _git(d, "merge-base", "--is-ancestor", self.pin, sha).returncode != 0:
                self.memo[sha] = f"sha {sha[:12]} is not the code pin {self.pin[:12]} or a later commit"
            else:
                self.memo[sha] = None
        return self.memo[sha]


# --- the check ----------------------------------------------------------------------------------------------------

def problems(tag, resolve):
    """What a tag lacks, as a list of short reasons (empty when it is complete)."""
    f, out = tag["fields"], []
    # a commit of snapy: bare, or `snapy@` / `chengcli/snapy@`; another repository's sha (pyharp@...) does not count
    ours = [h for repo, h in _HEX.findall(f["sha"]) if not repo or repo in ("snapy", "chengcli/snapy")]
    full = [h for h in ours if len(h) == 40]
    if not full:
        out.append(f"short sha {ours[0]}" if ours else f"no full sha (sha field: {f['sha'][:40]!r})")
    else:
        why = resolve(full[0])
        if why:
            out.append(why)
    command = _CODE.match(f.get("run", "").lstrip())
    # A command leads the field; inline API examples embedded in prose are not invocations.
    shell_start = r"(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*[A-Za-z0-9_./~+-]+(?:\s|$)"
    if command is None or not re.match(shell_start, command.group()[1:-1]):
        out.append("no command (no `run:` field with a backticked command)")
    if _TIMING.search(tag["text"]):
        missing = [k for k in ("scope", "reps") if not f.get(k)]
        if missing:
            out.append("timing without " + " and ".join(f"`{k}:`" for k in missing))
    return out


def read_pin(report):
    text = (Path(report) / "code_pin.yml").read_text(encoding="utf-8")
    return re.search(r"\b([0-9a-f]{40})\b", text).group(1)


def violations(report, resolve=None):
    report = Path(report)
    resolve = resolve or Resolver(report, read_pin(report))
    out = []
    for t in tags(report / "book"):
        for why in problems(t, resolve):
            out.append(dict(file=t["file"], line=t["line"], chapter=t["chapter"], why=why))
    return out


def counts(found):
    c = {}
    for v in found:
        c[v["chapter"]] = c.get(v["chapter"], 0) + 1
    return dict(sorted(c.items()))


def base_ref(report):
    """The target branch the baseline may not rise against, picked as the ratchet step of CI picks it.

    PROVENANCE_BASE_REF when set; in GitHub Actions, the pull request's target (GITHUB_BASE_REF) or else the previous
    tip of a push, fetched shallowly into the checkout; otherwise origin/tech-report."""
    if os.environ.get("PROVENANCE_BASE_REF"):
        return os.environ["PROVENANCE_BASE_REF"]
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return f"origin/{TARGET}"
    target = os.environ.get("GITHUB_BASE_REF", "")
    if target:
        _git(report, "fetch", "--quiet", "--depth=1", "origin", target)
        return f"origin/{target}"
    before = ""
    try:
        before = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")).get("before") or ""
    except (KeyError, OSError, ValueError):
        pass
    if re.fullmatch(r"[0-9a-f]{40}", before) and before.strip("0"):
        _git(report, "fetch", "--quiet", "--depth=1", "origin", before)
        return before
    _git(report, "fetch", "--quiet", "--depth=1", "origin", TARGET)
    return f"origin/{TARGET}"


def file_findings(rel, text, resolve):
    """The identities of one file's tag problems: (sha field, reason), one per problem."""
    out = []
    for m in _TAG.finditer(text):
        tag = dict(file=rel, fields=_fields(m.group(1)), text=m.group(1) + _context(text, m.end()))
        out += [(tag["fields"]["sha"], why) for why in problems(tag, resolve)]
    return out


def raised(report, ref, baseline, resolve=None):
    """Chapters whose baseline rises against the baseline at `ref` (tools/ratchet_gate.py's baseline_raised, for
    this file). A `ref` that names no commit is a problem; no baseline at `ref` means this change introduces it.
    Tag problems of files git sees as renamed into another chapter, matched one for one with the same file's
    problems at `ref`, move from the source chapter's count at `ref` to the target's (tools/relocation_credit.py)."""
    from ratchet_gate import _git_show
    from relocation_credit import adjusted, credit, describe, relocated_files
    if _git(report, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
        return [f"base ref {ref!r} names no commit here (fetch it first): the baseline cannot be compared"]
    old = _git_show(Path(report), ref, BASELINE)
    if old is None:
        return []
    old_pin = re.search(r"\b([0-9a-f]{40})\b", _git_show(Path(report), ref, "code_pin.yml") or "")
    if old_pin and old_pin.group(1) != read_pin(report):
        return []  # the pin moves in this change: which shas count moves with it
    old = json.loads(old)
    resolve = resolve or Resolver(Path(report), read_pin(report))
    credits = credit(relocated_files(Path(report), ref, _chapter, _counted),
                     lambda rel: file_findings(rel, _git_show(Path(report), ref, f"book/{rel}") or "", resolve),
                     lambda rel: file_findings(rel, (Path(report) / "book" / rel).read_text(encoding="utf-8"),
                                               resolve))
    if credits:
        print(f"[provenance] relocation credit (renamed files only): {describe(credits)}")
    old = adjusted(old, credits)
    return [f"{ch}: the baseline rises {old.get(ch, 0)} -> {baseline.get(ch, 0)} against {ref}"
            for ch in sorted(set(baseline) | set(old)) if baseline.get(ch, 0) > old.get(ch, 0)]


def check(report, baseline=None, resolve=None, ref=None):
    """Return (status, messages, found): status 1 when the baseline is missing, differs from the counts, or rises
    against the baseline at `ref`."""
    report = Path(report)
    baseline = Path(baseline) if baseline else report / BASELINE
    found = violations(report, resolve)
    if not baseline.exists():
        return 1, [f"{BASELINE} is missing; write it with python3 tools/check_provenance.py --update"], found
    base = json.loads(baseline.read_text(encoding="utf-8"))
    have, msgs = counts(found), []
    for ch in sorted(set(have) | set(base)):
        n, b = have.get(ch, 0), base.get(ch, 0)
        if n > b:
            new = [v for v in found if v["chapter"] == ch]
            msgs.append(f"{ch}: {n} tag problem(s), baseline {b}; e.g. "
                        + "; ".join(f"{v['file']}:{v['line']}: {v['why']}" for v in new[:3])
                        + " (python3 tools/check_provenance.py --list lists every one)")
        elif n < b:
            msgs.append(f"{ch}: {n} tag problem(s), baseline {b}; the baseline must equal the count "
                        "(record a fall with python3 tools/check_provenance.py --update)")
    if ref:
        msgs += raised(report, ref, base, resolve)
    return (1 if msgs else 0), msgs, found


def main(argv):
    ref = argv[argv.index("--base-ref") + 1] if "--base-ref" in argv[:-1] else None
    args = [a for a in argv if not a.startswith("--") and a != ref]
    report = Path(args[0]) if args else Path(__file__).resolve().parent.parent
    baseline = report / BASELINE
    status, msgs, found = check(report, baseline, ref=ref)
    if "--update" in argv:
        baseline.write_text(json.dumps(counts(found), indent=1) + "\n", encoding="utf-8")
        print(f"[provenance] baseline written: {len(found)} tag problem(s)")
        return 0
    base = json.loads(baseline.read_text(encoding="utf-8")) if baseline.exists() else {}
    print(f"[provenance] {len(tags(report / 'book'))} tags, {len(found)} problem(s); baseline {sum(base.values())}")
    if "--list" in argv:
        for v in found:
            print(f"  {v['file']}:{v['line']}: {v['why']}")
    for m in msgs:
        print(f"  ✗ {m}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
