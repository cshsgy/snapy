"""Relocation credit (tools/relocation_credit.py) in the two baseline rise checks.

For each gate, two cases on a small git repository standing in for the target branch:
  (a) a pure relocation: a section file moves (git mv) from one chapter into another, and the baseline entry moves
      with it. The target chapter's baseline rises against the target branch, but every finding in it matches one
      of the moved file's findings at the target branch, so the rise check passes.
  (b) the same relocation plus one unrelated new finding in the target chapter, recorded in the baseline: the new
      finding gets no credit, and the rise check still fails.
Without the credit, (a) fails like (b).
"""
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ratchet_gate as rg  # noqa: E402
from test_provenance import NO_COMMAND, _gate, _report, _set, _target, snapy  # noqa: E402,F401
from test_ratchet_gate import OTHER, OWNERS, check, git, make  # noqa: E402

OLDER = "2" * 40   # another off-pin snapy sha


# --- the ratchet's off-pin count ----------------------------------------------------------------------------------

def _ratchet_base(tmp_path):
    """A report whose chapter 3 holds one section file with two off-pin shas, committed as the target branch."""
    root = make(tmp_path, owners=OWNERS + "ch03\tcarol\nch04\tdave\n")
    sec = root / "book" / "chapters" / "03-old" / "_sec.qmd"
    sec.parent.mkdir(parents=True)
    sec.write_text(f"## Old section\n\nRe-run at snapy@{OTHER}. Compared with snapy@{OLDER}.\n")
    assert rg.main(["--root", str(root), "update"]) == 0
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    git(root, "tag", "base")
    return root


def _relocate(root):
    """Move the section file into chapter 4 and move its baseline entry with it (counts unchanged)."""
    (root / "book" / "chapters" / "04-new").mkdir()
    git(root, "mv", "book/chapters/03-old/_sec.qmd", "book/chapters/04-new/_sec.qmd")
    b = json.loads((root / "tools" / "ratchet-baseline.json").read_text())
    b["chapters"]["ch04"] = b["chapters"].pop("ch03")
    (root / "tools" / "ratchet-baseline.json").write_text(json.dumps(b))


def test_ratchet_a_pure_relocation_passes(tmp_path, capsys):
    root = _ratchet_base(tmp_path)
    _relocate(root)
    assert check(root) == 0                                   # the counts equal the moved baseline
    assert check(root, "--base-ref", "base") == 0             # ch04 offpin 0 -> 2, all credited
    assert "relocation credit (offpin, renamed files only): ch03 -> ch04: 2" in capsys.readouterr().out


def test_ratchet_a_relocation_with_a_new_finding_fails(tmp_path, capsys):
    root = _ratchet_base(tmp_path)
    _relocate(root)
    other = root / "book" / "chapters" / "04-new" / "_other.qmd"
    other.write_text(f"## Other\n\nSee snapy@{'3' * 40}.\n")
    b = json.loads((root / "tools" / "ratchet-baseline.json").read_text())
    b["chapters"]["ch04"]["offpin"] += 1                      # the new finding written into the baseline by hand
    (root / "tools" / "ratchet-baseline.json").write_text(json.dumps(b))
    assert check(root) == 0
    assert check(root, "--base-ref", "base") == 1
    assert "ch04 offpin: the baseline rises" in capsys.readouterr().out


def test_ratchet_a_new_finding_inside_the_moved_file_is_not_credited(tmp_path, capsys):
    root = _ratchet_base(tmp_path)
    _relocate(root)
    sec = root / "book" / "chapters" / "04-new" / "_sec.qmd"
    sec.write_text(sec.read_text() + f"And snapy@{'4' * 40}.\n")
    b = json.loads((root / "tools" / "ratchet-baseline.json").read_text())
    b["chapters"]["ch04"]["offpin"] += 1
    (root / "tools" / "ratchet-baseline.json").write_text(json.dumps(b))
    assert check(root, "--base-ref", "base") == 1             # 2 credited, the third is a rise


# --- the provenance gate ------------------------------------------------------------------------------------------

def _provenance_base(tmp_path, snapy):
    """A report whose chapter 3 holds one section file with two tags lacking a command, committed as the target."""
    report = _report(tmp_path, snapy, "complete.qmd", baseline={"ch03": 2})
    sec = report / "book" / "chapters" / "03-old" / "_sec.qmd"
    sec.parent.mkdir(parents=True)
    sec.write_text("## Old\n" + (NO_COMMAND * 2).replace("{pin}", snapy[1]["pin"]))
    return report, _target(report)


def _relocate_provenance(report):
    (report / "book" / "chapters" / "04-new").mkdir()
    subprocess.run(["git", "-C", str(report), "mv", "book/chapters/03-old/_sec.qmd", "book/chapters/04-new/_sec.qmd"],
                   check=True)


def test_provenance_a_pure_relocation_passes(tmp_path, snapy, monkeypatch, capsys):
    report, base = _provenance_base(tmp_path, snapy)
    _relocate_provenance(report)
    _set(report, {"ch04": 2})                                 # the entry moved, the count unchanged
    assert _gate(report, snapy, monkeypatch, base) == 0
    assert "relocation credit (renamed files only): ch03 -> ch04: 2" in capsys.readouterr().out


def test_provenance_a_relocation_with_a_new_finding_fails(tmp_path, snapy, monkeypatch, capsys):
    report, base = _provenance_base(tmp_path, snapy)
    _relocate_provenance(report)
    q = report / "book" / "chapters" / "04-new" / "_sec.qmd"
    q.write_text(q.read_text() + NO_COMMAND.replace("{pin}", snapy[1]["pin"]))   # a third, unrelated problem
    _set(report, {"ch04": 3})
    assert _gate(report, snapy, monkeypatch, base) >= 1
    assert "ch04: the baseline rises" in capsys.readouterr().out


# --- a renamed file the gate does not count carries no credit (review 6095629692) ---------------------------------

def test_ratchet_an_uncounted_renamed_file_gives_no_credit(tmp_path, capsys):
    """A non-.qmd file with off-pin shas moves out of a chapter that counts none, and three new findings appear in
    the target chapter, written into its baseline by hand: no credit, the rise fails."""
    root = make(tmp_path, owners=OWNERS + "ch04\tdave\nch05\terin\n")
    notes = root / "book" / "chapters" / "05-src" / "notes.txt"
    notes.parent.mkdir(parents=True)
    notes.write_text("".join(f"snapy@{c * 40}\n" for c in "567"))
    assert rg.main(["--root", str(root), "update"]) == 0
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    git(root, "tag", "base")
    (root / "book" / "chapters" / "04-new").mkdir()
    git(root, "mv", "book/chapters/05-src/notes.txt", "book/chapters/04-new/notes.txt")
    (root / "book" / "chapters" / "04-new" / "_x.qmd").write_text(
        "## X\n\n" + "".join(f"See snapy@{c * 40}.\n" for c in "89a"))
    b = json.loads((root / "tools" / "ratchet-baseline.json").read_text())
    b["chapters"]["ch04"] = {"history": 0, "offpin": 3, "status": 0, "xref": 0}
    (root / "tools" / "ratchet-baseline.json").write_text(json.dumps(b))
    assert check(root) == 0
    assert check(root, "--base-ref", "base") == 1
    out = capsys.readouterr().out
    assert "ch04 offpin: the baseline rises 0 -> 3" in out
    assert "relocation credit" not in out


def test_provenance_an_uncounted_renamed_file_gives_no_credit(tmp_path, snapy, monkeypatch, capsys):
    report = _report(tmp_path, snapy, "complete.qmd", baseline={})
    notes = report / "book" / "chapters" / "05-src" / "notes.txt"
    notes.parent.mkdir(parents=True)
    notes.write_text((NO_COMMAND * 3).replace("{pin}", snapy[1]["pin"]))
    base = _target(report)
    (report / "book" / "chapters" / "04-new").mkdir()
    subprocess.run(["git", "-C", str(report), "mv", "book/chapters/05-src/notes.txt",
                    "book/chapters/04-new/notes.txt"], check=True)
    (report / "book" / "chapters" / "04-new" / "_x.qmd").write_text(
        "## X\n" + (NO_COMMAND * 3).replace("{pin}", snapy[1]["pin"]))
    _set(report, {"ch04": 3})
    assert _gate(report, snapy, monkeypatch, base) >= 1
    out = capsys.readouterr().out
    assert "ch04: the baseline rises 0 -> 3 against" in out
    assert "relocation credit" not in out
