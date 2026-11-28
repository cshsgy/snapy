"""Tests of the PASS / PENDING / FAIL outcomes of tools/render_gate.py, without a render.

A missing snapy_report function is PENDING (lead's ruling); a snapy_report module that raises on import is
a FAIL. Run from the report directory: `python3 -m pytest tools/tests`.
"""

import shutil
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import render_gate  # noqa: E402

CHAPTER = '''\
# Demo {#sec-ch01}

```{python}
#| label: fig-ch01-deps
#| fig-cap: "Dependency map."
#| echo: false
from snapy_report.depmap import make_fig
make_fig()
```

```{python}
#| echo: false
from snapy_report.ch01.fig_x import (make_fig as fig_x,
                                     make_other)
import snapy_report.ch01.table_y
```
'''


@pytest.fixture
def book(tmp_path, monkeypatch):
    b = tmp_path / "book"
    (b / "chapters").mkdir(parents=True)
    (b / "chapters" / "01-demo.qmd").write_text(CHAPTER)
    pkg = tmp_path / "pkg" / "snapy_report"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "pkg"))
    monkeypatch.setenv("QUARTO_PYTHON", sys.executable)
    return b, pkg


def test_imports_are_found_with_their_lines(book):
    b, _ = book
    assert render_gate.snapy_report_imports(b) == [
        ("chapters/01-demo.qmd", 7, "snapy_report.depmap", "make_fig"),
        ("chapters/01-demo.qmd", 13, "snapy_report.ch01.fig_x", "make_fig"),
        ("chapters/01-demo.qmd", 13, "snapy_report.ch01.fig_x", "make_other"),
        ("chapters/01-demo.qmd", 15, "snapy_report.ch01.table_y", None),
    ]


def test_missing_functions_are_pending(book):
    b, pkg = book
    (pkg / "depmap.py").write_text("def other():\n    pass\n")      # module exists, function does not
    pending, errors = render_gate.gate_snapy_report(b)
    assert errors == []
    assert sorted(pending) == ["snapy_report.ch01.fig_x.make_fig", "snapy_report.ch01.fig_x.make_other",
                               "snapy_report.ch01.table_y", "snapy_report.depmap.make_fig"]


def test_existing_functions_pass(book):
    b, pkg = book
    (pkg / "depmap.py").write_text("def make_fig():\n    pass\n")
    (pkg / "ch01").mkdir()
    (pkg / "ch01" / "__init__.py").write_text("")
    (pkg / "ch01" / "fig_x.py").write_text("def make_fig():\n    pass\ndef make_other():\n    pass\n")
    (pkg / "ch01" / "table_y.py").write_text("")
    assert render_gate.gate_snapy_report(b) == ({}, [])


def test_a_module_that_raises_fails(book):
    b, pkg = book
    (pkg / "depmap.py").write_text("def make_fig(:\n")
    (pkg / "ch01").mkdir()
    (pkg / "ch01" / "__init__.py").write_text("import numpy_not_installed_anywhere\n")
    pending, errors = render_gate.gate_snapy_report(b)
    assert pending == {}
    assert any("depmap.make_fig: SyntaxError" in e for e in errors), errors
    assert any("ModuleNotFoundError" in e and "numpy_not_installed_anywhere" in e for e in errors), errors


def test_stand_ins_import_and_draw(book):
    """The stand-ins, imported as the render's IPython startup file imports them, in a fresh python."""
    import subprocess
    b, _ = book
    pending, _ = render_gate.gate_snapy_report(b)
    env = render_gate.pending_render_env(pending)
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, sys.argv[1])
        import _snapy_report_pending
        from snapy_report.depmap import make_fig
        from snapy_report.ch01.fig_x import make_fig as fig_x, make_other
        import snapy_report.ch01.table_y as t
        print(repr(make_fig())); print(repr(fig_x())); print(repr(make_other())); print(repr(t.make_fig()))
        fig = make_fig()        # an axes, or IPython's figure printer shows only the repr
        print(len(fig.axes), fig.axes[0].texts[0].get_url())
    """)
    try:
        p = subprocess.run([sys.executable, "-c", code, env["IPYTHONDIR"]], capture_output=True, text=True,
                           env=env)
    finally:
        shutil.rmtree(env["IPYTHONDIR"])
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == [
        "<snapy_report PENDING: snapy_report.depmap.make_fig>",
        "<snapy_report PENDING: snapy_report.ch01.fig_x.make_fig>",
        "<snapy_report PENDING: snapy_report.ch01.fig_x.make_other>",
        "<snapy_report PENDING: snapy_report.ch01.table_y.make_fig>",
        "1 snapy-report-pending:snapy_report.depmap.make_fig",
    ]


@pytest.mark.parametrize("results, release, code", [
    ({"a": "pass", "b": "pass"}, False, 0),
    ({"a": "pass", "snapy_report": "pending"}, False, render_gate.PENDING_EXIT),
    ({"a": "pass", "snapy_report": "pending"}, True, 1),
    ({"a": "fail", "snapy_report": "pending"}, False, 1),
])
def test_verdict(capsys, results, release, code):
    assert render_gate._verdict(results, release) == code
    out = capsys.readouterr().out
    if code == render_gate.PENDING_EXIT:
        assert "nothing can be released while anything is pending" in out


def test_no_book_is_pending(tmp_path, capsys):
    assert render_gate.main([str(tmp_path / "book")]) == render_gate.PENDING_EXIT


UNWRITTEN = Path(__file__).resolve().parent / "fixtures" / "book-unwritten"


def test_references_into_unwritten_chapters_are_listed_apart():
    expected = render_gate.unwritten_chapter_refs(UNWRITTEN)
    assert {lab: ch for lab, (ch, _) in expected.items()} == {
        "sec-ch05": "ch05", "sec-ch05-wbref": "ch05", "sec-ch07b": "ch07b", "eq-ch07b-vic": "ch07b"}
    # an id that is no chapter of STYLE 10.1 is a real dangling ref (a scheme not written yet is not:
    # see test_references_to_unwritten_schemes_of_a_written_chapter_are_expected)
    assert render_gate.gate_dangling_refs(UNWRITTEN) == 1


@pytest.fixture
def reader_standards_pass(monkeypatch):
    # These fixtures isolate the unwritten-reference verdict. Reader-standard
    # input validation is exercised separately on complete report fixtures.
    import gen_topic_index
    import check_reader_hygiene
    import gen_reader_reference
    import gen_forcing_contract
    import gen_boundary_contract
    import gen_scheme_contract
    import gen_scheme_native
    import glossary_registry
    import gen_glossary_appendix
    import gen_species_contract
    monkeypatch.setattr(gen_topic_index, "main", lambda argv: 0)
    monkeypatch.setattr(check_reader_hygiene, "main", lambda argv: 0)
    monkeypatch.setattr(gen_reader_reference, "check_glossary", lambda report: 0)
    monkeypatch.setattr(gen_reader_reference, "check_defaults", lambda report: 0)
    monkeypatch.setattr(gen_forcing_contract, "check", lambda report: 0)
    monkeypatch.setattr(gen_boundary_contract, "check", lambda report: 0)
    monkeypatch.setattr(gen_scheme_contract, "check", lambda report: 0)
    monkeypatch.setattr(gen_scheme_native, "check", lambda report: 0)
    monkeypatch.setattr(glossary_registry, "check", lambda report: 0)
    monkeypatch.setattr(gen_glossary_appendix, "check", lambda report: 0)
    monkeypatch.setattr(gen_species_contract, "check", lambda report: 0)


def test_unwritten_chapter_refs_fail_and_are_labelled_expected(capsys, reader_standards_pass):
    assert render_gate.main([str(UNWRITTEN), "--labels-only"]) == 1
    out = capsys.readouterr().out
    assert "✗ @sec-ch99: referenced in chapters/06-demo/_one.qmd but defined nowhere" in out
    assert ("✗ expected: unresolved cross-reference @sec-ch05 -> ch05 (not written yet), "
            "referenced in chapters/06-demo.qmd") in out
    assert "  fail: dangling refs\n" in out
    assert ("  fail, expected until those chapters are written: unwritten-chapter refs: "
            "@eq-ch07b-vic @sec-ch05 @sec-ch05-wbref @sec-ch07b") in out
    assert out.rstrip().endswith("GATE FAILED")


def test_only_expected_failures_are_said_so(tmp_path, capsys, reader_standards_pass):
    import shutil
    book = tmp_path / "book"
    shutil.copytree(UNWRITTEN, book)
    one = book / "chapters" / "06-demo" / "_one.qmd"
    one.write_text("## One scheme {#sec-ch06-one}\n\nNothing dangling here.\n")
    assert render_gate.main([str(book), "--labels-only"]) == 1
    assert "GATE FAILED: only the expected failures above" in capsys.readouterr().out


def test_quarto_warnings_for_expected_refs_are_not_counted_twice():
    out = ("WARNING (main.lua:14840) Unable to resolve crossref @sec-ch05\n"
           "WARNING (main.lua:14840) Unable to resolve crossref @sec-ch06-two\n")
    hits = render_gate.gate_stdout(out, {"sec-ch05"})
    assert hits == ["WARNING (main.lua:14840) Unable to resolve crossref @sec-ch06-two"]


def test_references_to_unwritten_schemes_of_a_written_chapter_are_expected(capsys):
    """The editor's ruling (1791649734): a reference into a scheme of a written chapter whose scheme file is not
    written yet (@sec-ch06-two, no chapters/06-demo/_two.qmd) is an expected failure listed by target id,
    like an unwritten chapter; a reference that is no chapter id at all (@sec-ch99) stays dangling."""
    expected = render_gate.unwritten_scheme_refs(UNWRITTEN)
    assert {lab: target for lab, (target, *_) in expected.items()} == {"sec-ch06-two": "ch06 scheme two"}
    assert render_gate.gate_dangling_refs(UNWRITTEN) == 1
    assert render_gate.main([str(UNWRITTEN), "--labels-only"]) == 1
    out = capsys.readouterr().out
    assert ("✗ expected: unresolved cross-reference @sec-ch06-two -> ch06 scheme two (no "
            "chapters/06-demo/_two.qmd yet), referenced in chapters/06-demo/_one.qmd") in out
    assert "  fail, expected until those schemes are written: unwritten-scheme refs: @sec-ch06-two" in out
    assert "✗ @sec-ch99: referenced in chapters/06-demo/_one.qmd but defined nowhere" in out
    assert "✗ @sec-ch06-two: referenced" not in out
    hits = render_gate.gate_stdout("WARNING (x) Unable to resolve crossref @sec-ch06-two\n",
                                   set(render_gate.unwritten_scheme_refs(UNWRITTEN)))
    assert hits == []


DEPMAP_CHAPTER = '''\
# Demo {#sec-ch05}

```{python}
#| label: fig-ch05-deps
#| fig-cap: "Dependency map."
#| echo: false
from snapy_report.depmap import make_fig
make_fig("ch05")
```

```{python}
#| label: fig-ch05-deps-wbref
#| fig-cap: "Dependency map of one scheme."
#| echo: false
from snapy_report.depmap import make_fig
make_fig('ch05-wbref')
```
'''

FAKE_DEPMAP = '''\
from collections import namedtuple
Status = namedtuple("Status", "state key detail")
def make_fig(target):
    pass
def status(target):
    if target == "ch05":
        return Status("pending", "snapy_report.depmap.ch05", "no scheme of 'ch05' is in the registry yet")
    if target == "ch05-wbref":
        return Status("ok", "snapy_report.depmap.ch05_wbref", "")
    raise ValueError("bad registry")
'''


def test_depmap_targets_are_found(book):
    b, _ = book
    (b / "chapters" / "05-demo.qmd").write_text(DEPMAP_CHAPTER)
    assert render_gate.depmap_targets(b) == [("chapters/05-demo.qmd", 8, "ch05"),
                                             ("chapters/05-demo.qmd", 16, "ch05-wbref")]


def test_empty_registry_target_is_pending(book):
    """The editor's ruling: a chapter with no scheme-registry entries is PENDING, under the key depmap reports."""
    b, pkg = book
    (b / "chapters" / "01-demo.qmd").unlink()
    (b / "chapters" / "05-demo.qmd").write_text(DEPMAP_CHAPTER)
    (pkg / "depmap.py").write_text(FAKE_DEPMAP)
    pending, errors = render_gate.gate_snapy_report(b)
    assert errors == []
    assert sorted(pending) == ["snapy_report.depmap.ch05"]
    assert pending["snapy_report.depmap.ch05"][2] == ["chapters/05-demo.qmd:8"]
    freeze = b / "_freeze" / "chapters" / "05-demo" / "execute-results"
    freeze.mkdir(parents=True)
    (freeze / "html.json").write_text('"<snapy_report PENDING: snapy_report.depmap.ch05>"')
    assert render_gate.gate_frozen_placeholders(b, pending) == 0      # still pending: not stale
    assert render_gate.gate_frozen_placeholders(b, {}) == 1           # registry filled: stale


def test_depmap_status_that_raises_fails(book):
    b, pkg = book
    (b / "chapters" / "01-demo.qmd").unlink()
    (b / "chapters" / "05-demo.qmd").write_text(DEPMAP_CHAPTER.replace('"ch05"', '"ch07"'))
    (pkg / "depmap.py").write_text(FAKE_DEPMAP)
    pending, errors = render_gate.gate_snapy_report(b)
    assert pending == {}
    assert any("depmap ch07" in e and "ValueError: bad registry" in e for e in errors), errors


def test_report_mode_checks_list_hits_and_switch_to_fail(tmp_path, monkeypatch, capsys):
    """The '?@' (rendered output) and hand-typed 'Chapter N' (sources) checks list every hit as file:line; they do
    not fail while REPORT_ONLY is True, and fail with one count per hit when it is False."""
    book = tmp_path / "book"
    (book / "chapters").mkdir(parents=True)
    (book / "chapters" / "03-demo.qmd").write_text("# Demo {#sec-ch03}\n\nSee Chapter 7 and chapter 12.\n")
    (book / "_book" / "chapters").mkdir(parents=True)
    (book / "_book" / "chapters" / "03-demo.html").write_text('<p>see <span class="quarto-unresolved-ref">?sec-ch05</span></p>\n')
    assert render_gate.gate_hand_typed_chapters(book) == 0
    assert render_gate.gate_unresolved_in_output(book) == 0
    out = capsys.readouterr().out
    assert "chapters/03-demo.qmd:3: Chapter 7" in out and "chapters/03-demo.qmd:3: chapter 12" in out
    assert "_book/chapters/03-demo.html:1: ?@sec-ch05" in out and "REPORT mode, not failing" in out
    monkeypatch.setattr(render_gate, "REPORT_ONLY", False)
    assert render_gate.gate_hand_typed_chapters(book) == 2
    assert render_gate.gate_unresolved_in_output(book) == 1
