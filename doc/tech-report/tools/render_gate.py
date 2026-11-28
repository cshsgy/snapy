#!/usr/bin/env python3
"""G1 render gate that can actually see LuaLaTeX's complaints.

THE PROBLEM THIS EXISTS FOR (ISSUES #9)
---------------------------------------
BOOKCRAFT section 6 defines G1 as a grep over `quarto render`'s **stdout** for
`warn|error|unable|missing character|undefined`. LuaLaTeX does not write most of
those to stdout. It writes them to `book/index.log`, **and quarto deletes that
log on a successful render** -- so for nine waves the gate greped for strings
that could never appear, and passed.

What it could not see, measured on the wave-11 build:

  * **42 `Missing character` warnings.** A missing character is not cosmetic:
    LuaLaTeX DROPS the glyph from the PDF. A tilde or a mu vanishes from a
    printed sentence while the HTML is fine and nothing is reported.
  * **2 multiply-defined labels** (`tbl-whipple-e`, `tbl-pcr-e1`), which make a
    cross-reference resolve to the wrong target.
  * 2034 overfull and 85 underfull boxes, which are typography rather than
    defects, and are reported here only as a count.

THE THIRD BLIND SPOT (ISSUES #10), FOLDED IN 2026-09-02
-------------------------------------------------------
Maths written into a GENERATED TABLE CELL reaches the reader as literal LaTeX,
and a cross-reference in a cell does not resolve -- in the HTML and the PDF
alike. That is invisible to every gate above: the source is well formed, so
`check_math_spans.py` passes; the stdout gate sees nothing; and the log only
notices when a macro happens to be a character-mapping one, which caught 4 of
390. The instrument is a scan of the RENDERED `<th>`/`<td>` contents, which
lives in `tools/check_table_cells.py` and is run from here so that G1 covers
it. **It is folded in rather than left standalone because ISSUES #9's whole
lesson is that a gate nobody runs is a gate that does not exist.**

ADAPTED FOR THE snapy TECHNICAL REPORT (STYLE.md 10.9)
------------------------------------------------------
Imported from the house books unchanged (commit "move the house render-gate
scripts") and changed only where this report differs:

  * Sources: a chapter is `book/chapters/NN-slug.qmd` plus its scheme files
    `book/chapters/NN-slug/_<scheme>.qmd` (STYLE 10.1), so every source gate
    reads `chapters/**/*.qmd` and names files by their path under `book/`.
  * The LaTeX log is `book/index.log`, or the newest `book/*.log` if the book
    sets another output name; `--log` overrides.
  * Overfull boxes wider than 10 pt FAIL (STYLE 10.9); smaller ones and the
    underfull ones are still only counted. `--overfull-pt` sets the limit.
  * Which of those failures are reported rather than failed is
    `tools/render-gate-policy.json`, applied here on every run and judged by
    CI through this script's exit status, so the documented command and the
    workflow cannot disagree about a run. `--strict` returns the untreated
    verdict; put it after the book path (`render_gate.py book/ --strict`).
  * After the render the HTML and the PDF must both be in `book/_book/`
    (STYLE 10.9 renders both from one command; quarto deletes the HTML of a
    book when `--to pdf` is rendered on its own).
  * The freeze gate asks for a tracked `_freeze/` cache only for chapters
    that have a python cell; a chapter without one has nothing to freeze.
  * The table-cell fix message names this report's route (a markdown table).
  * Three outcomes, PASS, PENDING and FAIL (lead's ruling): a `snapy_report`
    function that a chapter imports but that is not written yet is PENDING,
    not a failure, and so is a book with no `_quarto.yml` yet. The render runs
    with a placeholder figure for each pending function, so every other gate
    still sees the whole book. Nothing is released while anything is pending.

USAGE
-----
    python tools/render_gate.py book/            # render, then gate
    python tools/render_gate.py book/ --no-render   # gate an existing log
    python tools/render_gate.py book/ --labels-only # source gates only, no build
    python tools/render_gate.py book/ --release     # PENDING counts as a failure

Exit status: 0 PASS, 1 FAIL, 2 PENDING (nothing failed, something is pending).

⚠️ **Do not run this while another session is rendering the same book.**
`_book/` and `_freeze/` are shared state, which is why chapter writers are
forbidden to run quarto at all. To gate a book another session owns, copy the
book directory elsewhere and run there -- the `_freeze` cache is committed, so
the cells do not re-run and the copy costs a couple of minutes.

⚠️ `-M latex-clean:false` is what keeps `index.log` alive. It is passed on the
command line rather than written into `_quarto.yml` so that this gate needs no
change to a file other sessions own.
"""

import json
import re
import subprocess
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # for book_chapters
import generated_values  # noqa: E402  (render-time values of the generated appendices)

# things that are defects, not taste
FATAL = {
    "missing character": (
        r"Missing character: There is no (.+?) \(U\+([0-9A-F]+)\)",
        "a glyph is being DROPPED from the PDF"),
    "multiply defined": (
        r"LaTeX Warning: Label `([^']+)' multiply defined",
        "a cross-reference resolves to the wrong target"),
    "undefined reference": (
        r"LaTeX Warning: Reference `([^']+)' on page (\d+) undefined",
        "a cross-reference points at nothing"),
    "undefined citation": (
        r"LaTeX Warning: Citation `([^']+)' on page (\d+) undefined",
        "a citation is missing from the bibliography"),
}

# real, but typography rather than correctness: counted, never fatal
NOISE = {
    "overfull hbox": r"Overfull \\hbox",
    "underfull hbox": r"Underfull \\hbox",
    "overfull vbox": r"Overfull \\vbox",
}


def _chapter_sources(book):
    """Every chapter file and every included scheme file (STYLE 10.1), sorted."""
    return sorted((book / "chapters").rglob("*.qmd"))


def _rel(book, q):
    return str(q.relative_to(book))


def render(book, env=None):
    cmd = ["quarto", "render", str(book), "-M", "latex-clean:false"]
    print("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout + p.stderr


#: Quarto's per-cell progress line, e.g. ``Cell 13/25: 'tbl-p9-errorprop'........Done``.
#: 🔴 **It carries the cell's own LABEL, so a label containing "error", "warn" or
#: "undefined" makes the stdout gate fail on a clean render.** That fired for real on
#: 2026-09-04: ch43's `tbl-p9-errorprop` had always existed but had always been FROZEN, so
#: the line was never printed; clearing ch43's freeze for an unrelated constant fix
#: re-executed the cell and the gate reported ``GATE FAILED: 2 problems`` against a render
#: with nothing wrong in it. A false positive in a gate is not harmless — it is how a gate
#: gets ignored, which is `ISSUES.md` #9's whole lesson running in reverse.
#: ⚠ The exclusion is deliberately narrow: it matches the progress line's exact shape and
#: only up to its terminating ``Done``, so a genuine error printed on any other line — or
#: after a cell fails rather than completing — still reaches the gate.
_CELL_PROGRESS = re.compile(r"^\s*Cell \d+/\d+: '[^']*'\.*Done\s*$")


_UNRESOLVED = re.compile(r"Unable to resolve crossref @([\w-]*\w)")


def gate_stdout(out, expected=()):
    """The historical check. Necessary, and by itself not sufficient.

    Quarto's "Unable to resolve crossref @x" for an x in `expected` (a reference into a chapter not
    written yet, or a scheme not written yet) is left to the unwritten-chapter and unwritten-scheme gates,
    which already count it.
    """
    pat = re.compile(r"warn|error|unable|missing character|undefined", re.I)
    hits = [l for l in out.splitlines()
            if pat.search(l) and "IPKernelApp" not in l
            and not _CELL_PROGRESS.match(l)
            and not (_UNRESOLVED.search(l) and _UNRESOLVED.search(l).group(1) in expected)]
    return hits


def _source_label_counts(book):
    """How many times each label is DEFINED in the .qmd sources.

    Needed to tell a real collision from longtable's own noise: see
    `gate_log`.
    """
    counts = {}
    for q in _chapter_sources(book):
        text = q.read_text()
        for lab in re.findall(r"\{#([a-zA-Z][\w-]*)\}", text):
            counts.setdefault(lab, []).append(_rel(book, q))
        for lab in re.findall(r"^#\|\s*label:\s*(\S+)", text, re.M):
            counts.setdefault(lab, []).append(_rel(book, q))
    return counts


#: The ONLY three crossref categories quarto declares with `kind = "float"`, and therefore
#: the only three that carry a `caption_location`.  Read out of quarto 1.10.18's own
#: `share/filters/main.lua`, not remembered:
#:
#:     awk '/kind = "float"/{f=1} f && /ref_type = "/{print; f=0}' main.lua | sort -u
#:
#: ⚠️ **Quarto registers eleven `ref_type`s** -- cau, fig, imp, lst, nte, prf, rem, sol,
#: tbl, tip, wrn -- **and only these three are floats.**  The other eight are callout and
#: theorem families with no caption location at all, so a CAPTIONED cell labelled `sol-...`
#: crashes exactly as `chk-...` does.  ★ This gate's own first draft listed `sol` as safe
#: and would have passed the wave-16 defect that motivated it; it was caught by reading
#: quarto's source instead of trusting the category list.  Do not widen this tuple without
#: re-running that awk.
FLOAT_PREFIXES = ("fig", "tbl", "lst")


def gate_float_label_prefixes(book):
    """A captioned cell whose label prefix is not a quarto crossref type KILLS THE RENDER.

    Found the hard way in wave 16.  ch46 carried five cells labelled `chk-...` and one
    `sol-...`, each with a real `tbl-cap`/`fig-cap`.  A caption makes the cell a FLOAT, and
    quarto then looks the label's prefix up in `crossref.categories.by_ref_type` to decide
    where to put the caption.  For an unknown prefix that lookup returns nil, and the code
    that is supposed to report the problem does

        error("Invalid caption location for float: " .. obj.identifier ..
              " requested " .. result .. ...)

    with `result` nil -- so it dies concatenating nil *while trying to name the offender*,
    and the render's only output is a Lua stack trace with no filename, no line and no
    identifier in it.  ⚠️ **The three existing label gates are all blind to this**: the
    label is unique, so the collision gate passes; nothing references it, so the dangling
    gate passes; and it is a source-level defect, so the log gate never runs because the
    render never completes.  The book had 1503 clean labels and could not build.

    ⚠️ **An UNcaptioned cell with any label at all is fine** -- quarto only classifies a
    cell as a float when it has a caption -- so this must test the pair, not the label.
    Divs (`::: {#exm-...}`, `{#eq-...}`) are not cells and are not in scope here.
    """
    bad = []
    for q in _chapter_sources(book):
        label = None
        for n, line in enumerate(q.read_text().splitlines(), 1):
            m = re.match(r"^#\|\s*label:\s*(\S+)", line)
            if m:
                label = (m.group(1), n)
                continue
            if re.match(r"^#\|\s*(fig-cap|tbl-cap|lst-cap)\s*:", line) and label:
                if label[0].split("-")[0] not in FLOAT_PREFIXES:
                    bad.append((_rel(book, q), label[1], label[0]))
                label = None
            elif line.strip() == "```":
                label = None
    if not bad:
        print("\n[float gate] every captioned cell carries a crossref-typed label")
        return 0
    print(f"\n[float gate] {len(bad)} captioned cells whose label prefix is not a quarto"
          " crossref type -- THIS KILLS THE RENDER WITH NO FILENAME IN THE ERROR:")
    for fname, n, lab in bad:
        print(f"   {fname}:{n}  {lab}   (prefix '{lab.split('-')[0]}')")
    return len(bad)


#: The two-letter font-selection commands LaTeX2e deprecated in 1994.  Under a KOMA-Script
#: class -- and this book is `scrbook` -- they are not deprecated but **removed**, and using
#: one is a hard error.
_OLD_FONT_COMMANDS = ("rm", "bf", "it", "sf", "tt", "cal", "sl", "sc", "mit")
_OLD_FONT_RE = re.compile(r"\\(" + "|".join(_OLD_FONT_COMMANDS) + r")(?![a-zA-Z])")
#: A markdown inline-code span, so that quoting the rule does not violate it.
_INLINE_CODE = re.compile(r"`{1,3}[^`]*`{1,3}")

#: A python cell that hands a string to pandoc rather than to matplotlib.  This is the
#: discriminator, and it is a property of the CELL and not of the line: a label is routinely
#: built on one line (``lab = f"... $M_{\\rm J}$"``) and handed to matplotlib on another, so
#: a line-local test misreports four of ch10's five and ch47's one.  A cell that emits
#: markdown is on pandoc's path and is fatal; a cell that does not is a figure, and
#: matplotlib's own mathtext accepts the old commands.
#: ⚠ ``show`` is `classic_papers.tables.show`, this book's DataFrame renderer -- the very
#: route `ISSUES` #10 was filed about.
_MARKDOWN_EMITTER = re.compile(
    r"\bMarkdown\s*\(|to_markdown\s*\(|\bdisplay\s*\(|\bHTML\s*\(|"
    r"\btables\.show\s*\(|(?<![\w.])show\s*\(|to_latex\s*\(|to_html\s*\(")


def gate_old_font_commands(book):
    """A bare ``\\rm`` in maths is FATAL under `scrbook`, and the error names no .qmd.

    🔴 **Measured by three renders on 2026-09-04 (wave 17), because `ISSUES` #21 said this
    rule was FALSE and it is not.**  A minimal document on this book's own PDF settings dies
    with

        Class scrbook Error: undefined old font command `\\rm'.

    at the first inline occurrence, and dies identically when the only occurrence is inside
    a **generated table cell** -- so the "narrower, untested" hypothesis #21 raised is the
    same failure, not `ISSUES` #10's family.

    ⚠️ **But the rule's stated REASON was wrong, and that is what to carry forward.**
    `COMMON.md` said *"LuaLaTeX dies"*.  LuaLaTeX does not: the killer is the **KOMA-Script
    document class**, which removes the old two-letter font commands that plain `book` only
    deprecates.  A sibling book on `article` would read the rule, fail to reproduce it, and
    learn to discount the playbook -- which is exactly the harm #21 was filed about,
    arriving through the other door.

    ⚠️ **And #21's four counter-example chapters are a false positive of the FINDING.**
    ch07 (3), ch10 (6), ch22 (2) and ch26 (9) do contain a bare ``\\rm``, and **all twenty
    occurrences are matplotlib label strings inside python cells** -- ``set_ylabel``,
    ``label=``, ``ax.text`` -- which matplotlib renders with its own mathtext and which
    never reach LaTeX.  Confirmed by a fourth render: a figure whose axis label carries
    ``$\\tau_{\\rm I}$`` builds clean.  The finding was made by grepping ``\\rm`` over whole
    files without excluding code cells.  **The rule stands; the counter-examples were never
    in maths that LaTeX sees.**
    """
    fatal, allowed = [], []
    for q in _chapter_sources(book):
        lines = q.read_text().splitlines()
        # Split into cells first: the fatal/safe distinction is a property of the CELL, not
        # of the line.  A label string is often built on one line (`lab = "... $M_{\\rm J}$"`)
        # and handed to matplotlib on another, so a line-local test misreports it.
        blocks, cur, fence = [], [], None
        for n, line in enumerate(lines, 1):
            m = re.match(r"^\s*(`{3,})", line)
            if m and fence is None:
                blocks.append(("prose", cur)); cur = []
                fence = m.group(1); cur = [(n, line)]
            elif m and fence is not None and line.strip() == fence:
                cur.append((n, line)); blocks.append(("cell", cur)); cur = []; fence = None
            else:
                cur.append((n, line))
        blocks.append(("prose" if fence is None else "cell", cur))

        for kind, body in blocks:
            text = "\n".join(l for _, l in body)
            emits_markdown = bool(_MARKDOWN_EMITTER.search(text))
            for n, line in body:
                # An inline-code span is not maths: a chapter documenting this very rule
                # would otherwise fail the gate for quoting it.  Stripped only in PROSE --
                # inside a cell a backtick is a python string delimiter, not markdown.
                probe = _INLINE_CODE.sub("", line) if kind == "prose" else line
                hits = _OLD_FONT_RE.findall(probe)
                if not hits:
                    continue
                if kind == "cell" and not emits_markdown:
                    allowed.append((_rel(book, q), n, hits))
                else:
                    fatal.append((_rel(book, q), n, hits, line.strip()[:80]))
    if allowed:
        n = sum(len(h) for _, _, h in allowed)
        print(f"\n[font gate] {n} old font commands on {len(allowed)} lines, all inside "
              "figure cells (matplotlib mathtext, never reaches LaTeX -- allowed)")
    if not fatal:
        print("[font gate] no old font command reaches LaTeX")
        return 0
    print(f"\n[font gate] {len(fatal)} old font commands on a path to LaTeX -- EACH ONE"
          " KILLS THE PDF WITH 'undefined old font command' AND NAMES NO .qmd:")
    for fname, n, hits, text in fatal:
        print(f"   {fname}:{n}  {sorted(set(hits))}  {text}")
    return len(fatal)


def gate_cross_chapter_labels(book):
    """A label defined in TWO chapters, checked at SOURCE level and before the render.

    The wave-11/ISSUES-#9 finding: four labels were defined in two chapters each -- one of
    them introduced by wave 11 itself -- because the mandatory cross-chapter diff had only
    ever checked symbols and numbers. **A cross-reference that resolves to the wrong chapter
    is invisible to every test**, produces no warning quarto reports on stdout, and in the
    PDF only shows as a `multiply defined` line inside a log quarto deletes. Checking the
    sources costs milliseconds and needs no build, so it runs first.
    """
    counts = _source_label_counts(book)
    clashes = {lab: files for lab, files in counts.items() if len(set(files)) > 1}
    if not clashes:
        print(f"\n[label gate] {len(counts)} labels, none defined in more than one chapter")
        return 0
    print(f"\n[label gate] {len(clashes)} labels defined in more than one chapter:")
    for lab, files in sorted(clashes.items()):
        print(f"  ✗ {lab}: " + ", ".join(sorted(set(files))))
    print("    a cross-reference to one of these resolves to whichever chapter quarto saw last")
    return len(clashes)


def _xrefs(book):
    """A `@sec-`/`@eq-`/`@fig-`/`@tbl-` reference with no definition, or one quarto cannot reach.

    The label gate above catches a label defined TWICE. This catches the opposite and equally
    silent failure: a reference to a label that is defined nowhere, or -- the case that
    actually occurred -- defined on a heading DEEPER than quarto's cross-reference depth, so
    the label exists in the source and resolves to nothing in the output. ch40 labelled a
    `####` heading and was the only chapter in the book to do so; every other chapter stops at
    `###`. A dead cross-reference is invisible to every test and reads as a missing number.

    Returns (used, defined, deep): {label: files referencing it}, the defined labels, and the
    labels quarto cannot reach with the reason.
    """
    defined = set()
    deep = {}
    for q in _chapter_sources(book):
        text = q.read_text()
        defined |= set(re.findall(r"\{#([a-zA-Z][\w-]*)\}", text))
        defined |= set(re.findall(r"^#\|\s*label:\s*(\S+)", text, re.M))
        for lvl, lab in re.findall(r"^(#{4,})\s+.*\{#(sec-[\w-]+)\}", text, re.M):
            deep[lab] = (_rel(book, q), len(lvl), "on a level-%d heading, too deep to reference"
                         % len(lvl))
        # 🔴 A `sec-` label on a heading INSIDE a fenced div (`::: {#exm-...}` and friends)
        # is unreachable BY CONSTRUCTION: a heading inside a div is not a document section,
        # so quarto registers no section to point at and `@sec-...` renders as literal text.
        # ch40 hit this, and it survived being "fixed" from #### to ### because the depth was
        # never the real cause. Reference the DIV (`@exm-...`) instead.
        depth = 0
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith(":::"):
                depth += 1 if re.match(r":::+\s*\{", s) else (-1 if set(s) <= {":"} else 0)
                depth = max(depth, 0)
                continue
            if depth:
                m = re.match(r"#{1,6}\s+.*\{#(sec-[\w-]+)\}", s)
                if m:
                    deep[m.group(1)] = (_rel(book, q), 0, "on a heading INSIDE a fenced div, which "
                                        "is not a section -- reference the div itself")
    used, values = {}, None
    for q in _chapter_sources(book):
        text = q.read_text()
        # a generated appendix's render-time values (a symbol's first use) are references the reader gets too
        if generated_values.SHORTCODE.search(text):
            values = values if values is not None else generated_values.computed(book)
            text = generated_values.expand(text, values)
        # ⚠️ a trailing hyphen belongs to the PROSE, not the label: `@eq-foo--` is
        # `@eq-foo` followed by a markdown en-dash, and a greedy `[\w-]+` invents two
        # dangling references in ch09 that do not exist. The label cannot end in a hyphen.
        for lab in re.findall(r"@((?:sec|eq|fig|tbl|exr|exm|thm|prp|lem)-(?:[\w-]*\w)?)", text):
            used.setdefault(lab, set()).add(_rel(book, q))
    return used, defined, deep


#: Chapter identifiers of STYLE 10.1: ch01 ... ch17, split chapters with a letter (ch07b), appendices.
_CHAPTER_ID = re.compile(r"^(?:sec|eq|fig|tbl|exr|exm|thm|prp|lem)-((?:ch(?:0[1-9]|1[0-7])[a-z]?)|app[a-z])"
                         r"(?:-|$)")


def unwritten_chapter_refs(book):
    """{label: (chapter id, files)} for references into a chapter that is not in the book yet.

    The target is a valid chapter identifier (STYLE 10.1) whose chapter heading `{#sec-<id>}` is
    defined nowhere. The editor's ruling: these are EXPECTED failures until that chapter exists. They stay
    failures, but they are listed apart from real dangling references so a reviewer can tell them
    apart. A reference into a chapter that exists is either into a scheme not written yet
    (unwritten_scheme_refs) or a real dangling one.
    """
    used, defined, _ = _xrefs(book)
    out = {}
    for lab, files in used.items():
        m = _CHAPTER_ID.match(lab)
        if lab not in defined and m and f"sec-{m.group(1)}" not in defined:
            out[lab] = (m.group(1), files)
    return out


#: A label inside a scheme: <kind>-<chapter id>-<scheme>[-<what>] (STYLE 10.1).
_SCHEME_LABEL = re.compile(r"^(?:sec|eq|fig|tbl|exr|exm|thm|prp|lem)-((?:ch(?:0[1-9]|1[0-7])[a-z]?)|app[a-z])"
                           r"-([a-z0-9]+)(?:-|$)")


def _chapter_stems(book):
    """{chapter id: file stem} of the chapter files: chapters/06-gravity-energy.qmd gives ch06."""
    stems = {}
    for q in (book / "chapters").glob("*.qmd"):
        m = re.match(r"(?:(\d\d[a-z]?)|app([a-z]))-", q.name)
        if m:
            stems[f"ch{m.group(1)}" if m.group(1) else f"app{m.group(2)}"] = q.stem
    return stems


def unwritten_scheme_refs(book):
    """{label: (target, files, scheme file)} for references into a scheme not written yet, in a written chapter.

    The editor's ruling (1791649734): like a reference into an unwritten chapter, a reference whose target is a
    scheme of a chapter that exists, but whose scheme file `chapters/<chapter>/_<scheme>.qmd` does not
    exist yet, is an EXPECTED failure, listed by target id. If the scheme file exists, an undefined
    label in it is a real dangling reference.
    """
    used, defined, _ = _xrefs(book)
    chapters = unwritten_chapter_refs(book)
    stems = _chapter_stems(book)
    out = {}
    for lab, files in used.items():
        m = _SCHEME_LABEL.match(lab)
        if lab in defined or lab in chapters or not m:
            continue
        ch, scheme = m.groups()
        if f"sec-{ch}" not in defined or ch not in stems:
            continue
        rel = f"chapters/{stems[ch]}/_{scheme}.qmd"
        if not (book / rel).exists():
            out[lab] = (f"{ch} scheme {scheme}", files, rel)
    return out


def gate_dangling_refs(book):
    """References that resolve to nothing, other than those into a chapter or scheme not written yet."""
    used, defined, deep = _xrefs(book)
    expected = {**unwritten_chapter_refs(book), **unwritten_scheme_refs(book)}
    dangling = {l: f for l, f in used.items() if l not in defined and l not in expected}
    unreachable = {l: v for l, v in deep.items() if l in used}
    if not dangling and not unreachable:
        print(f"\n[xref gate] {len(used)} distinct references; none dangling or unreachable"
              + (f" ({len(expected)} into chapters or schemes not written yet: see their gates)"
                 if expected else ""))
        return 0
    print(f"\n[xref gate] {len(dangling)} dangling, {len(unreachable)} unreachable reference(s)")
    for lab, files in sorted(dangling.items()):
        print(f"  ✗ @{lab}: referenced in {', '.join(sorted(files))} but defined nowhere")
    for lab, (fn, _lvl, why) in sorted(unreachable.items()):
        print(f"  ✗ @{lab}: defined in {fn} {why}")
    return len(dangling) + len(unreachable)


def gate_unwritten_chapter_refs(book):
    """EXPECTED failure (editor's ruling): references into chapters that are not written yet."""
    expected = unwritten_chapter_refs(book)
    chapters = sorted({c for c, _ in expected.values()})
    print(f"\n[unwritten-chapter gate] {len(expected)} reference(s) into {len(chapters)} chapter(s) "
          "not written yet" + (f": {', '.join(chapters)}" if chapters else ""))
    if expected:
        print("    EXPECTED failure until those chapters exist; not a dangling reference")
    for lab, (ch, files) in sorted(expected.items()):
        print(f"  ✗ expected: unresolved cross-reference @{lab} -> {ch} (not written yet), "
              f"referenced in {', '.join(sorted(files))}")
    return len(expected)


def gate_unwritten_scheme_refs(book):
    """EXPECTED failure (editor's ruling): references into schemes of written chapters, not written yet."""
    expected = unwritten_scheme_refs(book)
    schemes = sorted({t for t, _, _ in expected.values()})
    print(f"\n[unwritten-scheme gate] {len(expected)} reference(s) into {len(schemes)} scheme(s) "
          "not written yet" + (f": {', '.join(schemes)}" if schemes else ""))
    if expected:
        print("    EXPECTED failure until those schemes exist; not a dangling reference")
    for lab, (target, files, rel) in sorted(expected.items()):
        print(f"  ✗ expected: unresolved cross-reference @{lab} -> {target} (no {rel} yet), "
              f"referenced in {', '.join(sorted(files))}")
    return len(expected)


def find_log(book):
    """`index.log`, or the newest log in the book directory if the book names its output."""
    log = book / "index.log"
    if log.exists():
        return log
    logs = sorted(book.glob("*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else log


#: STYLE 10.9: an overfull box wider than this many points fails the gate.
OVERFULL_PT = 10.0
_OVERFULL = re.compile(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)[^\n]*")


def gate_log(log_path, book, overfull_pt=OVERFULL_PT):
    if not log_path.exists():
        return None, {}, {}
    text = log_path.read_text(errors="replace")
    fatal = {}
    wide = [m.group(0) for m in _OVERFULL.finditer(text) if float(m.group(1)) > overfull_pt]
    if wide:
        fatal[f"overfull box > {overfull_pt:g} pt"] = (
            wide, "text runs into the margin or over the next column (STYLE 10.9)")
    for name, (pat, why) in FATAL.items():
        found = re.findall(pat, text)

        if name == "multiply defined" and found:
            # ⚠️ longtable writes a caption's \label to the .aux TWICE (it
            # typesets the caption once for the first head and again for
            # continuation heads), so EVERY captioned longtable in the book
            # raises this warning with a single \label in the .tex and a
            # single definition in the source. On the wave-11 build that is
            # 16 of 19 warnings -- pure noise.
            #
            # The real defect is a label defined in more than one PLACE, which
            # makes a cross-reference resolve to the wrong target. Wave 11
            # found three: eq-annulus and eq-scaleheight in both ch22 and
            # ch23, and eq-teq in both ch14 and ch22.
            #
            # A gate that reports the 16 cannot ever come back clean, and a
            # gate that cannot come back clean gets ignored -- which is how
            # this book lost its render gate for nine waves in the first
            # place. So cross-check the source and keep only the real ones.
            src = _source_label_counts(book)
            real = [lab for lab in found if len(set(src.get(lab, []))) > 1]
            noisy = len(found) - len(real)
            if noisy:
                print(f"    ({noisy} 'multiply defined' warnings are "
                      f"longtable's own double-write, not collisions)")
            found = [f"{lab}  (defined in {', '.join(sorted(set(src[lab])))})"
                     for lab in real]

        if found:
            fatal[name] = (found, why)
    noise = {n: len(re.findall(p, text)) for n, p in NOISE.items()}
    return text, fatal, noise


def _gate_table_cells(book):
    """ISSUES #10: rendered table cells carrying unrendered maths or a dead ref."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    try:
        from check_table_cells import scan_path
    except ImportError:
        print("\n[table-cell gate] check_table_cells.py NOT FOUND -- gate is BLIND")
        return 1
    rendered = book / "_book"
    if not rendered.is_dir():
        print(f"\n[table-cell gate] {rendered} not built -- gate is BLIND")
        return 1
    found = scan_path(rendered)
    total = sum(len(v) for v in found.values())
    n_html = len(list(rendered.rglob("*.html")))
    if not total:
        print(f"\n[table-cell gate] {n_html} rendered files, no cell carries "
              "literal $...$ or an unresolved (@ref)")
        return 0
    print(f"\n[table-cell gate] {total} offending cells in {len(found)} of {n_html} files")
    for f, hits in sorted(found.items()):
        print(f"  ✗ {f.name}: {len(hits)}")
        for kind, cell in hits[:3]:
            print(f"       [{kind}] {' '.join(cell.split())[:90]}")
    print("    fix: emit the table as markdown, Markdown(df.to_markdown(index=False)),"
          " not a bare DataFrame")
    return total



def gate_orphan_caption_blocks(book):
    """A pandoc caption block (``: text {#tbl-x}``) placed after a CODE FENCE binds to
    nothing, and reaches the reader as a literal colon and a literal ``{#tbl-x}``.

    Found in wave 28, in ch51, three times.  The trailing-caption form is valid pandoc
    *after a literal pipe table written in the .qmd*; it is NOT valid after a ```{python}
    cell, because the cell's output is not a table pandoc has parsed.  A code cell takes its
    caption as a cell option -- ``#| label: tbl-x`` plus ``#| tbl-cap: '...'``.

    🔴 **The reason this needs its own gate is that only the REFERENCED ones are visible.**
    Two of ch51's three were caught, and only because ``@tbl-...`` appeared elsewhere in the
    chapter, so quarto emitted "Unable to resolve crossref" and the stdout gate saw it.  The
    third was never referenced, so:

      * the dangling-reference gate passed  -- nothing referenced it;
      * the cross-chapter label gate passed -- the label is unique;
      * the float-prefix gate passed        -- it is not a captioned CELL, so not a float;
      * the table-cell gate passed          -- the stray text is a paragraph, not a cell;
      * the log gate passed                 -- LaTeX is perfectly happy to typeset a colon;
      * the render exited 0.

    The rendered page carried ``: The paper's own Section 5 condensate ranges ... {#tbl-ac-
    ladder}`` in the body text.  ★ **Being referenced is what made the other two visible, and
    that is a property of the rest of the chapter, not of the defect.**

    The rule is deliberately narrow, so it cannot fire on a legitimate pipe-table caption:
    a caption block whose nearest preceding non-blank line is a CLOSING code fence.
    """
    bad = []
    for q in _chapter_sources(book):
        lines = q.read_text().splitlines()
        for n, line in enumerate(lines):
            if not re.match(r"^:\s+\S", line):
                continue
            j = n - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and lines[j].strip() == "```":
                bad.append((_rel(book, q), n + 1, line.strip()[:60]))
    print(f"\n[caption gate] {len(bad)} caption block(s) attached to a code fence")
    for name, n, txt in bad:
        print(f"  \u2717 {name}:{n}  {txt}...")
        print("       a code cell takes '#| label:' + '#| tbl-cap:'; a trailing")
        print("       ': caption {#tbl-x}' after a fence binds to nothing")
    if not bad:
        print("    every caption block follows a table pandoc has parsed")
    return len(bad)


def gate_unbalanced_quotes(book):
    """A paragraph of prose carrying an ODD number of ``"`` marks.

    Found in wave 29, in ch52, by an adversarial reviewer reading the rendered page -- one
    stray opening quote inside a block quotation, which reached the printed page as a
    quotation mark with nothing to close it.

    🔴 **No other gate in this file can see it, and the reason generalises.** Every other gate
    here asks whether a *structure* resolves -- a label, a reference, a float type, a font
    command, a caption's binding.  A stray quotation mark resolves perfectly: pandoc emits a
    curly quote, LaTeX typesets it, the render exits 0, the log is clean.  It is wrong only to
    a reader.  ★ **A defect that is well-formed at every level except meaning needs a gate that
    counts rather than one that parses.**

    The rule is deliberately crude, and the measurement is what makes it usable: run over all
    54 chapter files of a book that had shipped 51 reviewed chapters, it returns **zero**.
    A gate that cannot come back empty on good prose is worse than no gate, because readers
    learn to ignore it -- so the zero is the licence, not the idea.

    ⚠️ Code is excluded before counting, in three forms, because there a quote is syntax and
    not punctuation: fenced blocks, inline code spans, and ``#|`` cell options.  ⚠️ It counts
    the ASCII ``"`` only.  Typographic quotes are already balanced by whoever inserted them
    and an apostrophe would make ``'`` unusable as a parity mark.
    """
    bad = []
    for q in sorted(_chapter_sources(book) + list(book.glob("index.qmd"))):
        t = q.read_text()
        t = re.sub(r"^```.*?^```", "", t, flags=re.S | re.M)   # fenced code
        t = re.sub(r"`[^`\n]*`", "", t)                        # inline code spans
        t = re.sub(r"^\s*#\|.*$", "", t, flags=re.M)           # cell options
        for para in re.split(r"\n\s*\n", t):
            n = para.count('"')
            if n % 2:
                first = next((l for l in para.strip().splitlines() if l.strip()), "")
                bad.append((_rel(book, q), n, first.strip()[:70]))
    print(f"\n[quote gate] {len(bad)} paragraph(s) with an odd number of \" marks")
    for name, n, txt in bad:
        print(f"  \u2717 {name}  ({n} quotes)  {txt}...")
        print("       a quotation mark with nothing to close it reaches the printed page;")
        print("       every other gate here passes it, because it is well-formed")
    if not bad:
        print("    every prose paragraph closes the quotations it opens")
    return len(bad)


def gate_maths_span_broken_by_a_bullet(book):
    r"""An inline ``$...$`` span, INSIDE A LIST ITEM, whose continuation line starts with a
    markdown bullet marker.  ⚠️ **Fatal — it kills the render.**

    Added at the W34 close (2026-09-14, `ISSUES` #54), found by ch57 failing its first full
    render after every source gate above had passed.

    **The mechanism, reproduced in isolation before the rule was written.**  Inside a list
    item -- and ``(a)``, ``(b)`` are list items, pandoc renders them with
    ``\def\labelenumi{(\alph{enumi})}`` -- a continuation line beginning ``- `` or ``+ ``
    starts a REAL sibling bullet list.  That ends the paragraph, and with it the open inline
    maths span.  The opening ``$`` never closes, pandoc escapes the lot as text
    (``n\_\{\mathrm{H^0}\}Q\_\{31\}\$``) **but passes ``\mathrm`` through as a control
    sequence**, and LuaLaTeX dies with *"\mathrm allowed only in math mode"*.

    ⛔ **THE LIST-ITEM CONDITION IS THE WHOLE RULE, AND A PATTERN MATCH WITHOUT IT IS WRONG.**
    In ordinary prose a leading ``-``/``+`` is a lazy continuation and pandoc keeps the maths
    span intact.  A first scan for "maths span containing a line that starts with a bullet"
    returned **three** hits across the book -- ch02:39 and ch54:589 as well as ch57 -- and
    **both of the others are in plain prose and have been rendering correctly for dozens of
    waves.**  They were checked against `pandoc -t latex` and left alone.  ★ Had the rule been
    written from the pattern rather than from the mechanism, this gate's first act would have
    been to send someone to "fix" two shipped chapters that are not broken.

    The fix is to reflow so no continuation line of an open span begins with a bullet marker,
    or to promote the equation to a display ``$$...$$``.
    """
    bad = []
    bullet = re.compile(r"^\s{0,7}[-*+]\s")
    item = re.compile(r"^\s{0,7}(?:[-*+]\s|\(?[0-9a-z]{1,3}[.)]\s)")
    for q in sorted(_chapter_sources(book) + list(book.glob("index.qmd"))):
        t = q.read_text()
        t = re.sub(r"^```.*?^```", lambda m: "\n" * m.group(0).count("\n"), t,
                   flags=re.S | re.M)
        t = re.sub(r"`[^`\n]*`", "", t)
        depth, in_item = 0, False
        for i, line in enumerate(t.split("\n"), 1):
            if not line.strip():
                in_item = False
            elif item.match(line) and not depth:
                in_item = True
            if depth and in_item and bullet.match(line):
                bad.append((_rel(book, q), i, line.strip()[:70]))
            n = len(re.findall(r"(?<!\\)\$", re.sub(r"\$\$", "", line)))
            depth = (depth + n) % 2
    print(f"\n[maths-span gate] {len(bad)} inline maths span(s) broken by a list bullet")
    for name, i, txt in bad:
        print(f"  \u2717 {name}:{i}  {txt}...")
        print("       inside a list item this line STARTS A BULLET LIST, closing the paragraph")
        print("       and the open $...$ with it -- the render dies on \\mathrm outside maths")
    if not bad:
        print("    no inline maths span is interrupted by a bullet inside a list item")
    return len(bad)


def _has_python_cell(book, chapter):
    """Whether the chapter file or one of its scheme files has an executable python cell."""
    files = [book / "chapters" / f"{chapter}.qmd"]
    files += sorted((book / "chapters" / chapter).rglob("*.qmd"))
    return any(re.search(r"^```+\s*\{python", f.read_text(), re.M) for f in files if f.exists())


def gate_freeze_is_tracked(book):
    """Every chapter quarto renders must have a TRACKED `_freeze/` cache.

    Added at the W32 close (`ISSUES` #46), after W31 shipped ch54 with its freeze left
    untracked -- so a fresh clone re-executed that chapter on its first render, silently,
    which is the one thing `BOOKCRAFT` §5's freeze exists to prevent.

    🔴 **The reason this needs a gate is the check that MISSES it, and it is the sharpest
    instance of W30's carry-forward the book has produced.** At the moment the defect was
    live, tracked freeze directories numbered **54** and chapters in `_quarto.yml` numbered
    **54**. That does not read like a coincidence -- it reads like the property being
    asserted. It was two errors cancelling: ch54 had no tracked freeze, and `ch01-template`
    has one while not being a chapter. ★ **An aggregate can be correct while its source rows
    are wrong; check rows, not sums.**

    ⛔ **The rule is deliberately ONE-DIRECTIONAL, and the bijection was considered and
    rejected.** A chapter with no tracked freeze is a real defect: it costs every fresh clone
    a re-execution. A freeze with no chapter -- which is what `ch01-template` is -- costs four
    small files and nothing else. Asserting the bijection would fail on a clean book today and
    force either an untracking or a named exception, and a gate that fires on a clean book is
    what `ISSUES` #43 was killed for. **So this gate asserts the implication that has teeth
    and stays silent about the clutter that does not.**

    ⚠️ It asks **git**, not the filesystem: the failure is a file existing on disk and not in
    the index, so a `Path.exists()` check is blind to it by construction. If git is
    unavailable the gate reports that and does not fail -- an unanswerable question is not a
    defect.
    """
    import subprocess
    qmd = [c for c in re.findall(r"chapters/(\S+)\.qmd", (book / "_quarto.yml").read_text())
           if _has_python_cell(book, c)]
    try:
        tracked = subprocess.run(
            ["git", "ls-files", str(book / "_freeze" / "chapters")],
            capture_output=True, text=True, timeout=30, cwd=book.parent)
    except Exception as exc:                                   # pragma: no cover
        print(f"\n[freeze gate] SKIPPED -- could not ask git ({exc})")
        return 0
    if tracked.returncode != 0:
        print(f"\n[freeze gate] SKIPPED -- git returned {tracked.returncode}")
        return 0
    have = {ln.split("/chapters/", 1)[1].split("/", 1)[0]
            for ln in tracked.stdout.splitlines() if "/chapters/" in ln}
    missing = [c for c in qmd if c not in have]
    print(f"\n[freeze gate] {len(qmd)} rendered chapters, "
          f"{len(missing)} with no tracked _freeze/ cache")
    for c in missing:
        print(f"  \u2717 {c}: renders, but its freeze is not in the index")
        print("       a fresh clone re-executes this chapter on its first build, silently")
    if not missing:
        print("    every chapter quarto renders has its computed results committed")
    return len(missing)


# --- pending: snapy_report functions a chapter calls that are not written yet -------------
#
# Lead's ruling (round 2): a missing `snapy_report` function is PENDING, not a failure, and
# nothing is released while anything is pending. So the gate has three outcomes: PASS (exit 0),
# PENDING (exit 2: nothing failed, but at least one called function or the book itself does not
# exist yet) and FAIL (exit 1). `--release` turns PENDING into a failure.
#
# The render must still run while a function is pending, or every other gate is blind. The cells
# that import a pending function get a placeholder: before the render, an IPython startup file
# (in a temporary IPYTHONDIR, so nothing outside the render is touched) puts a stand-in module or
# attribute into the kernel. Each stand-in returns a small figure that says PENDING, so the
# figure's label and caption still exist and its cross-references still resolve. Anything else
# a snapy_report module raises on import (a syntax error, a bad import inside it) is a FAIL.
#
# A placeholder that reaches `_freeze/` stays there after the function is written, because
# `freeze: auto` re-runs a chapter only when its .qmd changes. The placeholder figure's text
# repr carries PENDING_MARK, and the freeze gate fails on a frozen placeholder for a function
# that now exists.

PENDING_EXIT = 2
PENDING_MARK = "snapy_report PENDING:"
#: The placeholder's text carries this link (plus the function's name), which matplotlib writes into
#: the SVG and the PDF of the figure, so a placeholder can be found in `_freeze/`.
PENDING_URL = "snapy-report-pending:"

_PY_CELL = re.compile(r"^```+\s*\{python[^}]*\}\s*\n(.*?)^```+\s*$", re.S | re.M)
_FROM_IMPORT = re.compile(r"^[ \t]*from[ \t]+(snapy_report(?:\.\w+)*)[ \t]+import[ \t]+"
                          r"(\([^)]*\)|[^\n#]+)", re.M)
_PLAIN_IMPORT = re.compile(r"^[ \t]*import[ \t]+(snapy_report(?:\.\w+)*)", re.M)


def snapy_report_imports(book):
    """(file, line, module, name) for every snapy_report import in a python cell; name is None
    for `import snapy_report.x`."""
    found = []
    for q in _chapter_sources(book):
        text = q.read_text()
        for cell in _PY_CELL.finditer(text):
            body, start = cell.group(1), cell.start(1)
            for m in _FROM_IMPORT.finditer(body):
                line = text.count("\n", 0, start + m.start()) + 1
                names = m.group(2).strip().strip("()")
                for part in names.split(","):
                    name = part.split(" as ")[0].strip()
                    if name:
                        found.append((_rel(book, q), line, m.group(1), name))
            for m in _PLAIN_IMPORT.finditer(body):
                line = text.count("\n", 0, start + m.start()) + 1
                found.append((_rel(book, q), line, m.group(1), None))
    return found


# The editor's ruling (round 2): a chapter-opening dependency map, `make_fig("<chapter or scheme>")` from
# snapy_report.depmap, whose target has no entry in the scheme registry yet is PENDING too. That is
# known only by asking depmap: `depmap.status(target)` returns ("ok" | "pending", key, detail), and
# for a pending target depmap's own make_fig returns a placeholder whose repr is
# "<snapy_report PENDING: <key>>", key = snapy_report.depmap.<target> ("-" as "_"). The key goes
# into the same pending table, so the verdict and the frozen-placeholder gate treat it like a
# missing function.
_DEPMAP_IMPORT = re.compile(r"^[ \t]*from[ \t]+snapy_report\.depmap[ \t]+import[ \t]+[^\n#]*\bmake_fig\b", re.M)
_DEPMAP_CALL = re.compile(r"\bmake_fig\(\s*['\"]([\w-]+)['\"]")


def depmap_targets(book):
    """(file, line, target) for every `make_fig("<target>")` in a python cell that imports
    snapy_report.depmap's make_fig."""
    found = []
    for q in _chapter_sources(book):
        text = q.read_text()
        for cell in _PY_CELL.finditer(text):
            body, start = cell.group(1), cell.start(1)
            if not _DEPMAP_IMPORT.search(body):
                continue
            for m in _DEPMAP_CALL.finditer(body):
                found.append((_rel(book, q), text.count("\n", 0, start + m.start()) + 1, m.group(1)))
    return found


# One JSON line per target: [target, state, key, detail]. No output when depmap or its status()
# cannot be imported: the import probe above already reports that.
_DEPMAP_PROBE = r"""
import json, sys
try:
    from snapy_report.depmap import status
except Exception:
    sys.exit(0)
for target in json.loads(sys.argv[1]):
    try:
        st = status(target)
        print(json.dumps([target, st.state, st.key, st.detail]))
    except Exception as e:
        print(json.dumps([target, "error", "", f"{type(e).__name__}: {e}"]))
"""


def gate_depmap_targets(book, pending, errors):
    """Add the dependency-map targets with no registry entries to pending, and a status() that
    raises to errors."""
    import json
    targets = depmap_targets(book)
    if not targets:
        return
    names = sorted({t for _, _, t in targets})
    p = subprocess.run([_cell_python(), "-c", _DEPMAP_PROBE, json.dumps(names)],
                       capture_output=True, text=True)
    state = {t: (st, key, detail) for t, st, key, detail in map(json.loads, p.stdout.splitlines())}
    if p.returncode != 0:
        errors.append(f"depmap probe: {p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode}")
    n_pending = 0
    for f, n, target in targets:
        if target not in state:
            continue
        st, key, detail = state[target]
        if st == "pending":
            n_pending += key not in pending
            pending.setdefault(key, ("snapy_report.depmap", "make_fig", []))[2].append(f"{f}:{n}")
        elif st == "error":
            errors.append(f"{f}:{n}: depmap {target}: {detail}")
    print(f"[snapy_report gate] {len(names)} dependency-map target(s): {n_pending} pending"
          " (no scheme in the registry yet)")


# Run in the python the cells run in. One JSON line per import: [module, name, state, detail],
# state "ok", "pending" (the module or the name does not exist) or "error" (it raises).
_PROBE = r"""
import importlib, json, sys, traceback
for module, name in json.loads(sys.argv[1]):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        ours = e.name is not None and (e.name == module or module.startswith(e.name + "."))
        state = "pending" if ours and e.name.split(".")[0] == "snapy_report" else "error"
        print(json.dumps([module, name, state, f"{type(e).__name__}: {e}"])); continue
    except Exception as e:
        print(json.dumps([module, name, "error", traceback.format_exc(limit=-1).strip()])); continue
    if name is None or hasattr(mod, name):
        print(json.dumps([module, name, "ok", ""])); continue
    try:
        importlib.import_module(module + "." + name)
        print(json.dumps([module, name, "ok", ""]))
    except ModuleNotFoundError:
        print(json.dumps([module, name, "pending", f"{module} has no {name}"]))
    except Exception as e:
        print(json.dumps([module, name, "error", traceback.format_exc(limit=-1).strip()]))
"""


def _cell_python():
    """The interpreter the cells run in: QUARTO_PYTHON if set, as quarto uses it, else this one."""
    import os
    return os.environ.get("QUARTO_PYTHON") or sys.executable


def gate_snapy_report(book):
    """Sort every snapy_report import of the chapters into ok, pending and error.

    Returns (pending, errors): pending is {"module.name": (module, name, [file:line, ...])}, with
    name None for `import module`; errors is a list of lines to print. A pending import is not a
    problem; an error is.
    """
    import json
    imports = snapy_report_imports(book)
    if not imports:
        print("\n[snapy_report gate] no chapter imports snapy_report")
        return {}, []
    pairs = sorted({(m, n) for _, _, m, n in imports}, key=lambda t: (t[0], t[1] or ""))
    p = subprocess.run([_cell_python(), "-c", _PROBE, json.dumps(pairs)],
                       capture_output=True, text=True)
    state = {}
    for line in p.stdout.splitlines():
        module, name, st, detail = json.loads(line)
        state[(module, name)] = (st, detail)
    pending, errors = {}, []
    for f, n, module, name in imports:
        st, detail = state.get((module, name), ("error", p.stderr.strip()[-300:] or "probe failed"))
        what = module + ("." + name if name else "")
        if st == "pending":
            pending.setdefault(what, (module, name, []))[2].append(f"{f}:{n}")
        elif st == "error":
            errors.append(f"{f}:{n}: {what}: {detail.splitlines()[-1] if detail else ''}")
    ok = len(pairs) - len(pending) - len({e.split(": ")[1] for e in errors})
    print(f"\n[snapy_report gate] {len(pairs)} imported function(s): {ok} exist, "
          f"{len(pending)} pending, {len(errors)} raising")
    gate_depmap_targets(book, pending, errors)
    for what, (_, _, where) in sorted(pending.items()):
        print(f"  … PENDING {what}  ({', '.join(where)})")
    for e in errors:
        print(f"  ✗ {e}")
    return pending, errors


# The module the render's IPython startup file imports: stand-ins for the pending functions.
_STARTUP = r'''
import sys, types, importlib

PENDING = %(pending)r      # (module, name), name None for `import module`
MARK = %(mark)r
URL = %(url)r


def _placeholder(what):
    def make(*args, **kwargs):
        # matplotlib is imported in the cell, after quarto's kernel setup has chosen its formats.
        from matplotlib.figure import Figure

        class _PendingFigure(Figure):
            def __repr__(self):
                return f"<{MARK} {self._pending_what}>"

        fig = _PendingFigure(figsize=(3.4, 0.9))
        fig._pending_what = what
        # on an axes: IPython's figure printer skips a figure that has none, leaving only the repr
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "PENDING: " + what, ha="center", va="center", fontsize=8,
                url=URL + what)
        return fig
    make.__name__ = what.rsplit(".", 1)[-1]
    return make


def _stand_in(module):
    def getattr_(attr):
        if attr.startswith("_"):     # inspect, pickle, ... look for dunders
            raise AttributeError(attr)
        return _placeholder(module + "." + attr)
    return getattr_


def _module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        parent, _, leaf = name.rpartition(".")
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__getattr__ = _stand_in(name)
        sys.modules[name] = mod
        if parent:
            setattr(_module(parent), leaf, mod)
        return mod


for _mod, _name in PENDING:
    _m = _module(_mod)
    if _name is not None and not hasattr(_m, _name):
        setattr(_m, _name, _placeholder(_mod + "." + _name))
'''


def pending_render_env(pending):
    """Environment for a render whose cells call pending functions, or None if there are none."""
    import os
    import tempfile
    if not pending:
        return None
    ipdir = pathlib.Path(tempfile.mkdtemp(prefix="render-gate-ipython-"))
    startup = ipdir / "profile_default" / "startup"
    startup.mkdir(parents=True)
    # The stand-ins live in a module of their own: names an IPython startup file defines in the
    # user namespace do not survive into the cells.
    (ipdir / "_snapy_report_pending.py").write_text(
        _STARTUP % {"pending": sorted(v[:2] for v in pending.values()), "mark": PENDING_MARK,
                    "url": PENDING_URL})
    (startup / "00-snapy-report-pending.py").write_text(
        f"import sys\nsys.path.insert(0, {str(ipdir)!r})\nimport _snapy_report_pending\n"
        "sys.path.pop(0)\n")
    env = dict(os.environ, IPYTHONDIR=str(ipdir))
    print(f"\n[snapy_report gate] rendering with {len(pending)} placeholder(s)")
    return env


def gate_frozen_placeholders(book, pending):
    """A placeholder in `_freeze/` for a function that now exists: the chapter must be re-run.

    The placeholder is found by its link (PENDING_URL) in the frozen figures (SVG, PDF) or by its
    text repr (PENDING_MARK) in the frozen markdown.
    """
    stale = set()
    pat = re.compile(rb"(?:" + re.escape(PENDING_URL.encode()) + rb"|"
                     + re.escape(PENDING_MARK.encode()) + rb" )([\w.]+)")
    files = sorted(f for f in (book / "_freeze").rglob("*") if f.is_file()) \
        if (book / "_freeze").is_dir() else []
    for f in files:
        for what in set(m.decode() for m in pat.findall(f.read_bytes())):
            if what not in pending:
                stale.add((str(f.parent.relative_to(book)), what))
    print(f"\n[frozen-placeholder gate] {len(stale)} frozen placeholder(s) for functions that exist")
    for f, what in sorted(stale):
        print(f"  ✗ {f}: {what} -- render that chapter alone to replace it (STYLE 10.9)")
    return len(stale)


def gate_chapter_list(book):
    """The chapter list of `_quarto.yml` is the files that exist, in OUTLINE order (tools/book_chapters.py).

    Checked in the report's layout (OUTLINE.md next to book/) or wherever the list markers are; a test book
    elsewhere with a hand-written list is left alone.
    """
    import book_chapters
    has_markers = book_chapters.current_block((book / "_quarto.yml").read_text()) is not None
    if not has_markers and not (book.parent / "OUTLINE.md").exists():
        print("\n[chapter-list gate] not the report layout and no list markers: skipped")
        return 0
    status, msgs = book_chapters.check(book)
    print(f"\n[chapter-list gate] " + ("the list in _quarto.yml is the chapter files that exist"
                                       if status == 0 else f"{len(msgs)} problem(s)"))
    for m in msgs:
        print(f"  ✗ {m}" if not m.startswith("fix:") else f"    {m}")
    return 0 if status == 0 else max(1, len([m for m in msgs if not m.startswith("fix:")]))


def gate_task_map(book):
    """N2: every question of the bank has one task-map record, every destination is a label of the book, and the
    committed Appendix G is a fresh generation (tools/gen_task_map.py). Checked in the report's layout, where
    tools/task-map.yaml is."""
    data = book.parent / "tools" / "task-map.yaml"
    if not data.exists():
        print("\n[task-map gate] no tools/task-map.yaml next to the book: skipped")
        return 0
    import gen_task_map
    try:
        records = gen_task_map.load(data.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"\n[task-map gate] {data.name} does not parse: {exc}")
        return 1
    bad = gen_task_map.problems(records, gen_task_map.labels(book))
    frag = book / "chapters" / "17-appendices" / gen_task_map.FRAGMENT.name
    if not bad and (not frag.exists() or frag.read_text(encoding="utf-8")
                    != gen_task_map.render(records, gen_task_map.labels(book))):
        bad.append(f"{frag.name} is stale; run python3 tools/gen_task_map.py")
    c = gen_task_map.counts(records)
    print("\n[task-map gate] " + (f"{sum(c.values())} questions, every one with a record and a destination or a reason"
                                    if not bad else f"{len(bad)} problem(s)"))
    for m in bad:
        print(f"  ✗ {m}")
    return len(bad)


def gate_provenance(book):
    """N7: every provenance tag names a full snapy sha at or after the code pin that resolves, and the command; a
    quoted timing adds its scope and repetitions. Per-chapter counts of failing tags equal tools/provenance-baseline.json,
    which must exist and may not rise against the target branch (tools/check_provenance.py, STYLE 4). Checked in the
    report's layout, where code_pin.yml is."""
    report = book.parent
    if not (report / "code_pin.yml").exists() or not (report / "tools").is_dir():
        print("\n[provenance gate] no code_pin.yml and tools/ next to the book: skipped")
        return 0
    import check_provenance
    ref = check_provenance.base_ref(report)
    status, msgs, found = check_provenance.check(report, resolve=None, ref=ref)
    print("\n[provenance gate] " + (f"every chapter at its baseline, not raised against {ref} "
                                     f"({len(found)} tag problem(s) in all)" if status == 0
                                     else f"{len(msgs)} problem(s) with the baseline"))
    for m in msgs:
        print(f"  ✗ {m}")
    return len(msgs)


def gate_figstyle(book):
    """Figure code styles only through snapy_report.figstyle; per-chapter counts may only shrink
    (tools/check_figstyle.py, STYLE 6). Checked in the report's layout, where the baseline file is."""
    report = book.parent
    baseline = report / "tools" / "figstyle-baseline.json"
    if not baseline.exists():
        print("\n[figstyle gate] no tools/figstyle-baseline.json next to the book: skipped")
        return 0
    import check_figstyle
    status, msgs = check_figstyle.check(report, baseline)
    print("\n[figstyle gate] " + ("every chapter at its baseline of styling outside figstyle"
                                  if status == 0 else f"{len(msgs)} chapter(s) differ from the baseline"))
    for m in msgs:
        print(f"  ✗ {m}")
    return len(msgs)


#: The two report-mode checks below list every hit by file:line but do not fail the gate while this is True.
#: Set it to False to make them fail (the editor calls the switch once the chapters' cross-ref commits are in).
REPORT_ONLY = True

_HAND_CHAPTER = re.compile(r"\b[Cc]hapter [0-9]+")


def _report(name, hits, where=()):
    """Print the hits of a report-mode check (and, uncounted, where they come from); return the number
    that counts as failures."""
    mode = "REPORT mode, not failing" if REPORT_ONLY else "failing"
    print(f"\n[{name} gate] {len(hits)} hit(s)" + (f" ({mode})" if hits else ""))
    for h in hits:
        print(f"  {'·' if REPORT_ONLY else '✗'} {h}")
    for w in where:
        print(f"      at {w}")
    return 0 if REPORT_ONLY else len(hits)


def gate_fig_output(book):
    """The drawn figures against STYLE 6.2 (house width, nothing wider than the text, no text under 8 pt, at most
    four panels, no overlapping texts, an identical redraw, no warning); per-chapter counts may only shrink
    (tools/check_fig_output.py). Needs the figure code importable, as the render does."""
    report = book.parent
    baseline = report / "tools" / "figoutput-baseline.json"
    if not baseline.exists():
        print("\n[figure-output gate] no tools/figoutput-baseline.json next to the book: skipped")
        return 0
    import check_fig_output
    status, msgs = check_fig_output.check(report, baseline)
    print("\n[figure-output gate] " + ("every chapter at its baseline of drawn-figure findings"
                                       if status == 0 else f"{len(msgs)} count(s) differ from the baseline"))
    for m in msgs:
        print(f"  ✗ {m}")
    return len(msgs)


def gate_figure_manifest(book):
    """FIGURES.md, the figure manifest, lists every figure cell with its section file and the figure modules its code
    imports, and every path it lists exists (tools/render_figures.py; static, nothing is drawn)."""
    report = book.parent
    if not (report / "tools" / "render_figures.py").exists():
        print("\n[figure-manifest gate] no tools/render_figures.py next to the book: skipped")
        return 0
    import render_figures
    msgs = render_figures.check_manifest(report)
    print("\n[figure-manifest gate] " + ("every figure cell listed in FIGURES.md, every listed path present"
                                         if not msgs else f"{len(msgs) - 1} problem(s)"))
    for m in msgs:
        print(f"  ✗ {m}")
    return len(msgs)


def gate_tables(book):
    """Pipe tables against STYLE 10.5 and the Code/Tests headers of STYLE 2 (at most 5 columns, a labelled caption,
    the fixed headers, one table per Code or Tests layer); per-chapter counts may only shrink
    (tools/check_tables.py)."""
    report = book.parent
    baseline = report / "tools" / "table-baseline.json"
    if not baseline.exists():
        print("\n[table gate] no tools/table-baseline.json next to the book: skipped")
        return 0
    import check_tables
    status, msgs = check_tables.check(report, baseline)
    print("\n[table gate] " + ("every chapter at its baseline of table findings"
                               if status == 0 else f"{len(msgs)} count(s) differ from the baseline"))
    for m in msgs:
        print(f"  ✗ {m}")
    return len(msgs)


def gate_do_cards(book):
    """The runnable cards against tools/do-cards.yaml (STYLE "Reader standards", N8; tools/check_do_cards.py): the same
    card set in the sources and the data file, a deck for every runnable card, every deck resolving in the repository,
    and no stale named exception."""
    report = book.parent
    data = report / "tools" / "do-cards.yaml"
    if not data.exists():
        print("\n[do-cards gate] no tools/do-cards.yaml next to the book: skipped")
        return 0
    import check_do_cards
    status, msgs = check_do_cards.check(report, data)
    print("\n[do-cards gate] " + ("every card has a record, a deck that resolves, and no stale exception"
                                  if status == 0 else f"{len(msgs)} problem(s)"))
    for m in msgs:
        print(f"  ✗ {m}")
    return len(msgs)


def gate_hand_typed_chapters(book):
    """A chapter named by a hand-typed number ("Chapter 7", "chapter 12") instead of a cross-ref
    (@sec-chNN) or, for a chapter not written yet, its name in words (STYLE 10.1)."""
    hits = []
    for q in _chapter_sources(book):
        for n, line in enumerate(q.read_text().splitlines(), 1):
            for m in _HAND_CHAPTER.finditer(line):
                hits.append(f"{_rel(book, q)}:{n}: {m.group(0)}")
    return _report("hand-typed chapter", hits)


def gate_unresolved_in_output(book):
    """Any reference quarto could not resolve, in the rendered output, whatever the gates above saw: '?@...'
    on a page of the PDF (pdftotext), and the quarto-unresolved-ref spans of the HTML, by file and line."""
    import shutil
    out = book / "_book"
    hits = []
    for pdf in sorted(out.glob("*.pdf")) if out.is_dir() and shutil.which("pdftotext") else []:
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
        for page, body in enumerate(text.split("\f"), 1):
            for m in re.finditer(r"\?@[\w-]*\w", body):
                hits.append(f"{pdf.name} p{page}: {m.group(0)}")
    html = []
    for f in sorted(out.rglob("*.html")) if out.is_dir() else []:
        for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            for m in re.finditer(r'quarto-unresolved-ref">\??@?([\w-]+)<|\?@([\w-]*\w)', line):
                html.append(f"{f.relative_to(book)}:{n}: ?@{m.group(1) or m.group(2)}")
    # the PDF count is the measure when there is a PDF; the HTML lines say where to look
    return _report("unresolved-in-output", hits or html, html if hits else ())


def _option(argv, name, default):
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def gate_outputs(book):
    """Both formats of STYLE 10.9 were rendered into `_book/`."""
    out = book / "_book"
    missing = [kind for kind, pat in (("HTML", "*.html"), ("PDF", "*.pdf"))
               if not list(out.rglob(pat))]
    if not missing:
        print(f"\n[output gate] {out} holds the HTML and the PDF")
        return 0
    print(f"\n[output gate] {out} has no {' and no '.join(missing)}: list html and pdf"
          " under format: in _quarto.yml and render them in one command")
    return len(missing)


#: Gates whose failure is expected for now (editor's ruling); still failures, listed apart.
EXPECTED_FAIL = {"unwritten-chapter refs": "expected until those chapters are written",
                 "unwritten-scheme refs": "expected until those schemes are written"}

#: Which failures are reported rather than failed. CI judges the gate by this script's exit code,
#: so this file is the one place the rule lives; before it, the workflow carried a second copy and a
#: fresh clone exited 1 on gates CI was green on.
POLICY_PATH = pathlib.Path(__file__).resolve().parent / "render-gate-policy.json"


def load_policy(path=POLICY_PATH):
    """The report-only lists, as sets. A missing or unreadable file tolerates nothing."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"[policy] {path} unreadable ({exc}); tolerating nothing")
        return frozenset(), frozenset()
    return (frozenset(raw.get("report_only_gates", ())),
            frozenset(raw.get("report_only_log_items", ())))


def apply_policy(status, failed, log_items, policy):
    """Map the gate's own verdict through the report-only policy.

    `failed` is what the verdict printed on its `fail:` line -- the real failures, with the
    expected ones already set aside. Returns (exit status, the names tolerated).

    A run whose only failures are the expected ones still fails: `failed` is then empty while the
    status is 1, and a ruling to revisit is not a tolerated state.

    Only failures are treated. PENDING keeps its own exit status, which this repository has always
    given it and which `test_no_book_is_pending` pins: it is not a failure, `--release` turns it
    into one, and the one shape it takes that CI would call green -- a rendered book with a
    placeholder in it -- is a state to notice locally rather than to report as a pass.
    """
    if status != 1:
        return status, frozenset()
    gates, items = policy
    tolerated = set(failed) & set(gates)
    if "log" in failed and log_items and set(log_items) <= set(items):
        tolerated.add("log")
    remaining = set(failed) - tolerated
    if remaining or not failed:
        return 1, frozenset(tolerated)
    return 0, frozenset(tolerated)


def _verdict(results, release, expected_targets=None, log_items=(), policy=None):
    """Print the pass/pending/fail tally of the gates and return the exit status."""
    tally = {k: sorted(n for n, v in results.items() if v == k) for k in ("pass", "pending", "fail")}
    print(f"\n{len(tally['pass'])} gate(s) passed, {len(tally['pending'])} pending, "
          f"{len(tally['fail'])} failed")
    real = [n for n in tally["fail"] if n not in EXPECTED_FAIL]
    if tally["pending"]:
        print(f"  pending: {', '.join(tally['pending'])}")
    if real:
        print(f"  fail: {', '.join(real)}")
    for n in tally["fail"]:
        if n in EXPECTED_FAIL:
            targets = (expected_targets or {}).get(n, ())
            print(f"  fail, {EXPECTED_FAIL[n]}: {n}"
                  + (f": {' '.join('@' + t for t in sorted(targets))}" if targets else ""))
    if tally["fail"]:
        print("GATE FAILED" + ("" if real else ": only the expected failures above"))
        status = 1
    elif tally["pending"]:
        print("GATE PENDING: nothing failed, but nothing can be released while anything is pending"
              + (" (--release: this counts as a failure)" if release else ""))
        status = 1 if release else PENDING_EXIT
    else:
        print("GATE PASSED")
        status = 0
    if policy is None:
        return status
    treated, tolerated = apply_policy(status, real, log_items, policy)
    if tolerated:
        print("  reported, not failed, per tools/render-gate-policy.json: "
              + ", ".join(sorted(tolerated))
              + (f" (log items: {', '.join(sorted(log_items))})" if "log" in tolerated else ""))
        print("GATE PASSED under the policy" if treated == 0 else "GATE FAILED despite the policy")
    return treated


def main(argv):
    argv = list(argv)
    log_opt = _option(argv, "--log", None)
    overfull_pt = float(_option(argv, "--overfull-pt", OVERFULL_PT))
    book = pathlib.Path(argv[0] if argv and not argv[0].startswith("--") else "book")
    do_render = "--no-render" not in argv
    release = "--release" in argv
    # The policy is on by default so that the documented command and CI judge a run alike;
    # --strict shows the untreated verdict.
    policy = None if "--strict" in argv else load_policy()
    results = {}

    def record(name, problems):
        results[name] = "fail" if problems else "pass"
        return problems

    if not (book / "_quarto.yml").exists():
        print(f"[book gate] {book / '_quarto.yml'} does not exist: there is no book to render yet"
              " (STYLE 8; the editor adds it)")
        results["book"] = "pending"
        return _verdict(results, release, policy=policy)

    source_gates = (("labels", gate_cross_chapter_labels), ("dangling refs", gate_dangling_refs),
                    ("float labels", gate_float_label_prefixes),
                    ("font commands", gate_old_font_commands),
                    ("caption blocks", gate_orphan_caption_blocks),
                    ("quotes", gate_unbalanced_quotes),
                    ("maths spans", gate_maths_span_broken_by_a_bullet),
                    ("freeze tracked", gate_freeze_is_tracked),
                    ("chapter list", gate_chapter_list),
                    ("figstyle", gate_figstyle),
                    ("figure manifest", gate_figure_manifest),
                    ("provenance", gate_provenance),
                    ("tables", gate_tables),
                    ("do cards", gate_do_cards),
                    ("hand-typed chapter", gate_hand_typed_chapters))
    for name, gate in source_gates:
        record(name, gate(book))
    # Reader standards share the normal entry point, including --labels-only.
    import gen_topic_index
    import check_reader_hygiene
    import check_limitations
    # N9 is advisory: do not record warnings as a failed (or passing) gate.
    check_limitations.main(["--book", str(book)])
    import gen_reader_reference
    import gen_boundary_contract
    import gen_forcing_contract
    import gen_scheme_contract
    import gen_scheme_native
    import glossary_registry
    import gen_glossary_appendix
    import gen_species_contract
    report = book.resolve().parent
    for name, check in (
        ("N1 navigation", lambda: gen_topic_index.main([
            "--book", str(book), "--data", str(report / "tools/topic-index.yaml"),
            "--fragment", str(book / "chapters/17-appendices/_appf-topic-index.qmd"), "--check"])),
        ("N3 reader hygiene", lambda: check_reader_hygiene.main(["--book", str(book)])),
        ("N4 glossary", lambda: gen_reader_reference.check_glossary(report)),
        ("N4 concept registry", lambda: glossary_registry.check(report)),
        ("N4 glossary appendix", lambda: gen_glossary_appendix.check(report)),
        ("N5 pinned defaults", lambda: gen_reader_reference.check_defaults(report)),
        ("N6 boundary contract", lambda: gen_boundary_contract.check(report)),
        ("N6 forcing contract", lambda: gen_forcing_contract.check(report)),
        ("N6 scheme contract", lambda: gen_scheme_contract.check(report)),
        ("N6 scheme native", lambda: gen_scheme_native.check(report)),
        ("N6 species contract", lambda: gen_species_contract.check(report)),
    ):
        try:
            record(name, check())
        except (OSError, ValueError, SystemExit) as exc:
            print(f"[{name}] cannot validate inputs: {exc}")
            record(name, 1)

    expected = {"unwritten-chapter refs": sorted(unwritten_chapter_refs(book)),
                "unwritten-scheme refs": sorted(unwritten_scheme_refs(book))}
    record("unwritten-chapter refs", gate_unwritten_chapter_refs(book))
    record("unwritten-scheme refs", gate_unwritten_scheme_refs(book))
    record("task map", gate_task_map(book))
    import gen_reader_tasks
    record("N2 reader tasks", gen_reader_tasks.check(book.resolve().parent, book))
    pending, errors = gate_snapy_report(book)
    results["snapy_report"] = "fail" if errors else ("pending" if pending else "pass")
    record("frozen placeholders", gate_frozen_placeholders(book, pending))

    if "--labels-only" in argv:
        return _verdict(results, release, expected, policy=policy)
    # runs the figure code, so not a source gate: in every full run, before the render
    record("figure output", gate_fig_output(book))

    env = pending_render_env(pending) if do_render else None
    rc, out = (0, "") if not do_render else render(book, env)
    if env:
        import shutil
        shutil.rmtree(env["IPYTHONDIR"], ignore_errors=True)
    if rc != 0:
        print(f"\nRENDER FAILED (exit {rc}). Last lines:\n")
        print("\n".join(out.splitlines()[-15:]))
        results["render"] = "fail"
        return _verdict(results, release, expected, policy=policy)

    # G0: the source preflight is not rendered-book evidence.
    for pdf in sorted((book / "_book").glob("*.pdf")):
        check_limitations.main(["--pdf", str(pdf)])

    record("outputs", gate_outputs(book))
    record("unresolved-in-output", gate_unresolved_in_output(book))
    import check_rendered_glossary
    record("N1/N4 rendered glossary", check_rendered_glossary.check(book))

    hits = gate_stdout(out, {t for ts in expected.values() for t in ts})
    print(f"\n[stdout gate] {len(hits)} matching lines"
          + ("" if hits else "  (this is the check that has always run)"))
    for h in hits[:10]:
        # without colour codes and quarto's filter path, so the message itself fits on the line
        h = re.sub(r"\x1b\[[0-9;]*m", "", h)
        h = re.sub(r"^(\s*WARNING) \([^)]*\)", r"\1", h)
        print("   ", h.strip()[:110])
    record("stdout", len(hits))
    record("N2 rendered task map", gen_reader_tasks.check_rendered(book, book.resolve().parent))

    log = pathlib.Path(log_opt) if log_opt else find_log(book)
    text, fatal, noise = gate_log(log, book, overfull_pt)
    if text is None:
        print(f"\n[log gate] {log} NOT FOUND -- the gate is BLIND.")
        print("   quarto deleted it. Re-run without --no-render so that")
        print("   -M latex-clean:false keeps the log alive.")
        results["log"] = "fail"
        return _verdict(results, release, expected, policy=policy)

    print(f"\n[log gate] {log} ({len(text):,} bytes)")
    if not fatal:
        print("    no dropped glyphs, no bad cross-references")
    for name, (found, why) in fatal.items():
        uniq = sorted(set(map(str, found)))
        print(f"  ✗ {name}: {len(found)} ({why})")
        for u in uniq[:8]:
            print(f"       {u}")
    record("log", sum(len(found) for found, _ in fatal.values()))

    record("table cells", _gate_table_cells(book))

    print("    typography (not fatal): "
          + ", ".join(f"{n} {c}" for n, c in noise.items() if c))

    # `fatal` keys are the log gate's own item names; the policy tolerates the log gate only when
    # every one of them is listed there.
    return _verdict(results, release, expected, log_items=frozenset(fatal), policy=policy)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
