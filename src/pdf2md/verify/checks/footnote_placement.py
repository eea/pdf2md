"""Check: footnote references in the source survive as resolved footnotes in the .qmd.

Counts superscript reference marks in the source PDF (¹²³… or the ^1/^2 caret form)
and compares them against Pandoc-style footnote markers [^n] and their [^n]: definitions
in the .qmd. Warns on a source/output mismatch and on dangling markers (a [^n] with no
matching definition, or a definition nothing points at)."""

import re
from functools import lru_cache

from .. import CheckResult, Finding, register
from ..textutil import pdf_lines

# unicode superscript digits (⁰ U+2070, ¹ U+00B9, ² U+00B2, ³ U+00B3, ⁴–⁹ U+2074–2079)
_SUPER_DIGITS = set("⁰¹²³⁴⁵⁶⁷⁸⁹")
# superscript signs/letters (⁺⁻ⁿⁱ…) and subscript chars (₀-₉, ₊, ₐ…) that ride
# alongside digits in chemical formulae (²²⁷₉₀Th⁺) and math — never lone footnotes
_SUPER_OTHER = set("⁺⁻⁼⁽⁾ⁿⁱᵃᵇᶜᵈᵉ")
_SUB_CHARS = set("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓₕₖₗₘₙₚₛₜ")
_SUPERSUB = _SUPER_DIGITS | _SUPER_OTHER | _SUB_CHARS
_CARET_REF = re.compile(r"\^\d+")

# lines above this index are the title page / author-affiliation block, where
# superscripts mark affiliations (¹²) rather than footnotes
_TITLE_BLOCK_LINES = 15


def _math_heavy(line: str) -> bool:
    """A line dominated by non-ASCII characters (math operators, Greek,
    sub/superscript runs) — an equation or chemical formula, not prose that
    happens to carry a footnote mark."""
    dense = [c for c in line if not c.isspace()]
    if not dense:
        return True
    heavy = sum(1 for c in dense if not c.isascii())
    return heavy >= 3 and heavy / len(dense) >= 0.30


def _line_footnote_marks(line: str) -> int:
    """Count lone superscript digits in a line. A superscript that is part of a
    contiguous super/subscript run (affiliation ¹², chemical ²²⁷₉₀) is skipped;
    only an isolated superscript digit reads as a real footnote reference."""
    n, i, count = len(line), 0, 0
    while i < n:
        if line[i] in _SUPERSUB:
            j = i
            while j < n and line[j] in _SUPERSUB:
                j += 1
            run = line[i:j]
            if len(run) == 1 and run in _SUPER_DIGITS:
                count += 1
            i = j
        else:
            i += 1
    return count

# a footnote reference [^id] NOT immediately followed by ':' (which would be a definition)
_QMD_REF = re.compile(r"\[\^([^\]\s]+)\](?!:)")
# a footnote definition [^id]: at the start of a line
_QMD_DEF = re.compile(r"^\[\^([^\]\s]+)\]:", re.MULTILINE)


@lru_cache(maxsize=8)
def _source_footnote_count(pdf_str: str, mtime: float) -> tuple:
    """Footnote reference marks in the source PDF: (count, pages carrying them).
    Filters out the affiliation superscripts of the title block, chemical/math
    superscript runs, and equation-heavy lines that are not prose footnotes."""
    total, pages = 0, []
    for idx, (pno, line) in enumerate(pdf_lines(pdf_str)):
        if idx < _TITLE_BLOCK_LINES:
            continue                       # title page / author-affiliation block
        if _math_heavy(line):
            continue                       # equations, chemical formulae
        n = _line_footnote_marks(line) + len(_CARET_REF.findall(line))
        if n:
            total += n
            if pno + 1 not in pages:
                pages.append(pno + 1)
    return total, tuple(pages)


def _try_source_count(ctx) -> tuple:
    if not (ctx.original_pdf and ctx.original_pdf.exists()):
        return 0, ()
    try:
        return _source_footnote_count(str(ctx.original_pdf), ctx.original_pdf.stat().st_mtime)
    except Exception:
        return 0, ()


@register
class FootnotePlacementCheck:
    name = "footnote_placement"

    def applicable(self, ctx) -> bool:
        if not ctx.qmd_text:
            return False
        return bool(_QMD_REF.search(ctx.qmd_text)) or _try_source_count(ctx)[0] > 0

    def run(self, ctx) -> CheckResult:
        ref_ids = _QMD_REF.findall(ctx.qmd_text)
        def_ids = set(_QMD_DEF.findall(ctx.qmd_text))
        ref_set = set(ref_ids)

        resolved = sorted(ref_set & def_ids)
        dangling = sorted(ref_set - def_ids)        # markers with no definition
        orphaned = sorted(def_ids - ref_set)        # definitions nothing points at
        src_count, src_pages = _try_source_count(ctx)

        findings = []
        for fid in dangling:
            findings.append(Finding(
                f"footnote marker [^{fid}] has no matching [^{fid}]: definition", "warn"))
        for fid in orphaned:
            findings.append(Finding(
                f"footnote definition [^{fid}]: is never referenced in the body", "warn"))

        # source vs output mismatch: PDF has superscript marks the .qmd didn't turn
        # into footnotes (or vice-versa)
        if src_count and abs(src_count - len(ref_ids)) > max(1, src_count // 5):
            findings.append(Finding(
                f"source has ~{src_count} superscript reference mark(s) but the .qmd "
                f"has {len(ref_ids)} footnote marker(s)", "warn", "footnotes"))

        status = "warn" if findings else "ok"
        if not ref_set and findings:
            # "0/0 resolved" reads as fine when it means the opposite: the source has
            # footnote marks and none survived
            summary = (f"source has ~{src_count} footnote mark(s); none survived "
                       f"into the .qmd")
        else:
            summary = (f"{len(resolved)}/{len(ref_set)} footnote reference(s) resolved"
                       + (f", {len(dangling)} dangling" if dangling else "")
                       + (f", {len(orphaned)} orphaned definition(s)" if orphaned else ""))
        return CheckResult(
            self.name, status, summary,
            metric=f"{len(resolved)}/{len(ref_set)} resolved" if ref_set else None,
            findings=findings,
            detail={"src_count": src_count, "src_pages": list(src_pages)},
        )
