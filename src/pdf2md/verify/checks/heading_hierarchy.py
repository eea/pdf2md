"""Check: the source document's heading outline survives, in order, in the .qmd.

Uses the source PDF's bookmark outline (PyMuPDF get_toc()) as the reference heading
sequence and compares it against the ATX headings (#, ##, ###…) in the .qmd. Warns when
the heading count differs significantly, when outline entries are missing from the .qmd,
or when the sections that do survive appear out of their source order."""

import bisect
import difflib
import re
from functools import lru_cache

from .. import CheckResult, Finding, register
from ..textutil import normalize

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

# ATX headings; excludes fenced-code and YAML lines by stripping frontmatter first
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
# a heading count within this fraction of the source is "close enough"
_COUNT_TOLERANCE = 0.25
_MAX_LISTED = 30


@lru_cache(maxsize=8)
def _source_toc_titles(pdf_str: str, mtime: float) -> tuple:
    """Normalized outline heading titles from the source PDF, in document order."""
    doc = fitz.open(pdf_str)
    try:
        toc = doc.get_toc(simple=True)  # [[level, title, page], …]
    finally:
        doc.close()
    return tuple(normalize(t[1]) for t in toc if normalize(t[1]))


def _try_source_titles(ctx) -> tuple:
    if not (_FITZ_AVAILABLE and ctx.original_pdf and ctx.original_pdf.exists()):
        return ()
    try:
        return _source_toc_titles(str(ctx.original_pdf), ctx.original_pdf.stat().st_mtime)
    except Exception:
        return ()


def _qmd_headings(qmd_text: str) -> list:
    body = _FRONTMATTER_RE.sub("", qmd_text, count=1)
    return [normalize(m.group(2)) for m in _HEADING_RE.finditer(body) if normalize(m.group(2))]


def _fuzzy(text: str) -> str:
    """A normalized heading key with section numbering removed, so the source
    outline ("1 structural and layout challenges") and the .qmd ("1. Structural
    and Layout Challenges" or an unnumbered "Structural and Layout Challenges")
    reduce to the same comparable form. Assumes `text` is already normalize()d."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", " ", text)).strip()


def _similar(a: str, b: str) -> bool:
    """True when two fuzzy heading keys denote the same heading — exact match,
    containment (one title is a prefix/subset of the other), or a small edit
    distance relative to length."""
    if not a or not b:
        return False
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    # containment, but only when the shorter key is substantial enough that a
    # substring match is meaningful (guards against tiny keys matching anything)
    if len(short) >= 6 and short in long:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _lis_length(seq: list) -> int:
    """Length of the longest strictly-increasing subsequence."""
    tails = []
    for x in seq:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


@register
class HeadingHierarchyCheck:
    name = "heading_hierarchy"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text) and bool(_try_source_titles(ctx))

    def run(self, ctx) -> CheckResult:
        toc_titles = list(_try_source_titles(ctx))
        qmd_titles = _qmd_headings(ctx.qmd_text)

        findings = []

        # ── map each source heading to its position in the .qmd (first fuzzy
        #    match, tolerant of section-number and punctuation differences) ──
        qmd_keys = [_fuzzy(t) for t in qmd_titles]
        matched_positions = []
        missing = []
        for title in toc_titles:
            key = _fuzzy(title)
            pos = next((i for i, q in enumerate(qmd_keys) if _similar(key, q)), None)
            if pos is None:
                missing.append(title)
            else:
                matched_positions.append(pos)
        for title in missing[:_MAX_LISTED]:
            findings.append(Finding(f"source heading missing from the .qmd: “{title}”",
                                    "warn", "headings"))
        if len(missing) > _MAX_LISTED:
            findings.append(Finding(f"… and {len(missing) - _MAX_LISTED} more", "info"))

        # ── ordering: headings that must move to restore source order ──
        reordered = len(matched_positions) - _lis_length(matched_positions)
        if reordered:
            findings.append(Finding(
                f"{reordered} surviving heading(s) appear out of their source order",
                "warn", "headings"))

        # ── count mismatch ──
        if abs(len(qmd_titles) - len(toc_titles)) > max(1, int(len(toc_titles) * _COUNT_TOLERANCE)):
            findings.append(Finding(
                f"source outline has {len(toc_titles)} heading(s) but the .qmd has "
                f"{len(qmd_titles)}", "warn", "headings"))

        status = "warn" if findings else "ok"
        summary = (f"{len(qmd_titles)} heading(s); {len(missing)} missing, "
                   f"{reordered} reordered vs the source outline")
        return CheckResult(
            self.name, status, summary,
            metric=f"{len(qmd_titles)} headings, {reordered} reordered",
            findings=findings,
        )
