"""Check: mathematical content in the source survives as math in the .qmd.

Scans the source PDF for equation-like lines (a high density of math glyphs such as
∫∑√∇∂±×÷≤≥≠∞→⇒ and Greek letters, or literal LaTeX $ delimiters) and groups consecutive
ones into regions. Compares that against the inline $…$ and display $$…$$ math blocks in
the .qmd, warning on a significant asymmetry. Only runs when the source actually has math."""

import re
from functools import lru_cache

from .. import CheckResult, Finding, register
from ..textutil import pdf_lines

# math glyphs commonly rendered directly into PDF text for equations
_MATH_SYMBOLS = set("∫∑∏√∛∜∇∂±∓×÷⋅≤≥≠≈≡∝∞→⇒⇔∈∉⊂⊆∪∩∀∃∅ℝℤℕℚαβγδεθλμπρσφψωΩΔΣΠ")
# a line counts as an equation region seed if it has this many math glyphs …
_MATH_MIN_SYMBOLS = 2
# … or carries a LaTeX delimiter (rare in rendered PDFs, but decisive when present)
_LATEX_DELIM = re.compile(r"\$\$?")

# .qmd math: display $$…$$ first, then inline $…$ that isn't part of a $$ pair
_QMD_DISPLAY = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_QMD_INLINE = re.compile(r"(?<!\$)\$(?!\$)[^\n$]+?\$(?!\$)")


def _line_is_math(text: str) -> bool:
    if _LATEX_DELIM.search(text):
        return True
    return sum(1 for ch in text if ch in _MATH_SYMBOLS) >= _MATH_MIN_SYMBOLS


@lru_cache(maxsize=8)
def _source_math_regions(pdf_str: str, mtime: float) -> int:
    """Count of contiguous equation-like line regions in the source PDF."""
    regions, in_region = 0, False
    for _, txt in pdf_lines(pdf_str):
        if _line_is_math(txt):
            if not in_region:
                regions += 1
                in_region = True
        else:
            in_region = False
    return regions


def _try_source_math(ctx) -> int:
    if not (ctx.original_pdf and ctx.original_pdf.exists()):
        return 0
    try:
        return _source_math_regions(str(ctx.original_pdf), ctx.original_pdf.stat().st_mtime)
    except Exception:
        return 0


def _qmd_math_count(qmd_text: str) -> int:
    display = _QMD_DISPLAY.findall(qmd_text)
    remainder = _QMD_DISPLAY.sub(" ", qmd_text)
    inline = _QMD_INLINE.findall(remainder)
    return len(display) + len(inline)


@register
class MathPresenceCheck:
    name = "math_presence"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text) and _try_source_math(ctx) > 0

    def run(self, ctx) -> CheckResult:
        src = _try_source_math(ctx)
        qmd = _qmd_math_count(ctx.qmd_text)
        preserved = min(src, qmd)

        findings = []
        # significant asymmetry: the .qmd captured well under half the source's math
        if qmd < src / 2:
            findings.append(Finding(
                f"source has ~{src} equation region(s) but the .qmd has only {qmd} "
                f"math block(s) — equations may have been dropped or rendered as prose",
                "warn", "math"))
        elif qmd > src * 2:
            findings.append(Finding(
                f"the .qmd has {qmd} math block(s) for ~{src} source equation region(s) "
                f"— possible over-detection", "info", "math"))

        status = "warn" if any(f.severity == "warn" for f in findings) else "ok"
        summary = f"~{src} source equation region(s); {qmd} math block(s) in the .qmd"
        return CheckResult(
            self.name, status, summary,
            metric=f"{preserved}/{src} equations preserved",
            findings=findings,
        )
