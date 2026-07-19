"""Check: monospaced/code content in the source survives as fenced code in the .qmd.

Scans the source PDF for pages carrying a run of monospaced text — detected via the
PyMuPDF span monospaced font flag or a font name containing Mono/Courier/Code/Consolas —
and compares that page count against the number of fenced ``` code blocks in the .qmd.
Warns on a significant asymmetry. Only runs when the source actually has code."""

import re
from functools import lru_cache

from .. import CheckResult, Finding, register

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

_MONO_FONT_FLAG = 1 << 3  # PyMuPDF span flag bit 3 = monospaced
_MONO_NAME_RE = re.compile(r"mono|courier|consol|code", re.IGNORECASE)
# a page needs at least this many monospaced spans to count as a code page — one stray
# glyph in a monospaced font (e.g. a ™) shouldn't flag the whole page
_MONO_SPANS_MIN = 4


def _span_is_mono(span: dict) -> bool:
    if span.get("flags", 0) & _MONO_FONT_FLAG:
        return True
    return bool(_MONO_NAME_RE.search(span.get("font", "")))


@lru_cache(maxsize=8)
def _source_code_pages(pdf_str: str, mtime: float) -> int:
    """Count of source pages carrying a run of monospaced text."""
    pages = 0
    doc = fitz.open(pdf_str)
    try:
        for pno in range(doc.page_count):
            mono = 0
            for block in doc[pno].get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip() and _span_is_mono(span):
                            mono += 1
            if mono >= _MONO_SPANS_MIN:
                pages += 1
    finally:
        doc.close()
    return pages


def _try_source_code(ctx) -> int:
    if not (_FITZ_AVAILABLE and ctx.original_pdf and ctx.original_pdf.exists()):
        return 0
    try:
        return _source_code_pages(str(ctx.original_pdf), ctx.original_pdf.stat().st_mtime)
    except Exception:
        return 0


def _qmd_code_blocks(qmd_text: str) -> int:
    """Fenced ``` code blocks, excluding Quarto raw blocks like ```{=html}."""
    blocks, in_fence = 0, False
    for line in qmd_text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            if not in_fence:
                in_fence = True
                if not s[3:].strip().startswith("{="):  # skip raw ```{=html} / ```{=typst}
                    blocks += 1
            else:
                in_fence = False
    return blocks


@register
class CodeBlockPresenceCheck:
    name = "code_block_presence"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text) and _try_source_code(ctx) > 0

    def run(self, ctx) -> CheckResult:
        src = _try_source_code(ctx)
        qmd = _qmd_code_blocks(ctx.qmd_text)
        preserved = min(src, qmd)

        findings = []
        if qmd < src / 2:
            findings.append(Finding(
                f"source has ~{src} page(s) of monospaced/code text but the .qmd has "
                f"only {qmd} fenced code block(s) — code listings may have been lost",
                "warn", "code"))
        elif qmd > src * 2:
            findings.append(Finding(
                f"the .qmd has {qmd} fenced code block(s) for ~{src} monospaced source "
                f"page(s) — possible over-detection", "info", "code"))

        status = "warn" if any(f.severity == "warn" for f in findings) else "ok"
        summary = f"~{src} monospaced source page(s); {qmd} fenced code block(s) in the .qmd"
        return CheckResult(
            self.name, status, summary,
            metric=f"{preserved}/{src} code blocks preserved",
            findings=findings,
        )
