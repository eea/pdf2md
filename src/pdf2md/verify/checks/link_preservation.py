"""Check: hyperlinks in the source PDF survive into the .qmd.

Reads the URL link annotations from the source PDF (PyMuPDF get_links()) and confirms
each target URL still appears somewhere in the .qmd. A dropped link means the reader
lost a citation or reference the original document carried."""

import re
from functools import lru_cache

from .. import CheckResult, Finding, register

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

_URI_SCHEMES = ("http://", "https://", "ftp://", "mailto:", "www.")
_MAX_LISTED = 30


@lru_cache(maxsize=8)
def _source_uris(pdf_str: str, mtime: float) -> tuple:
    """Unique external target URLs from the source PDF's link annotations."""
    seen = []
    doc = fitz.open(pdf_str)
    try:
        for pno in range(doc.page_count):
            for link in doc[pno].get_links():
                uri = (link.get("uri") or "").strip()
                if uri and uri.lower().startswith(_URI_SCHEMES) and uri not in seen:
                    seen.append(uri)
    finally:
        doc.close()
    return tuple(seen)


def _try_source_uris(ctx) -> tuple:
    if not (_FITZ_AVAILABLE and ctx.original_pdf and ctx.original_pdf.exists()):
        return ()
    try:
        return _source_uris(str(ctx.original_pdf), ctx.original_pdf.stat().st_mtime)
    except Exception:
        return ()


def _uri_in_qmd(uri: str, qmd_lower: str) -> bool:
    u = uri.rstrip("/").lower()
    if u in qmd_lower:
        return True
    # tolerate the .qmd dropping the scheme (e.g. "www.foo.org" or "foo.org/x")
    stripped = re.sub(r"^\w+://", "", u)
    return bool(stripped) and stripped in qmd_lower


@register
class LinkPreservationCheck:
    name = "link_preservation"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text) and bool(_try_source_uris(ctx))

    def run(self, ctx) -> CheckResult:
        uris = _try_source_uris(ctx)
        qmd_lower = ctx.qmd_text.lower()

        missing = [u for u in uris if not _uri_in_qmd(u, qmd_lower)]
        preserved = len(uris) - len(missing)

        findings = [Finding(f"source link not found in the .qmd: {u}", "warn", "links")
                    for u in missing[:_MAX_LISTED]]
        if len(missing) > _MAX_LISTED:
            findings.append(Finding(f"… and {len(missing) - _MAX_LISTED} more", "info"))

        status = "warn" if missing else "ok"
        summary = (f"{preserved}/{len(uris)} source link(s) preserved"
                   + (f", {len(missing)} missing" if missing else ""))
        return CheckResult(
            self.name, status, summary,
            metric=f"{preserved}/{len(uris)} preserved",
            findings=findings,
        )
