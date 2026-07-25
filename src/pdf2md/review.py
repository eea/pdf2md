"""Vision patch step of the iterative repair loop (postfix.run_repair_loop).

Verify's checks are text heuristics; they can flag content that actually
survived (reworded prose, split tables). `inspect_pages` shows a vision model
the ORIGINAL page image (plus the rendered page, when available) for every page
still flagged and asks for exact old_string→new_string patches for each true
defect; `_apply_patches` applies them to the .qmd with plain string
replacement. Orchestration, re-verification and reporting live in postfix.py.
"""

import base64
import json
import logging
import re
from pathlib import Path

from .llm_client import _post_with_retries

log = logging.getLogger(__name__)

_MODEL = "google/gemini-2.5-flash"
_MAX_TOKENS = 4096
_PAGE_DPI = 150
_MAX_PAGES = 20
# The .qmd source rides along in the diagnose prompt so old_string can be quoted
# verbatim; bounded so a huge document cannot blow the request.
_MAX_QMD_CHARS = 150_000


def _page_to_image(pdf_path, page_num, dpi=_PAGE_DPI):
    """Render 1-based page to a base64 PNG data URI. None if file/page missing."""
    import fitz
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None
    doc = fitz.open(str(pdf_path))
    try:
        if not 1 <= page_num <= doc.page_count:
            return None
        png = doc[page_num - 1].get_pixmap(dpi=dpi).tobytes("png")
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    finally:
        doc.close()


_LOC_PAGE = re.compile(r"(?:~?\bpage\s+|\bp)(\d+)", re.IGNORECASE)


def _flagged_pages(verify_results):
    """{1-based source page: [check names]} from warn/fail check results."""
    flagged = {}

    def add(page, name):
        try:
            page = int(page)
        except (TypeError, ValueError):
            return
        if page < 1:
            return
        flagged.setdefault(page, []).append(name)

    for r in verify_results:
        if r.status not in ("warn", "fail"):
            continue
        d = r.detail or {}
        for c in d.get("clusters") or []:
            a, b = (list(c.get("pages", ())) + [0, 0])[:2]
            for p in range(int(a), int(b) + 1):
                add(p, r.name)
        for t in d.get("thin") or []:
            add(t.get("page"), r.name)
        for p in d.get("src_pages") or []:
            add(p, r.name)
        for f in r.findings:
            if f.severity == "info":
                continue
            m = _LOC_PAGE.search(f.location or "")
            if m:
                add(m.group(1), r.name)
    return dict(sorted(flagged.items()))


_PROMPT_TEMPLATE = """Compare LEFT (source PDF page {page}) with RIGHT (rendered markdown page {page}).{rendered_note}

The automated verify step flagged these issues on this page:
{issues}

For each difference you find, return a patch that would fix it. Patches are
applied with plain Python string replacement, so old_string MUST be text copied
character-for-character from the .QMD SOURCE below — include enough surrounding
text that it occurs exactly once — and new_string is the full replacement
(empty to delete). If a flagged check actually rendered correctly, list its
name in false_positives instead of patching.

Return ONLY valid JSON:
{{
  "page": {page},
  "verdict": "ok" | "issues",
  "patches": [
    {{
      "severity": "critical"|"moderate"|"cosmetic",
      "check": "which verify check this relates to",
      "location": "text description of where",
      "description": "what is wrong",
      "old_string": "exact text currently in the .qmd to replace",
      "new_string": "exact replacement text"
    }}
  ],
  "false_positives": ["check_name"]
}}

--- .QMD SOURCE ---
{qmd}"""


def _build_prompt(page_num, issues, qmd_text="", has_rendered=True):
    names = "\n".join(f"- {n}" for n in issues)
    note = ("" if has_rendered else
            "\n(The rendered page image is unavailable — judge from the source "
            "page image and the .qmd source text alone.)")
    return _PROMPT_TEMPLATE.format(page=page_num, issues=names,
                                   qmd=qmd_text, rendered_note=note)


def _parse_json(text):
    """Best-effort JSON object from a model reply (fenced or bare). None on failure."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    if not raw.startswith("{"):
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _vision_pass(source_pdf, rendered_pdf, flagged_pages, api_key, prompt_fn, label):
    """Shared per-page vision loop: page images → model → parsed JSON.

    Returns [{page, issues, parsed, cost_usd}] with `parsed` None on JSON
    failure, or {page, issues, error, cost_usd} on a failed call.
    """
    results = []
    if not flagged_pages:
        return results
    try:
        import fitz  # noqa: F401
    except ImportError:
        log.warning("review skipped: PyMuPDF not available")
        return results

    source_pdf = Path(source_pdf)
    if not source_pdf.exists():
        log.warning("review skipped: source PDF not found at %s", source_pdf)
        return results
    rendered_pdf = Path(rendered_pdf) if rendered_pdf else None

    pages = sorted(flagged_pages)
    if len(pages) > _MAX_PAGES:
        log.warning("review: %d pages flagged — limiting to first %d",
                    len(pages), _MAX_PAGES)
        pages = pages[:_MAX_PAGES]

    for pno in pages:
        issues = flagged_pages[pno]
        src_uri = _page_to_image(source_pdf, pno)
        if not src_uri:
            log.warning("review: cannot render source page %d — skip", pno)
            continue
        rend_uri = _page_to_image(rendered_pdf, pno) if rendered_pdf else None

        content = [
            {"type": "text", "text": prompt_fn(pno, issues, bool(rend_uri))},
            {"type": "image_url", "image_url": {"url": src_uri}},
        ]
        if rend_uri:
            content.append({"type": "image_url", "image_url": {"url": rend_uri}})

        payload = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": _MAX_TOKENS,
        }
        try:
            text, usage = _post_with_retries(
                api_key=api_key, payload=payload,
                label=f"{label}-p{pno}", timeout=120)
        except RuntimeError as e:
            log.warning("%s p%d failed: %s", label, pno, e)
            results.append({"page": pno, "issues": issues,
                            "error": str(e), "cost_usd": 0.0})
            continue

        parsed = _parse_json(text)
        if parsed is None:
            log.warning("%s p%d: JSON parse failed", label, pno)
        results.append({"page": pno, "issues": issues, "parsed": parsed,
                        "cost_usd": (usage or {}).get("cost", 0.0) or 0.0})
    return results


def _clean_patches(raw):
    """Keep only applicable patches: dicts with a non-empty old_string and a
    string new_string. Adds the applied/apply_note bookkeeping fields."""
    out = []
    for p in raw or []:
        if not isinstance(p, dict):
            continue
        old, new = p.get("old_string"), p.get("new_string")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            continue
        out.append({
            "severity": p.get("severity", "?"),
            "check": p.get("check", ""),
            "location": p.get("location", ""),
            "description": p.get("description", ""),
            "old_string": old,
            "new_string": new,
            "applied": False,
            "apply_note": "",
        })
    return out


def inspect_pages(source_pdf, rendered_pdf, flagged_pages, api_key, qmd_path=None):
    """Vision pass: for each flagged page ask for exact old→new patches.

    Returns a list of per-page defect dicts:
    {page, issues, verdict, patches, false_positives, cost_usd} (or an
    {page, issues, error, cost_usd} entry when the call failed).
    """
    qmd_text = ""
    if qmd_path:
        qmd_text = Path(qmd_path).read_text(encoding="utf-8")
        if len(qmd_text) > _MAX_QMD_CHARS:
            log.warning("review: .qmd is %d chars — sending the first %d only; "
                        "patches beyond that point cannot be produced",
                        len(qmd_text), _MAX_QMD_CHARS)
            qmd_text = qmd_text[:_MAX_QMD_CHARS]

    def prompt(pno, issues, has_rendered):
        return _build_prompt(pno, issues, qmd_text, has_rendered)

    out = []
    for r in _vision_pass(source_pdf, rendered_pdf, flagged_pages, api_key,
                          prompt, "diagnose"):
        if "error" in r:
            r.update({"verdict": "unclear", "patches": [], "false_positives": []})
            out.append(r)
            continue
        parsed = r.pop("parsed") or {}
        out.append({
            "page": r["page"],
            "issues": r["issues"],
            "verdict": parsed.get("verdict", "unclear"),
            "patches": _clean_patches(parsed.get("patches")),
            "false_positives": parsed.get("false_positives") or [],
            "cost_usd": r["cost_usd"],
        })
    return out


def _apply_patches(qmd_path, defects):
    """Apply each patch to the .qmd with string replacement. Returns the count
    of patches applied. Annotates every patch in place with applied/apply_note;
    a patch is skipped (never guessed) when its old_string is absent or occurs
    more than once."""
    qmd_path = Path(qmd_path)
    text = qmd_path.read_text(encoding="utf-8")
    applied = 0
    for page in defects:
        for p in page.get("patches") or []:
            old, new = p["old_string"], p["new_string"]
            if old == new:
                p["apply_note"] = "no-op (old_string == new_string)"
                continue
            n = text.count(old)
            if n == 0:
                p["apply_note"] = "old_string not found in .qmd"
            elif n > 1:
                p["apply_note"] = f"old_string ambiguous ({n} occurrences)"
            else:
                text = text.replace(old, new, 1)
                p["applied"] = True
                applied += 1
    if applied:
        qmd_path.write_text(text, encoding="utf-8")
        log.info("review: applied %d patch(es) to %s", applied, qmd_path.name)
    return applied
