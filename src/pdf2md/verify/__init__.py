"""Verify phase: a registry of content checks.

After Pass 2, `run_verify` runs every registered check against a shared
`VerifyContext` and writes an information-only `verify_report.md` (+ `verify.json`).
Checks never mutate the .qmd; they surface findings for the operator review gate.

To add a check: drop a module in `verify/checks/` with a `name` / `applicable(ctx)` /
`run(ctx) -> CheckResult` class and decorate it `@register`.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ── Result / finding model ─────────────────────────────────────────────────────

@dataclass
class Finding:
    message: str
    severity: str = "warn"     # info | warn | fail
    location: str = ""         # e.g. "table 3", "p8", or a snippet anchor


@dataclass
class CheckResult:
    name: str
    status: str                # ok | warn | fail | skipped
    summary: str               # one-line headline
    metric: float = None       # optional number, e.g. coverage %
    findings: list = field(default_factory=list)
    detail: dict = None        # optional extras, e.g. {"effective": 99.1, "recovered": 21}


def ok(name, summary, metric=None):
    return CheckResult(name=name, status="ok", summary=summary, metric=metric)


def skipped(name, why):
    return CheckResult(name=name, status="skipped", summary=why)


# ── Context shared by all checks ────────────────────────────────────────────────

@dataclass
class VerifyContext:
    run_dir: Path
    original_pdf: Path         # raw source PDF, the reference for content fidelity
    working_pdf: Path          # chrome-stripped copy (Step 0)
    qmd_path: Path
    qmd_text: str
    detections: dict           # {"figures": [...], "other_detections": [...]}
    media_dir: Path
    rendered_pdf: Path = None  # optional; render-based checks skip when None

    @property
    def figures(self) -> list:
        return self.detections.get("figures", [])


# ── Registry ────────────────────────────────────────────────────────────────────

CHECKS = []


def register(cls):
    """Class decorator: instantiate and add to the registry."""
    CHECKS.append(cls())
    return cls


def _load_checks():
    from . import checks  # noqa: F401  (its __init__ imports each check module)


# ── Runner ──────────────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {"ok": 0, "skipped": 0, "warn": 1, "fail": 2}


def run_verify(ctx: VerifyContext) -> list:
    """Run every applicable registered check. Returns a list of CheckResult."""
    if not CHECKS:
        _load_checks()
    results = []
    for check in CHECKS:
        try:
            if not check.applicable(ctx):
                results.append(skipped(check.name, "inputs not available"))
                continue
            results.append(check.run(ctx))
        except Exception as exc:  # a broken check must not kill the run
            log.warning("check %s raised %s", getattr(check, "name", "?"), exc)
            results.append(CheckResult(check.name, "fail", f"check errored: {exc}"))
    return results


def overall_status(results: list) -> str:
    worst = 0
    for r in results:
        worst = max(worst, _SEVERITY_ORDER.get(r.status, 0))
    return {0: "ok", 1: "warn", 2: "fail"}[worst]


# ── Report writers ──────────────────────────────────────────────────────────────

_ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌", "skipped": "➖"}

# Per-check reading guide for the report: what the check verifies, and what a human
# can do when it warns. Keyed by check name; a check without an entry just gets no
# guide lines.
_CHECK_GUIDE = {
    "frontmatter": (
        "The .qmd starts with the YAML metadata block (title, date, category) that the "
        "publishing pipeline requires.",
        "Open the .qmd and add or correct the listed fields at the top of the file."),
    "structural_counts": (
        "Rough sanity numbers: how many figures and tables ended up in the .qmd vs. "
        "what was detected in the source.",
        "A large mismatch usually shows up in detail under figure_placement or "
        "table_coverage - fix it there."),
    "figure_placement": (
        "Every figure detected in the source PDF is actually referenced somewhere "
        "in the .qmd.",
        "Unplaced figures are in the media folder; add an image reference "
        "(![caption](media/file.png)) where the figure belongs."),
    "text_coverage": (
        "How much of the source text made it into the .qmd. 'Missing' means the "
        "sentence was not found; 'reworded' means the content is present but phrased "
        "differently (not counted as lost). Locations are approximate source pages.",
        "Open the source PDF at the listed page and copy the missing passage into the "
        ".qmd. Fragmented items (half-sentences) often mean the surrounding sentence "
        "is present in reworded form - check before re-adding."),
    "table_coverage": (
        "Whether each source table's data survived, matched by cell content. A table "
        "may legitimately be split or merged in the .qmd; only genuinely absent "
        "values count against it.",
        "For a thin table, find it on the listed source page and re-enter the missing "
        "values, or delete a duplicated partial table if the content moved."),
    "wide_table_legibility": (
        "Very wide tables get shrunk to fit the page; below ~6pt they become "
        "unreadable in the rendered PDF.",
        "Split the wide table into two, rotate it (landscape), or drop columns that "
        "repeat the same value."),
    "footnote_placement": (
        "Superscript footnote marks in the source should survive as markdown "
        "footnotes ([^1] … [^1]: text).",
        "Find the superscript marks on the source pages and add matching [^n] "
        "references plus [^n]: definitions in the .qmd."),
    "link_preservation": (
        "Hyperlink targets from the source PDF (including links whose URL is only in "
        "the PDF metadata, invisible in the text) still exist in the .qmd.",
        "Missing URLs are usually recoverable from the 'Source links' section if "
        "postfix ran; otherwise copy them from the source PDF's link annotations."),
    "heading_hierarchy": (
        "The .qmd's heading outline matches the source document's section structure.",
        "Compare the listed headings against the source table of contents and "
        "add/reorder the missing ones."),
    "math_presence": (
        "Equation regions in the source appear as math blocks in the .qmd.",
        "Locate the equation on the source page and transcribe it as $...$ / $$...$$."),
    "code_block_presence": (
        "Monospaced listings in the source appear as fenced code blocks.",
        "Copy the listing from the source page into a ``` fenced block."),
    "oversized_tables": (
        "Tables too large to transcribe are cropped as figures instead; this checks "
        "none slipped through as broken text.",
        "If one did, replace the mangled text with an image crop of the table."),
}


def _display_name(name: str) -> str:
    return {
        "frontmatter": "Metadata block", "structural_counts": "Structure counts",
        "figure_placement": "Figures", "text_coverage": "Missing text",
        "table_coverage": "Tables", "wide_table_legibility": "Wide tables",
        "footnote_placement": "Footnotes", "link_preservation": "Hyperlinks",
        "heading_hierarchy": "Heading structure", "math_presence": "Equations",
        "code_block_presence": "Code listings", "oversized_tables": "Oversized tables",
    }.get(name, name)


def _fmt_pages(pages: list) -> str:
    return ", ".join(str(p) for p in pages[:6]) + (" and more" if len(pages) > 6 else "")


def _text_section(md, r):
    d = r.detail or {}
    n_missing, total = d.get("missing_count", 0), d.get("total", 0)
    clusters = d.get("clusters") or []
    scattered = d.get("scattered", 0)
    if clusters:
        md.append(f"Most of the {n_missing} missing sentences sit in "
                  f"{len(clusters)} place{'s' if len(clusters) != 1 else ''}. "
                  + (f"The remaining {scattered} are scattered half-sentences; their "
                     f"other halves usually survived as rewordings, so check before "
                     f"re-adding anything (full list at the bottom)." if scattered else ""))
        md.append("")
        for c in clusters:
            a, b = c["pages"]
            where = f"Page {a}" if a == b else f"Pages {a}-{b}"
            md.append(f"### {where} (~{c['count']} sentences)")
            md.append("")
            for smp in c["samples"]:
                md.append(f"> \"{smp}\"")
            md.append("")
            spot = (f"In the converted file this belongs near the heading "
                    f"**\"{c['qmd_heading']}\"**." if c.get("qmd_heading") else
                    "In the converted file it belongs at the position matching those pages.")
            md.append(f"Open the original at page {a} and find the passage. {spot}")
            md.append("")
    elif total:
        md.append(f"{n_missing} of {total} sentences did not survive; they are "
                  f"scattered rather than clustered (full list at the bottom). "
                  f"Check the reworded list first - many scattered items are "
                  f"half-sentences whose other half survived.")
        md.append("")
    else:
        # no detail available (older run data) - fall back to the plain summary
        md.append(r.summary)
        md.append("")


def _footnote_section(md, r):
    d = r.detail or {}
    n, pages = d.get("src_count", 0), d.get("src_pages", [])
    if n:
        where = f" (pages {_fmt_pages(pages)})" if pages else ""
        md.append(f"The original uses ~{n} superscript footnote marks{where}; the "
                  f"converted file has none.")
    else:
        md.append(r.summary)
    md.append("")
    md.append("For each mark: note the superscript number and its footnote text at "
              "the bottom of that page, then add `[^n]` at the matching spot in the "
              "converted file and `[^n]: the text` below the paragraph.")
    md.append("")


def _code_section(md, r):
    d = r.detail or {}
    pages, blocks = d.get("src_pages", []), d.get("qmd_blocks", 0)
    where = f"Pages {_fmt_pages(pages)} of the original carry" if pages else "The original carries"
    md.append(f"{where} monospaced listings; the converted file has "
              f"{blocks} fenced code block{'s' if blocks != 1 else ''}. Compare those "
              f"pages against the converted file and copy anything still missing into "
              f"a ``` fence.")
    md.append("")


def _generic_section(md, r):
    md.append(r.summary)
    md.append("")
    shown = r.findings[:10]
    for f in shown:
        loc = f" ({f.location})" if f.location else ""
        md.append(f"- {f.message}{loc}")
    if len(r.findings) > len(shown):
        md.append(f"- ... and {len(r.findings) - len(shown)} more")
    if r.findings:
        md.append("")
    guide = _CHECK_GUIDE.get(r.name)
    if guide:
        md.append(guide[1])
        md.append("")


def write_report(results: list, run_dir: Path, meta: dict = None) -> Path:
    """Write verify_report.md (+ verify.json) into run_dir. Returns the .md path.

    The .md is written for a human fixing the document by hand: verdict first,
    problems grouped by place with both ends of each fix named, passed checks
    collapsed to a table. `meta` (optional) carries stem/pages/model/date, the
    postfix summary, and the render outcome for the header."""
    meta = meta or {}
    by = {r.name: r for r in results}
    overall = overall_status(results)
    issues = [r for r in results if r.status in ("warn", "fail")]
    issues.sort(key=lambda r: (r.status != "fail", r.name != "text_coverage"))

    title = meta.get("stem") or run_dir.name
    md = [f"# Conversion quality report - {title}", ""]
    from ..cost import fmt_eur
    cost_bits = []
    if meta.get("cost_convert") is not None:
        cost_bits.append(f"conversion {fmt_eur(meta['cost_convert'])}")
    if meta.get("cost_repair"):
        cost_bits.append(f"repair {fmt_eur(meta['cost_repair'])}")
    header_bits = [b for b in (meta.get("date"),
                               f"{meta['pages']} pages" if meta.get("pages") else None,
                               meta.get("model"),
                               " + ".join(cost_bits) if cost_bits else None) if b]
    if header_bits:
        md.append(" · ".join(header_bits))
        md.append("")
    # machine-readable line (replay and tooling read this, humans can ignore it)
    metrics = " ".join(f"{r.name}={r.metric}" for r in results
                       if isinstance(r.metric, (int, float)))
    md.append(f"<!-- verify: overall={overall} {metrics} -->")
    md.append("")

    icon = {"ok": "🟢", "warn": "🟡", "fail": "🔴"}[overall]
    verdict = {"ok": "clean - nothing needs your attention",
               "warn": f"good - {len(issues)} thing{'s' if len(issues) != 1 else ''} "
                       f"worth your attention",
               "fail": "problems - start with the failures below"}[overall]
    md.append(f"## {icon} Verdict: {verdict}")
    md.append("")

    rows = []
    tc = by.get("text_coverage")
    if tc and tc.metric is not None:
        d = tc.detail or {}
        extra = (f" ({d['missing_count']} of {d['total']} didn't)"
                 if d.get("total") else "")
        rows.append(("Text", tc.status, f"{tc.metric}% of sentences made it through{extra}"))
    tb = by.get("table_coverage")
    if tb and tb.metric is not None:
        n = (tb.detail or {}).get("n_tables")
        rows.append(("Tables", tb.status, f"{tb.metric}% of table content present"
                     + (f" ({n} source tables)" if n else "")))
    fp = by.get("figure_placement")
    if fp:
        rows.append(("Figures", fp.status, fp.summary))
    if meta.get("render") is not None:
        rows.append(("PDF render", "ok" if meta["render"] else "fail",
                     "ok" if meta["render"] else "failed - see the render log"))
    if rows or issues:
        md.append("| | |")
        md.append("|---|---|")
        dot = {"ok": "🟢", "warn": "🟡", "fail": "🔴", "skipped": "⚪"}
        for label, st, txt in rows:
            md.append(f"| {label} | {dot.get(st, '')} {txt} |")
        if issues:
            md.append("| Look at | " + " · ".join(_display_name(r.name) for r in issues) + " |")
        md.append("")

    if meta.get("postfixes"):
        # the numbers above are the FINAL, post-repair state; this line records what
        # the repair pass changed to get there (before → after coverage)
        deltas = []
        if meta.get("text_cov_before") is not None and tc and tc.metric is not None:
            deltas.append(f"text {meta['text_cov_before']}% → {tc.metric}%")
        if meta.get("table_cov_before") is not None and tb and tb.metric is not None:
            deltas.append(f"tables {meta['table_cov_before']}% → {tb.metric}%")
        delta_note = f" ({'; '.join(deltas)})" if deltas else ""
        md.append("The numbers above already include automatic repairs" + delta_note
                  + ". What the repair pass did: " + "; ".join(meta["postfixes"])
                  + ". You don't need to redo any of it.")
        md.append("")
    if issues:
        md.append("For everything below: page numbers refer to the **original PDF**, "
                  "and each item says where in the **converted file** the fix belongs.")
        md.append("")

    for n, r in enumerate(issues, 1):
        md.append(f"## {'🔴' if r.status == 'fail' else '🟡'} {n} · {_display_name(r.name)}")
        md.append("")
        if r.name == "text_coverage":
            _text_section(md, r)
        elif r.name == "footnote_placement":
            _footnote_section(md, r)
        elif r.name == "code_block_presence":
            _code_section(md, r)
        else:
            _generic_section(md, r)

    # minor notes: info-level findings from checks that passed
    minor = [(r, f) for r in results if r.status == "ok"
             for f in r.findings if f.severity == "info"]
    if minor:
        md.append("## ⚪ Minor, no action needed")
        md.append("")
        for r, f in minor:
            loc = f" ({f.location})" if f.location else ""
            md.append(f"- {f.message}{loc}")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## ✅ Everything that passed")
    md.append("")
    md.append("| Check | Result |")
    md.append("|---|---|")
    for r in results:
        if r.status in ("ok", "skipped"):
            res = r.summary if r.status == "ok" else "not applicable for this run"
            md.append(f"| {_display_name(r.name)} | {res} |")
    md.append("")

    if tc and tc.findings:
        listed = [f for f in tc.findings if f.message.startswith("missing:")]
        if listed:
            md.append("<details>")
            md.append(f"<summary>Full list of missing sentences</summary>")
            md.append("")
            for f in listed:
                loc = f"{f.location}: " if f.location else ""
                md.append(f"- {loc}\"{f.message[9:]}\"")
            md.append("")
            md.append("</details>")
            md.append("")

    md_path = run_dir / "verify_report.md"
    # house style: plain hyphens only, whatever the check summaries contain
    text = "\n".join(md).replace("—", "-").replace("–", "-")
    md_path.write_text(text, encoding="utf-8")

    json_path = run_dir / "verify.json"
    payload = {
        "overall": overall_status(results),
        "checks": [
            {
                "name": r.name, "status": r.status, "summary": r.summary,
                "metric": r.metric,
                "findings": [
                    {"message": f.message, "severity": f.severity, "location": f.location}
                    for f in r.findings
                ],
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote %s (overall: %s)", md_path.name, overall_status(results))
    return md_path
