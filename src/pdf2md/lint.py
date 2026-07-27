"""Emit-time lint gate for the final .qmd — deterministic, no LLM, no network.

The pipeline's repair passes (phase 2.5, postfix) run AFTER pass2's write and can
re-introduce structural hazards the earlier per-pass safeguards assume away. In
particular, a single stray code fence inverts every downstream "``` parity"
decision, so leaked page furniture ends up pasted *inside* a fence. This module
is the LAST thing to touch the .qmd:

  * SANITIZE the hazards that have exactly one correct repair, then
  * GATE — report as `fail` anything that cannot be safely auto-fixed (e.g. an
    image reference pointing at a file that was never extracted).

Observed failure modes this guards against (MRVPP ATBD, 2026-07):
  * nested image markdown  ![](![alt](path))  -> Quarto "File name too long" crash
  * leaked PDF page footers ("… Page N of M") pasted into the body / into a fence
  * doubled/mangled URL schemes ("https://doi.org/https:/doi.org/…")
  * a bare ``` fence adjacent to the frontmatter, masking whole regions as code
  * image references with no corresponding media file (unfixable -> fail)
"""
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Issue model
# ---------------------------------------------------------------------------

# severity semantics: 'fixed' = auto-repaired; 'warn' = reported, non-blocking;
# 'fail' = unfixable, gates the emit.
_SEVERITIES = ("fixed", "warn", "fail")


@dataclass
class Issue:
    code: str
    message: str
    severity: str
    count: int = 1

    def __post_init__(self):
        if self.severity not in _SEVERITIES:
            raise ValueError("bad severity: {}".format(self.severity))


@dataclass
class LintResult:
    path: Path
    issues: list = field(default_factory=list)
    changed: bool = False

    @property
    def fixes(self):
        return [i for i in self.issues if i.severity == "fixed"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def failures(self):
        return [i for i in self.issues if i.severity == "fail"]

    @property
    def ok(self):
        return not self.failures


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# ![](![alt](path))  — an image whose "URL" is itself an image node. The inner
# node is the real image; the outer wrapper is what breaks the renderer.
_NESTED_IMG = re.compile(r"!\[[^\]]*\]\(\s*(!\[[^\]]*\]\([^)]*\))\s*\)")

# a normal local image reference: ![alt](path) — path captured
_IMG_REF = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+?)\s*\)")

# a standalone running-footer line, optional "Version …" prefix then "Page N of M"
_PAGE_FOOTER = re.compile(r"^\s*(?:version\b.*?)?page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)

# a URL-ish run; broad enough to span a doubled scheme, stops at whitespace/closers
_URLISH = re.compile(r"https?:/{1,2}[^\s)>\]]+", re.IGNORECASE)

_FENCE_LINE = re.compile(r"(?m)^```")


# ---------------------------------------------------------------------------
# Fence-aware masking (so detectors don't fire on code samples)
# ---------------------------------------------------------------------------

def _mask_fences(text):
    """Blank out fenced code regions while preserving line structure, so text
    detectors never match a path/URL that only appears inside a code sample."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append("")
        elif in_fence:
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Sanitizers (each returns (text, n_changed))
# ---------------------------------------------------------------------------

def _unwrap_nested_images(text):
    total = 0
    while True:
        text, n = _NESTED_IMG.subn(lambda m: m.group(1), text)
        total += n
        if n == 0:
            break
    return text, total


def _strip_frontmatter_adjacent_fence(text):
    """Remove a bare ``` that sits right after the YAML frontmatter when it
    leaves the document's fence count odd — a classic converter artifact that
    otherwise swallows the whole body as a code block."""
    m = re.match(r"---\n.*?\n---\n", text, re.DOTALL)
    if not m:
        return text, 0
    rest = text[m.end():]
    m2 = re.match(r"((?:[ \t]*\n)*)(```[^\n]*)\n", rest)
    if not m2:
        return text, 0
    if m2.group(2).strip() != "```":
        return text, 0                      # a language-tagged fence is likely real
    if len(_FENCE_LINE.findall(text)) % 2 == 0:
        return text, 0                      # fences already balanced — leave it
    return text[:m.end()] + m2.group(1) + rest[m2.end():], 1


def _strip_page_footers(text):
    """Drop standalone leaked page-footer lines anywhere (incl. inside fences,
    where a parity slip can strand them). The pattern is specific enough that a
    real code line is very unlikely to match."""
    kept, removed = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s and len(s) < 90 and _PAGE_FOOTER.match(s):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def _fix_url(u):
    schemes = list(re.finditer(r"https?:/{1,2}", u, re.IGNORECASE))
    if len(schemes) > 1:                    # keep from the last embedded scheme
        u = u[schemes[-1].start():]
    return re.sub(r"^(https?):/(?!/)", lambda m: m.group(1) + "://", u, flags=re.IGNORECASE)


def _fix_doubled_urls(text):
    n = 0

    def repl(m):
        nonlocal n
        u = m.group(0)
        fixed = _fix_url(u)
        if fixed != u:
            n += 1
        return fixed

    return _URLISH.sub(repl, text), n


def sanitize(text):
    """Apply every deterministic single-correct-repair fix. Returns (text, fixes)."""
    issues = []
    text, n = _unwrap_nested_images(text)
    if n:
        issues.append(Issue("nested_image", "unwrapped {} nested image(s)".format(n), "fixed", n))
    text, n = _strip_frontmatter_adjacent_fence(text)
    if n:
        issues.append(Issue("stray_fence", "removed a stray code fence after the frontmatter", "fixed", n))
    text, n = _strip_page_footers(text)
    if n:
        issues.append(Issue("page_footer", "stripped {} leaked page-footer line(s)".format(n), "fixed", n))
    text, n = _fix_doubled_urls(text)
    if n:
        issues.append(Issue("doubled_url", "normalized {} malformed URL(s)".format(n), "fixed", n))
    return text, issues


# ---------------------------------------------------------------------------
# Checks (run on already-sanitized text; report what could not be fixed)
# ---------------------------------------------------------------------------

def check(text, qmd_dir):
    """Detect residual hazards. `fail` gates the emit; `warn` is informational."""
    issues = []
    masked = _mask_fences(text)

    if _NESTED_IMG.search(masked):
        issues.append(Issue("nested_image_residual",
                            "nested image markdown remains after sanitize", "fail"))

    if len(_FENCE_LINE.findall(text)) % 2 == 1:
        issues.append(Issue("unbalanced_code_fence",
                            "document has an odd number of ``` fences", "fail"))

    div_lines = [l for l in text.split("\n") if l.strip().startswith(":::")]
    if len(div_lines) % 2 == 1:
        issues.append(Issue("unbalanced_fenced_div",
                            "odd number of ::: fenced-div markers", "warn"))

    missing = []
    for m in _IMG_REF.finditer(masked):
        ref = m.group(1)
        if ref.startswith(("http://", "https://", "data:", "#", "/")):
            continue
        rel = ref.split("#", 1)[0].split("?", 1)[0]
        if rel and not (qmd_dir / rel).exists():
            missing.append(rel)
    if missing:
        uniq = sorted(set(missing))
        shown = ", ".join(uniq[:5]) + (" …" if len(uniq) > 5 else "")
        issues.append(Issue("missing_media",
                            "{} image reference(s) have no media file: {}".format(len(uniq), shown),
                            "fail", len(uniq)))
    return issues


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def lint_qmd(qmd_path, *, fix=True):
    """Sanitize (optionally in place) then gate the .qmd. Returns a LintResult.

    Writes atomically (tmp + os.replace) to match pass2's write contract, so a
    crash mid-write can't leave a truncated .qmd that reads as "already done".
    """
    qmd_path = Path(qmd_path)
    original = qmd_path.read_text(encoding="utf-8")
    text, fixes = sanitize(original)
    issues = fixes + check(text, qmd_path.parent)
    changed = fix and text != original
    if changed:
        tmp = qmd_path.with_suffix(qmd_path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, qmd_path)
    return LintResult(path=qmd_path, issues=issues, changed=changed)
