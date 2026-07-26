"""Check: leftover Word/Office field-code artifacts and broken cross-references
that sometimes survive PDF conversion (e.g. "Error! Reference source not
found.", "#REF!", broken bookmarks/hyperlinks)."""

import re

from .. import CheckResult, Finding, register

_PATTERNS = (
    re.compile(r"[Ee]rror!\s*Reference source not found"),
    re.compile(r"[Ee]rror!\s*Bookmark not defined"),
    re.compile(r"[Ee]rror!\s*Hyperlink reference not valid"),
    re.compile(r"[Ee]rror!\s*(?=[^\n]*(?:[Ff]igure|[Tt]able)\s*\d)"),
    re.compile(r"#REF!"),
)


@register
class ArtifactsCheck:
    name = "artifacts"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text)

    def run(self, ctx) -> CheckResult:
        findings = []
        lines = ctx.qmd_text.split("\n")
        seen = set()
        for i, line in enumerate(lines, start=1):
            for pat in _PATTERNS:
                m = pat.search(line)
                if not m:
                    continue
                key = (i, m.group(0))
                if key in seen:
                    continue
                seen.add(key)
                snippet = line.strip()[:100]
                findings.append(Finding(
                    f"artifact: \"{m.group(0)}\" in \"{snippet}\"",
                    "warn", location=f"line {i}"))

        status = "fail" if findings else "ok"
        summary = (f"{len(findings)} leftover conversion artifact(s) found" if findings
                   else "no leftover Office/reference artifacts found")
        return CheckResult(self.name, status, summary, findings=findings)
