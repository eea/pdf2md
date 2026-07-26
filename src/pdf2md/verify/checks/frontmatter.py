"""Check: required frontmatter fields are present (the PR-gate schema)."""

import re

from .. import CheckResult, Finding, register

REQUIRED = ("title", "subtitle", "date")


@register
class FrontmatterCheck:
    name = "frontmatter"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text)

    def run(self, ctx) -> CheckResult:
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", ctx.qmd_text, re.DOTALL)
        if not m:
            return CheckResult(self.name, "fail", "no YAML frontmatter block",
                               findings=[Finding("missing frontmatter", "fail")])
        fm = m.group(1)
        findings = []
        for key in REQUIRED:
            if not re.search(rf"^\s*{key}\s*:", fm, re.MULTILINE):
                findings.append(Finding(f"missing required field: {key}", "fail"))
        # category is optional (verbatim 1:1); validate only when present
        cat = re.search(r"^\s*category\s*:\s*(\S+)", fm, re.MULTILINE)
        if cat:
            val = cat.group(1).strip().strip('"').strip("'")
            if not re.match(r"^[a-z][a-z0-9_-]*$", val):
                findings.append(Finding(f"invalid category: {val}", "fail"))
        status = "fail" if findings else "ok"
        return CheckResult(self.name, status,
                           "all required fields present" if status == "ok"
                           else f"{len(findings)} frontmatter problem(s)",
                           findings=findings)