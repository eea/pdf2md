"""Check: the frontmatter YAML block is cleanly closed, with no body text leaked
inside the `---` delimiters or between the closing delimiter and the blank line
that should follow it."""

import re

from .. import CheckResult, Finding, register

# A plausible YAML `key: value` (or `key:` for block scalars) line.
_KEY_VALUE_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_-]*\s*:\s*.*$")
# Lines that are clearly not YAML: addresses, phone/fax numbers, bare postal
# codes, standalone URLs/domains, etc. These are the telltale sign of body
# text (e.g. a document's title-page footer) leaking into the frontmatter.
_BODY_LIKE_RE = re.compile(
    r"^\s*(Tel\.?:|Fax:|www\.|http|.*\.(com|eu|org|net)\b|"
    r"\d{1,5}\s+\w+.*\d|"          # street address with house number
    r"[A-Z][a-z]+ \d{4,5}\b|"       # "Copenhagen K" style, city + postal code
    r"[A-Za-z ]+\d{1,2}\s*$)"        # trailing street number e.g. "Kongens Nytorv 6"
)


@register
class YamlBoundaryCheck:
    name = "yaml_boundary"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text)

    def run(self, ctx) -> CheckResult:
        findings = []
        text = ctx.qmd_text
        m = re.match(r"^---\s*\n(.*?\n)---\s*\n", text, re.DOTALL)
        if not m:
            return CheckResult(self.name, "ok", "no frontmatter block to check")

        block = m.group(1)
        lines = block.split("\n")
        for i, line in enumerate(lines, start=2):  # line 1 is the opening ---
            stripped = line.strip()
            if not stripped:
                continue
            if _KEY_VALUE_RE.match(stripped):
                continue
            if _BODY_LIKE_RE.match(stripped) or self._looks_like_body(stripped):
                findings.append(Finding(
                    f"frontmatter boundary error: body-like text inside YAML block: "
                    f"\"{stripped[:60]}\"", "warn", location=f"frontmatter line {i}"))

        # Check what immediately follows the closing --- : it should be blank,
        # not more body text before the next blank line.
        after = text[m.end():]
        after_lines = after.split("\n")
        for line in after_lines:
            if line.strip() == "":
                break
            # A heading or normal markdown para right after frontmatter is fine;
            # this only flags when the *block itself* mis-closed and left
            # trailing junk directly glued to the closing delimiter (rare, but
            # covered defensively).
            if _BODY_LIKE_RE.match(line.strip()):
                findings.append(Finding(
                    f"frontmatter boundary error: body-like text immediately after "
                    f"closing '---': \"{line.strip()[:60]}\"", "warn"))
            break

        status = "fail" if findings else "ok"
        summary = (f"{len(findings)} frontmatter boundary problem(s)" if findings
                   else "frontmatter block cleanly closed")
        return CheckResult(self.name, status, summary, findings=findings)

    @staticmethod
    def _looks_like_body(line: str) -> bool:
        """Catch-all: a line with no colon at all inside a YAML block is
        almost certainly leaked body text (plain prose/address lines)."""
        if ":" in line:
            return False
        # short bare words like a country name, or lines ending in a digit
        # (street numbers, postal codes) are suspicious
        return bool(re.match(r"^[A-Za-z0-9 .,-]+$", line)) and len(line) < 80
