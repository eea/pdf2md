"""Check: detect near-duplicate paragraphs (repair or conversion artifacts that
copy the same passage into the document more than once)."""

import re

from .. import CheckResult, Finding, register

_MIN_LEN = 100
_OVERLAP_THRESHOLD = 0.7
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _word_set(text: str) -> set:
    return set(w.lower() for w in _WORD_RE.findall(text))


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    shared = a & b
    union = a | b
    return len(shared) / len(union) if union else 0.0


@register
class DuplicationCheck:
    name = "duplication"

    def applicable(self, ctx) -> bool:
        return bool(ctx.qmd_text)

    def run(self, ctx) -> CheckResult:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", ctx.qmd_text)]
        # skip short paragraphs and pure image/figure markdown (captions of
        # unrelated figures often share boilerplate wording and aren't real
        # content duplication)
        paragraphs = [p for p in paragraphs
                      if len(p) > _MIN_LEN and not re.match(r"^!\[", p)
                      and not (p.startswith(":::") and "tbl-caption" in p)]

        word_sets = [_word_set(p) for p in paragraphs]
        findings = []
        seen_pairs = set()
        for i in range(len(paragraphs)):
            for j in range(i + 1, len(paragraphs)):
                ov = _overlap(word_sets[i], word_sets[j])
                if ov > _OVERLAP_THRESHOLD:
                    key = (i, j)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    a80 = paragraphs[i][:80].replace("\n", " ")
                    b80 = paragraphs[j][:80].replace("\n", " ")
                    findings.append(Finding(
                        f"potential duplicate paragraph (word overlap {ov:.0%}): "
                        f"\"{a80}...\" vs \"{b80}...\"",
                        "warn", location=f"paragraphs {i + 1} & {j + 1}"))

        status = "fail" if findings else "ok"
        summary = (f"{len(findings)} potential duplicate paragraph pair(s)" if findings
                   else "no duplicate paragraphs detected")
        return CheckResult(self.name, status, summary, findings=findings)
