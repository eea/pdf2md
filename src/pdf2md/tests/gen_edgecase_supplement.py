#!/usr/bin/env python3
"""Regenerate the supplementary pages for pdf2md_edgecases.pdf.

Run from the repo root (or anywhere — paths are absolute) to append 3 pages
covering edge cases the original 11-page PDF doesn't test:

  1. Clickable hyperlink annotations
  2. Raw \\DATA, <sub>, <sup>, <br> patterns (_sanitise_for_typst)
  3. $$ display math + ambiguous notation (math false-positive guards)
  4. Running header/footer variants (chrome.py per-page detection)
  5. "Page N of M" footer pattern (lint.py chrome-leak guard)
  6. Nested/relative image paths and doubled URL schemes (lint.py)

Usage:
  python3 tests/gen_edgecase_supplement.py
    → merges 3 pages into src/pdf2md/tests/fixtures/pdf2md_edgecases.pdf
    → original backed up to pdf2md_edgecases.bak.pdf

Requires: fpdf2, pypdf
  pip install fpdf2 pypdf
"""

import shutil
import sys
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    sys.exit("fpdf2 is required: pip install fpdf2")

try:
    from pypdf import PdfWriter
except ImportError:
    sys.exit("pypdf is required: pip install pypdf")


# ── Paths ──────────────────────────────────────────────────────────────────────
# Script lives at src/pdf2md/tests/gen_edgecase_supplement.py
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MAIN_PDF = FIXTURES / "pdf2md_edgecases.pdf"
# Standard TTF fonts shipped with most Linux distros
DEJAVU = Path("/usr/share/fonts/truetype/dejavu")


class EdgecasePDF(FPDF):
    """3-page supplement with variant running headers/footers per page."""

    # Each page gets a different header text to test chrome.variant detection
    _header_titles = {1: "Introduction", 2: "Methods", 3: "Results"}

    def header(self):
        self.set_font("DejaVuSans", "I", 7)
        self.set_text_color(150, 150, 150)
        title = self._header_titles.get(self.page_no(), "Appendix")
        self.cell(0, 4, f"  Edge Case Supplement — pdf2md v2  |  {title}", align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVuSans", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()} of 3", align="C")

    def _title(self, num, title):
        self.set_font("DejaVuSans", "B", 13)
        self.set_text_color(0, 70, 130)
        self.cell(0, 8, f"{num} {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 70, 130)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def _body(self, text):
        self.set_font("DejaVuSans", "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def _code(self, text):
        self.set_font("DejaVuSansMono", "", 7.5)
        self.set_fill_color(238, 238, 238)
        self.set_text_color(50, 50, 50)
        for line in text.strip().split("\n"):
            self.cell(0, 4, f"  {line}", new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def _note(self, text):
        self.set_fill_color(255, 255, 220)
        self.set_draw_color(180, 150, 0)
        x, y = self.l_margin, self.get_y()
        w = self.w - self.l_margin - self.r_margin
        self.set_line_width(0.3)
        self.set_font("DejaVuSans", "", 8)
        self.set_text_color(100, 80, 0)
        self.rect(x, y, w, 24)
        self.set_xy(x + 2, y + 2)
        self.multi_cell(w - 4, 4, text)
        self.ln(26)

    def _link(self, text, url):
        """A blue italic clickable link."""
        self.set_text_color(0, 0, 200)
        self.set_font("DejaVuSans", "I", 9)
        w = self.get_string_width(text)
        self.write(4.5, text)
        self.link(self.l_margin + self.get_x() - w, self.get_y(), w, 4.5, url)
        self.set_text_color(0, 0, 0)
        self.set_font("DejaVuSans", "", 9)
        self.ln(5)


def build_supplement() -> FPDF:
    pdf = EdgecasePDF()
    pdf.set_auto_page_break(auto=True, margin=22)

    # Register fonts
    pdf.add_font("DejaVuSans", "", str(DEJAVU / "DejaVuSans.ttf"))
    pdf.add_font("DejaVuSans", "B", str(DEJAVU / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVuSans", "I", str(DEJAVU / "DejaVuSans-Oblique.ttf"))
    pdf.add_font("DejaVuSansMono", "", str(DEJAVU / "DejaVuSansMono.ttf"))

    # ═══════════════════════════════════════════════════════════════════════════
    # Page 1  — Hyperlinks + \DATA / <sub>/<sup>/<br> patterns
    # Header: "Introduction"
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()

    pdf._title("10", "Hyperlink Annotations (Clickable PDF Links)")
    pdf._body(
        "Tests whether the converter preserves clickable hyperlink annotations. "
        "The URLs below are real PDF link annotations, not plain text — the "
        "postfix link-recovery step (e29d0b1) reads these from PDF annotation "
        "objects, not from visible text."
    )
    pdf.ln(2)

    for label, url in [
        ("Click here to visit example.com", "https://example.com/link-test"),
        ("contact@example.org", "mailto:contact@example.org"),
        ("pdf2md GitHub repository", "https://github.com/eea/pdf2md"),
        ("https://doi.org/10.5194/essd-14-2785-2022", "https://doi.org/10.5194/essd-14-2785-2022"),
    ]:
        pdf.set_font("DejaVuSans", "B", 9)
        pdf.cell(0, 5, f"  — ", new_x="LMARGIN", new_y="NEXT")
        pdf._link(label, url)
    pdf.ln(2)

    # ── \DATA / HTML patterns ───────────────────────────────────────────────
    pdf._title("11", "Typst-Escape Patterns (\\DATA, HTML Tags)")
    pdf._body(
        "Tests _sanitise_for_typst() (Phase 2.5c). Bare backslashes before "
        "capitals are Typst escapes. Raw <br>/<sub>/<sup> tags are dropped "
        "by Typst. All must survive as correct Markdown."
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "11.1 Bare backslash before capitals:", new_x="LMARGIN", new_y="NEXT")
    pdf._code(
        r"""The \DATA field must be preserved. The \TABLE reference
should also survive. \TEXT and \VALUE are common.
Even \XYZ should pass through intact.

Backslash before lowercase stays: \name, \path.
Single backslash in LaTeX: \alpha, \beta, \gamma."""
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "11.2 Raw HTML sub/sup/br tags:", new_x="LMARGIN", new_y="NEXT")
    pdf._code(
        """The formula<sub>2</sub>H<sub>2</sub>O must survive.
Superscripts: E = mc<sup>2</sup>, refs<sup>[1]</sup>.

Line breaks: first line<br>second line<br>third line
<br>standalone break

Mixed: H<sub>2</sub>SO<sub>4</sub> at 25<sup>°</sup>C"""
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "11.3 Combined:", new_x="LMARGIN", new_y="NEXT")
    pdf._code(
        r"""The \DATA field<sub>raw</sub> was validated.<br>
The \TEXT value was <sub>nested</sub><sup>[3]</sup>."""
    )

    pdf._note(
        "VERIFY: \\\\DATA etc. verbatim in .qmd. After sanitiser: "
        "backslash+capital → \\\\\\\\DATA, <sub>→~...~, <sup>→^...^, "
        "<br>→newline."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Page 2  — $$ display math + ambiguous notation
    # Header: "Methods"
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._title("12", "LaTeX Math Delimiters and Ambiguous Notation")
    pdf._body(
        "Tests (a) whether $$...$$ display math survives the sanitiser "
        "unchanged, and (b) whether ambiguous notation like error bars, "
        "thresholds, and set notation is correctly classified as non-math "
        "(tightened in commit cd5e9b9)."
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "12.1 Display math (sanitiser must skip $$...$$):", new_x="LMARGIN", new_y="NEXT")
    pdf._code(
        r"""The Einstein field equations:

$$ G_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu} $$

And the fundamental theorem of calculus:

$$ \int_a^b f(x)\,dx = F(b) - F(a) $$"""
    )
    pdf._body(
        "Inline math: $E = mc^2$ is embedded here, and $\\alpha = "
        "\\frac{1}{2}$ appears in prose."
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "12.2 Ambiguous notation (false-positive guards):", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf._body("Error bars — must NOT become $$ math:")
    pdf._code(
        """Temperature: 23.5 deg C +/- 0.5 deg C (1 sigma)
Pressure: 1013 +/- 15 hPa (95% confidence)
Threshold: values > 2.5 sigma from the mean"""
    )
    pdf._body("Set notation — must NOT become $$ math:")
    pdf._code(
        """IDs: {A001, A002, A003} — set of identifiers
Interval: [0, 1], values outside excluded
Coords: (x, y) = (12.5, 34.2) deg N"""
    )
    pdf._body("Statistical expressions in prose:")
    pdf._code(
        r"""p < 0.001 (statistically significant)
n = 15 samples (mean +/- SD)
R-squared = 0.94, adjusted R-squared = 0.91"""
    )
    pdf._note(
        "VERIFY: 12.1 preserves $$...$$ intact. 12.2 stays plain text, "
        "NOT wrapped in $$. Error bars (+/-), set notation ({...}), and "
        "stats (p < 0.001) are TEXT — tightened equation detection must "
        "reject them."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Page 3  — Chrome variants + page numbers + image/lint patterns
    # Header: "Results"
    # ═══════════════════════════════════════════════════════════════════════════
    pdf.add_page()
    pdf._title("13", "Chrome Variants, Page Numbers, Lint Patterns")
    pdf._body(
        "This page has a different running header (\"Results\") than previous "
        "pages (\"Introduction\", \"Methods\"). The chrome.py module detects "
        "per-page header/footer text variants via majority-vote over margin "
        "regions. Every supplement page has a \"Page N of 3\" footer."
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "13.1 Page-number leakage guard:", new_x="LMARGIN", new_y="NEXT")
    pdf._body(
        "The footer on every supplement page reads \"Page N of 3\" — tests "
        "the lint.py chrome-leak guard. Inline references to pages are "
        "intentional content (NOT chrome):"
    )
    pdf._code(
        '''"see page 12 for the full table"
"continued on page 13"
"as shown on the next page"    # <-- NOT chrome'''
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "13.2 Nested/relative image paths:", new_x="LMARGIN", new_y="NEXT")
    pdf._body("The lint.py module catches nested image markdown — pattern "
              "that causes Quarto \"File name too long\" crashes:")
    pdf._code(
        """BAD (nested — lint must flag):
  ![](../media/image1.png)
  ![](././media/fig02.png)

BAD (invented path — no file exists):
  ![alt](edgecases-media/image42.jpeg)
  ![](edgecases-media/chart.png)"""
    )

    pdf.set_font("DejaVuSans", "B", 9)
    pdf.cell(0, 4.5, "13.3 Doubled URL schemes:", new_x="LMARGIN", new_y="NEXT")
    pdf._body("The linter catches doubled URL schemes:")
    pdf._code(
        """BAD (lint must flag):
  https://doi.org/https://doi.org/10.5194/essd-14-2785-2022
  https://doi.org/doi.org/10.1234/abc"""
    )

    pdf._note(
        "VERIFY: chrome.py detects 3 header variants (Introduction / Methods "
        "/ Results) + uniform footer. lint.py flags the BAD patterns in "
        "13.2-13.3."
    )

    return pdf


def main():
    if not MAIN_PDF.exists():
        sys.exit(f"Main PDF not found: {MAIN_PDF}\nRun from the repo root.")

    if not DEJAVU.exists():
        sys.exit(f"DejaVu font not found at {DEJAVU}\nInstall fonts-dejavu or edit DEJAVU path.")

    supp = build_supplement()
    supp_path = FIXTURES / "edgecase_supplement.pdf"
    supp.output(str(supp_path))
    print(f"Supplement written: {supp_path} ({supp.pages_count} pages)")

    backup = MAIN_PDF.with_suffix(".bak.pdf")
    shutil.copy2(str(MAIN_PDF), str(backup))

    merged = PdfWriter()
    merged.append(str(MAIN_PDF))
    merged.append(str(supp_path))
    merged.write(str(MAIN_PDF))

    supp_path.unlink()
    print(f"Merged into: {MAIN_PDF} ({len(merged.pages)} pages)")
    print(f"Backup at:   {backup}")


if __name__ == "__main__":
    main()