# pdf2md

pdf2md converts PDF documents into editable Markdown. It is aimed at technical
documents, reports and manuals: text, tables, figures and structure are carried
over, and each conversion is checked against the original. The conversion
itself is done by an LLM. The main output format is Quarto `.qmd`; plain `.md`
and GitHub-Flavored `.gfm` are also supported.

## Quick start

```bash
# Install (editable, so a git pull is enough to update later)
pip install -e .

# One-time setup (API key + default model)
python3 pdf2md.py --setup

# Convert a PDF (1:1 fidelity, all content preserved)
python3 pdf2md.py document.pdf

# Convert + strip chrome for web output (repair loop runs by default)
python3 pdf2md.py document.pdf --strip-chrome

# Batch convert a directory
python3 pdf2md.py inbox/ --out output/

# Output format
python3 pdf2md.py document.pdf --format gfm    # GitHub-Flavored Markdown
python3 pdf2md.py document.pdf --format md     # Plain Markdown
python3 pdf2md.py document.pdf --render        # Also render to PDF via Quarto

# Use a YAML frontmatter template (with --format qmd or gfm)
python3 pdf2md.py document.pdf --template path/to/template.qmd
python3 pdf2md.py document.pdf --template https://raw.githubusercontent.com/org/repo/main/template.qmd
```

With `--template`, the YAML frontmatter from the template is injected into the
conversion prompt. The LLM fills in the document-specific values (title, date
and so on) but keeps the field set and order from the template, so all
converted documents end up with the same header layout.

## How it works

A PDF becomes Markdown in five phases. The text goes through the LLM, the
images go around it.

<img src="docs/pipeline.svg" alt="How pdf2md works" width="840">

**Phase 1 — Detect.** Figures are cropped out of the PDF before conversion and
replaced with numbered `[FIG_n]` boxes. The PDF is processed 1:1 — all running
headers, footers, and page numbers are preserved so the verification and repair
steps can compare against a faithful copy.

**Phase 2 — Convert.** The LLM transcribes text and carries figure tokens
through. Tables are never cropped, so the model can transcribe their content.

**Phase 3 — Verify.** Every converted document is checked against the original:
text coverage, table fidelity, figure placement, link preservation, heading
structure, and more. The `verify_report.md` shows what passed and what needs
attention.

**Phase 4 — Iterative repair loop (`--postfix N`, default 3).** Each iteration
runs cheapest-first: deterministic fixes (missing tables, links, code blocks,
headings — €0), then LLM missing-text rescue on pages still flagged, then a
vision model that compares each still-flagged source page against the output
and applies exact text patches. After each iteration the document is
re-verified; the loop repeats while coverage keeps improving, and stops on a
clean verify, no improvement, or the iteration budget — whichever comes first.
Per-iteration cost and coverage deltas are saved in `repair_report.md`.
Disable with `--no-postfix`.

**Phase 5.5 — Strip chrome (`--strip-chrome`).** Optionally removes running
headers, footers, and page numbers from the final output — useful when the
target is a web page rather than a PDF replica. Deterministic (regex-based,
zero LLM cost). Runs after all repair and verification steps so the quality
checks always see the complete 1:1 document.

### Pipeline summary

| Step | What | When |
|------|------|------|
| Detect | 1:1 page images, figure crops, chrome preserved | Always |
| Convert | LLM transcribes text + places figures | Always |
| Verify | Mechanical fidelity checks | Always for .qmd |
| Repair loop | Deterministic fixes → LLM text rescue → vision patches, iterated | `--postfix N` (default 3; `--no-postfix` to disable) |
| Strip chrome | Removes running headers/footers/page numbers | `--strip-chrome` |

## Output files

Each converted document produces:

- `<stem>.qmd` — the converted document (or `.md`/`.gfm`, see `--format`)
- `<stem>-media/` — figures extracted from the source PDF
- `verify_report.md` — results of the fidelity checks
- `repair_report.md` — per-iteration repair loop history (fixes, cost, coverage)
- `result.json` — machine-readable summary (cost, figures, verify status)

## Configuration

API key and default model are stored in `~/.pdf2md/`:

- `key` — OpenRouter API key (mode 600)
- `config.json` — model selection + auto-cached model limits from OpenRouter API

```bash
python3 pdf2md.py --setup    # Interactive configuration
```

## Model Selection

Not all models work equally well for PDF conversion. Before switching models, run
the bundled edge-case benchmark to see how a candidate model handles tricky content:

```bash
python3 pdf2md.py src/pdf2md/tests/fixtures/pdf2md_edgecases.pdf \
    --out /tmp/benchmark-out/ \
    --model <model-slug> \
    --postfix 0
```

Then check `verify_report.md` — the document is designed to surface failures in:

- Table detection: text-based tables with merged cells, plus rasterized PNG tables
- Figure extraction: embedded charts, wrapped figures, subfigures
- Text fidelity: soft hyphens, ligatures, Unicode, intra-word formatting changes
- Structure: nested lists, definition lists, blockquotes, task lists, two-column layout
- Scientific content: chemical formulas, subscripts/superscripts, display equations
- Metadata: YAML frontmatter generation (title detection, date parsing)

Text-only models fail the figure placement check, so stick to multimodal models.

## Known model pitfalls

The tool checks the chosen model against OpenRouter's metadata before spending
anything (file input support, context window vs. document size) and prints a
`model check` box when something won't work. Two pitfalls it can only warn about:

- **Copyright / recitation guardrails.** Some models (seen with gemini-pro
  variants) refuse to reproduce text from published documents. The failure is
  confusing: no clear error, just near-zero coverage or empty responses while
  tokens are still billed. If a strong model produces inexplicably bad coverage
  on a published PDF, suspect this first and switch models —
  `google/gemini-2.5-flash` has not shown this behaviour.
- **Reasoning models burning the budget.** Thinking models (gemini-pro, gpt-5)
  can spend thousands of completion tokens on hidden reasoning and return little
  or no text at modest `max_tokens`. The error message says so when it happens.

## Authors & license

pdf2md was extracted from [CLMS_documents](https://github.com/MatMatt/CLMS_documents).

- Maciej Dudek
- Matteo Mattiuzzi

Copyright © 2026 European Union. Licensed under [EUPL-1.2](LICENSE).