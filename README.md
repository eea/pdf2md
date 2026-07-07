# pdf2md

Standalone multi-format PDF converter — detect figures and tables, then convert to Quarto `.qmd`, Markdown `.md`, or GitHub-Flavored Markdown `.gfm`.

Extracted from [CLMS_documents](https://github.com/MatMatt/CLMS_documents).

## Authors

- Maciej Dudek
- Matteo Mattiuzzi

Copyright © 2026 European Union. Licensed under EUPL-1.2.

## Quick start

```bash
# Install
pip install -e .

# One-time setup (API key + default model)
python3 pdf2md.py --setup

# Convert a PDF
python3 pdf2md.py document.pdf

# Batch convert a directory
python3 pdf2md.py inbox/ --out output/

# Output format
python3 pdf2md.py document.pdf --format gfm    # GitHub-Flavored Markdown
python3 pdf2md.py document.pdf --format md     # Plain Markdown
python3 pdf2md.py document.pdf --render        # Also render to PDF via Quarto

# Use a YAML frontmatter template (only with --format qmd)
python3 pdf2md.py document.pdf --template path/to/template.qmd
python3 pdf2md.py document.pdf --template https://raw.githubusercontent.com/org/repo/main/template.qmd
```

When --template is used, the template's YAML frontmatter block is injected into the
conversion prompt. The LLM fills in document-specific values (title, date, etc.)
while preserving the template's field set, order, and structure. This ensures
every converted document starts with a consistent, pre-defined header.

## Pipeline

1. **Phase 1 — Detect:** Extract figures/tables, strip chrome, detect cover metadata
2. **Phase 2 — Convert:** LLM transforms the placeholdered PDF to structured markdown
3. **Phase 2.5 — Rescue:** Deterministically resolve leftover figure tokens + LLM-driven insertion of unreferenced figures
4. **Phase 3 — Tablefix:** Deterministic table width, caption, and orientation fixes
5. **Phase 4 — Verify:** Content-fidelity check (text coverage, figure placement, table coverage)

## Configuration

API key and default model are stored in `~/.pdf2md/`:
- `key` — OpenRouter API key (mode 600)
- `config.json` — model selection + auto-cached model limits from OpenRouter API

```bash
python3 pdf2md.py --setup    # Interactive configuration
```

## License

EUPL-1.2 — see [LICENSE](LICENSE).
