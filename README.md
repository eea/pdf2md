# pdf2md

Standalone multi-format PDF converter — detect figures and tables, then convert to Quarto `.qmd`, Markdown `.md`, or GitHub-Flavored Markdown `.gfm`.


## Authors

- Maciej Dudek
- Matteo Mattiuzzi

Copyright © 2026 European Environment Agency. Licensed under EUPL-1.2.

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
```

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
