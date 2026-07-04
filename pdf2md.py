#!/usr/bin/env python3
"""Launcher for the production two-pass pdf2md flow.

Single command for: detect → convert → render → verify, writing output/<doc>/…
Accepts one PDF or a directory of PDFs (batch).

    python3 tools/pdf2md/pdf2md.py FILE.pdf
    python3 tools/pdf2md/pdf2md.py inbox/
    python3 tools/pdf2md/pdf2md.py FILE.pdf --no-render --no-verify

This is the production entry for the validated pipeline. The legacy single-pass
converter lives in scripts/convert_pdf.py.

Environment:
    OPENROUTER_API_KEY   (required)
    OPENROUTER_MODEL     (optional; overridden by --model)
"""

import sys
from pathlib import Path

# Put src/ on the path so `pdf2md` resolves as a top-level package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pdf2md.app_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())