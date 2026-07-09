"""CLI for the production two-pass pdf2md flow (detect, convert, render, verify).

Accepts a single PDF or a directory (batch); writes output/<doc>/… per document.
The production entry, distinct from the legacy single-pass `cli.py`.

    python3 tools/pdf2md/pdf2md.py FILE.pdf
    python3 tools/pdf2md/pdf2md.py inbox/                 # batch: every *.pdf in inbox/
    python3 tools/pdf2md/pdf2md.py FILE.pdf
    python3 tools/pdf2md/pdf2md.py inbox/ --out output --model google/gemini-2.5-pro

Environment:
    OPENROUTER_API_KEY   (required)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .app import DEFAULT_MODEL, Events, convert_batch, convert_one
from .cost import eur_to_usd, fmt_eur
from .cover import DEFAULT_COVER_MODEL
from . import __version__
from .ui import make_ui

log = logging.getLogger(__name__)


def _build_json_report(result, timing, model, cover_model):
    """Write a comprehensive machine-readable report alongside result.json."""
    import json, time as _time
    report = {
        "version": __version__,
        "generated": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "model": model,
        "cover_model": cover_model,
        "file": {
            "pdf": str(result.pdf),
            "stem": result.stem,
            "pages": result.est.get("pages") if result.est else None,
        },
        "status": result.status,
        "timing": timing,
        "cost": {
            "total_usd": result.cost_usd,
            "phases": result.phase_cost,
        },
        "verify": {
            "status": result.verify_status,
            "text_coverage": result.text_cov,
            "table_coverage": result.table_cov,
            "issues": result.verify_issues,
        },
        "figures": result.figures,
        "tables": result.tables,
        "postfix": {
            "items_recovered": result.postfix_items,
            "applied": result.postfixes_applied,
        },
        "tablefix": result.tablefix,
    }
    if result.est:
        report["estimate"] = {
            "expected_usd": result.est.get("expected_usd"),
            "low_usd": result.est.get("low_usd"),
            "high_usd": result.est.get("high_usd"),
            "candidate_pages": result.est.get("candidate_pages"),
        }
    if result.error:
        report["error"] = result.error
    return report

# ── Key & config helpers ───────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".pdf2md"
CONFIG_FILE = CONFIG_DIR / "config.json"
KEY_FILE = CONFIG_DIR / "key"


def resolve_key() -> str:
    """Resolve OpenRouter API key: env var -> key file -> error.
    Ignores env vars that don't look like real OpenRouter keys (e.g. '***' redactions)."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key and key.startswith("sk-or-"):
        return key
    key_file = os.environ.get("OPENROUTER_API_KEY_FILE", "")
    if key_file:
        p = Path(key_file)
        if p.exists():
            key = p.read_text(encoding="utf-8").strip()
            if key and key.startswith("sk-or-"):
                return key
    if KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
        if key and key.startswith("sk-or-"):
            return key
    return ""


def resolve_model(args_model=None):
    """Resolve model: CLI arg -> env var -> config file -> default."""
    if args_model:
        return args_model
    env_model = os.environ.get("OPENROUTER_MODEL", "").strip()
    if env_model:
        return env_model
    if CONFIG_FILE.exists():
        try:
            import json as _json
            cfg = _json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            m = cfg.get("model", "").strip()
            if m:
                return m
        except Exception:
            pass
    return DEFAULT_MODEL


def run_setup() -> int:
    """Interactive setup: API key + default model, saved to ~/.pdf2md/."""
    import json
    print("pdf2md — one-time setup\n")
    print("Paste your OpenRouter API key (or press Enter to skip):")
    key = input("> ").strip()
    if key:
        if not key.startswith("sk-or-"):
            print("  Error: key must start with 'sk-or-' (OpenRouter API key format).")
            return 1
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_text(key, encoding="utf-8")
        KEY_FILE.chmod(0o600)
        print(f"  Key saved to {KEY_FILE} (permissions 600)\n")
    else:
        print("  (skipped — set OPENROUTER_API_KEY env var to use pdf2md)\n")

    models = [
        ("google/gemini-2.5-pro",   "Best quality, slower (~EUR 0.15-0.60/doc)"),
        ("google/gemini-2.5-flash", "Fast, cheap (~EUR 0.02-0.10/doc)"),
        ("google/gemini-3.5-flash", "Newest flash model"),
    ]
    print("Pick a default model:")
    for i, (m, desc) in enumerate(models, 1):
        print(f"  {i}. {m}  -- {desc}")
    print(f"  {len(models)+1}. {DEFAULT_MODEL} (default)")
    choice = input(f"[1-{len(models)+1}, Enter=default]> ").strip()
    try:
        idx = int(choice) - 1
        model = models[idx][0] if 0 <= idx < len(models) else DEFAULT_MODEL
    except (ValueError, IndexError):
        model = DEFAULT_MODEL
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"model": model}
    
    # Quarto auto-detection (for --render)
    import shutil as _shutil
    quarto = _shutil.which("quarto")
    if quarto:
        print(f"\nQuarto found at: {quarto} (optional, only needed for --render)")
        print("Press Enter to accept, or type a different path (Enter to skip):")
        alt = input("> ").strip()
        if alt:
            quarto = alt
    else:
        print("\nQuarto not found in PATH (optional, only needed for --render).")
        print("Enter path to quarto binary, or press Enter to skip:")
        quarto = input("> ").strip() or None
    if quarto:
        cfg["quarto_path"] = quarto
    
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"  Default model set to: {model}")
    if quarto:
        print(f"  Quarto path: {quarto}")
    print(f"  (override with --model or OPENROUTER_MODEL env var)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf2md",
        description="Convert PDF(s) to Quarto .qmd (detect → convert → render → verify).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", nargs="?", type=Path, help="a PDF file, or a directory of PDFs (batch)")
    p.add_argument("--version", "-V", action="version", version=f"pdf2md {__version__}")
    p.add_argument("--format", "-f", default="qmd", choices=["qmd", "md", "gfm"], help="output format (default: qmd)")
    p.add_argument("--out", type=Path, default=Path("output"),
                   help="output root directory (default: output/)")
    p.add_argument("--model", default=None,
                   help=f"OpenRouter model for detect+convert (default: env OPENROUTER_MODEL or {DEFAULT_MODEL})")
    p.add_argument("--cover-model", default=DEFAULT_COVER_MODEL,
                   help=f"model for cover-metadata extraction (default: {DEFAULT_COVER_MODEL})")
    p.add_argument("--template", type=str, default=None, metavar="TEMPLATE",
                   help="path or URL to a .qmd template file; its YAML frontmatter is injected into the conversion prompt (with --format qmd or gfm)")
    p.add_argument("--render", action="store_true", help="render .qmd to PDF via Quarto/Typst")
    p.add_argument("--no-verify", action="store_true", help="skip the content-fidelity verify pass")
    p.add_argument("--json-report", action="store_true", help="write a comprehensive machine-readable <stem>-report.json alongside the output")
    p.add_argument("--postfix", type=int, default=1, metavar="N", help="post-conversion fixes after verify (default: 1, 0 to disable)")
    p.add_argument("--improve", action="store_true", help="skip conversion, only re-verify and postfix existing output")
    p.add_argument("--force", action="store_true", help="overwrite existing output/<doc>/")
    p.add_argument("--max-cost-per-file", type=float, default=None, metavar="EUR",
                   help="skip a file whose pre-flight estimate exceeds this (EUR); "
                        "no per-file gate if unset")
    p.add_argument("--max-cost-total", type=float, default=None, metavar="EUR",
                   help="batch backstop (EUR): stop before a file that would push "
                        "cumulative actual spend past this")
    p.add_argument("--allow-over-budget", action="store_true",
                   help="convert regardless of the cost estimate (override both gates)")
    p.add_argument("--dry-run", action="store_true",
                   help="replay the UI from an existing output dir (no LLM calls, no cost); "
                        "pass the output root (or a single output/<doc>/) as the path")
    p.add_argument("--delay", type=float, default=None,
                   help="per-step pause in dry-run replay (seconds; default 0.12)")
    p.add_argument("--detect-workers", type=int, default=8, metavar="N",
                   help="concurrent per-page figure-detection LLM calls in Phase 1 "
                        "(default 8; tuned to stay under Gemini rate limits — see README. "
                        "Use 1 for sequential)")
    p.add_argument("--quiet", action="store_true", help="plain logging output (no rich UI)")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    p.add_argument("--keep-headers", action="store_true", default=False,
                   help="keep running headers/footers (only useful with --format qmd)")
    p.add_argument("--setup", action="store_true", help="interactive setup: configure API key and default model")
    return p


def _plain_summary(results: list) -> int:
    """Plain end-of-run report (the rich UI replaces this when active).
    Returns an exit code (0 = no failures)."""
    resumed = [r for r in results if r.resumed]
    ok = [r for r in results if r.status == "ok" and not r.resumed]
    warn = [r for r in results if r.status == "warn"]
    fail = [r for r in results if r.status == "fail"]
    skip = [r for r in results if r.status == "skip"]
    print("\n" + "=" * 70)
    print("pdf2md — conversion summary")
    print("=" * 70)
    total_usd = 0.0
    for r in results:
        icon = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "skip": "[skip]"}.get(r.status, "[????]")
        if r.resumed:
            icon = "[done]"
        total_usd += r.cost_usd
        if r.status == "skip" or r.resumed:
            line = f"  {icon} {r.stem:<46} {r.error}"
        else:
            line = f"  {icon} {r.stem:<46} {r.figures} figures"
            if r.verify_status:
                line += f"  verify={r.verify_status}"
            if r.cost_usd:
                line += f"  {fmt_eur(r.cost_usd)}"
            if r.error:
                line += f"  ({r.error})"
        print(line)
    print("-" * 70)
    out_root = results[0].out_dir.parent if results else "?"
    tail = f"  {len(ok)} ok · {len(warn)} warn · {len(fail)} fail"
    if skip:
        tail += f" · {len(skip)} skip"
    if resumed:
        tail += f" · {len(resumed)} already done"
    tail += f"   ·   total {fmt_eur(total_usd)}   →  {out_root}"
    print(tail)
    print("=" * 70)
    return 1 if fail else 0


def _setup_ui_and_logging(args, batch):
    """Build the events sink and configure logging so log lines never fight the
    live rich display. With the rich UI on, logs go through the *same* console
    (Rich prints them above the live region; a separate stderr handler would
    desync the cursor and stack the active section); level drops to WARNING so the
    UI isn't buried in INFO chatter. Otherwise: plain timestamped logging."""
    console = None
    if not (args.quiet or args.verbose):
        try:
            from rich.console import Console
            console = Console()
        except ImportError:
            console = None
    events = make_ui(batch=batch, console=console, force=False) if console else Events()
    rich_active = type(events).__name__ != "Events"

    if rich_active:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=logging.WARNING, format="%(message)s", datefmt="%H:%M:%S",
            handlers=[RichHandler(console=console, show_path=False, markup=False,
                                  rich_tracebacks=False)],
            force=True)
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
            force=True)
    return events, rich_active


def _print_cancelled(events, out_root, rich_active) -> None:
    """Friendly Ctrl+C notice: what's saved and how to resume."""
    body = (f"Finished documents are saved under [b]{out_root}[/] and will be "
            f"skipped on resume.\nResume:  re-run the same command   "
            f"[dim](add --force to redo a document)[/]")
    if rich_active and hasattr(events, "con"):
        from rich.panel import Panel
        from rich.text import Text
        events.con.print()
        events.con.print(Panel(Text.from_markup(body), title="[b yellow]⚠ Cancelled[/]",
                               border_style="yellow", expand=False, padding=(0, 2)))
        events.con.print()
    else:
        log.warning("cancelled — finished documents under %s are skipped on resume; "
                    "re-run the same command (add --force to redo a document)", out_root)


def _dry_run(args) -> int:
    """Replay the UI from existing output artifacts (no LLM calls, no cost).
    `path` is the output root (batch) or a single `output/<doc>/` dir."""
    from .replay import DEFAULT_DELAY, replay_batch, replay_mock_batch, replay_one

    delay = args.delay if args.delay is not None else DEFAULT_DELAY
    try:
        single_dir = (args.path / f"{args.path.name}.qmd").exists()
        # no recorded outputs but the dir holds raw PDFs -> show MOCK data so the UX
        # can be previewed on un-converted PDFs (fabricated numbers, no LLM)
        has_outputs = single_dir or any(
            d.is_dir() and (d / f"{d.name}.qmd").exists() for d in args.path.iterdir())
    except NotADirectoryError:
        log.error("--dry-run: %s is a file, not a directory (re-run without --dry-run on a converted output directory)", args.path)
        return 1
    mock = not has_outputs and any(args.path.glob("*.pdf"))
    batch = not single_dir
    events, rich_active = _setup_ui_and_logging(args, batch)

    try:
        if mock:
            log.warning("no recorded runs under %s — showing MOCK data from %d PDF(s) "
                        "(no LLM, fabricated numbers)", args.path, len(list(args.path.glob("*.pdf"))))
            results = replay_mock_batch(args.path, events, delay=delay)
        elif single_dir:
            results = [replay_one(args.path, events, delay=delay)]
        else:
            results = replay_batch(args.path, events, delay=delay)
    except KeyboardInterrupt:
        events.abort()
        log.warning("cancelled")
        return 130

    if not results:
        log.error("no replayable output dirs under %s (need a converted <doc>/<doc>.qmd)", args.path)
        return 1
    if rich_active:
        return 1 if any(r.status == "fail" for r in results) else 0
    return _plain_summary(results)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
    )

    if args.setup:
        return run_setup()

    if args.path is None:
        parser.print_help()
        print("\nSupply a PDF file or directory, or use --setup to configure.")
        return 1

    if not args.path.exists():
        log.error("path not found: %s", args.path)
        return 1

    if args.dry_run:
        return _dry_run(args)

    api_key = resolve_key()
    if not api_key:
        log.warning("No API key configured — starting interactive setup.")
        run_setup()
        api_key = resolve_key()
        if not api_key:
            log.error("No API key provided. Set OPENROUTER_API_KEY or run 'pdf2md --setup'.")
            return 1
    model = resolve_model(args.model)

    batch = args.path.is_dir()
    events, rich_active = _setup_ui_and_logging(args, batch)

    common = dict(
        api_key=api_key, model=model, cover_model=args.cover_model,
        do_render=args.render, do_verify=not args.no_verify, force=args.force,
        format=args.format, strip_headers=(not args.keep_headers),
        postfix_passes=args.postfix,
        improve_only=args.improve,
        max_cost_per_file=eur_to_usd(args.max_cost_per_file),
        allow_over_budget=args.allow_over_budget,
        events=events,
        detect_workers=args.detect_workers,
        template=str(args.template) if args.template else None,
        json_report=args.json_report,
    )

    try:
        if batch:
            results = convert_batch(args.path, args.out,
                                    max_cost_total=eur_to_usd(args.max_cost_total), **common)
        else:
            results = [convert_one(args.path, args.out, **common)]

        # Write json reports
        if args.json_report:
            import json as _json
            for r in results:
                if r.status in ("ok", "warn") and hasattr(r, "timing"):
                    report = _build_json_report(r, r.timing, model, args.cover_model)
                    report_path = r.out_dir / f"{r.stem}-report.json"
                    report_path.write_text(_json.dumps(report, indent=2, default=str), encoding="utf-8")
                    log.info("Wrote json report: %s", report_path)
    except KeyboardInterrupt:
        events.abort()
        _print_cancelled(events, args.out, rich_active)
        return 130

    # The rich UI already rendered its own summary/aggregate panel.
    if rich_active:
        return 1 if any(r.status == "fail" for r in results) else 0
    return _plain_summary(results)


if __name__ == "__main__":
    sys.exit(main())