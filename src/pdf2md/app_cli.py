"""CLI for the production two-pass pdf2md flow (detect, convert, render, verify).

Accepts a single PDF or a directory (batch); writes output/<doc>/… per document.
The production entry, distinct from the legacy single-pass `cli.py`.

    python3 tools/pdf2md/pdf2md.py FILE.pdf
    python3 tools/pdf2md/pdf2md.py inbox/                 # batch: every *.pdf in inbox/
    python3 tools/pdf2md/pdf2md.py FILE.pdf
    python3 tools/pdf2md/pdf2md.py inbox/ --out output    # batch to a chosen dir

Environment:
    OPENROUTER_API_KEY   (required)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from .app import CONFIG_DIR, CONFIG_FILE, DEFAULT_MODEL, Events, convert_batch, convert_one
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
            "iterations": result.repair_iterations,
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

KEY_FILE = CONFIG_DIR / "key"  # CONFIG_DIR/CONFIG_FILE are the canonical defs in app


def resolve_key() -> str:
    """Resolve OpenRouter API key: env var -> key file -> shell profile -> error.
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
    # Fallback: check .bashrc / .profile for a raw export
    for rc_path in [Path.home() / ".bashrc", Path.home() / ".profile",
                    Path.home() / ".bash_profile"]:
        if rc_path.exists():
            try:
                for line in rc_path.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("export OPENROUTER_API_KEY="):
                        val = stripped.split("=", 1)[1].strip().strip("\"'")
                        if val.startswith("sk-or-") and "..." not in val:
                            return val
            except Exception:
                continue
    return ""


def describe_key_sources() -> str:
    """Explain where we looked for a key and what we found, for the no-key error."""
    parts = []
    env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not env:
        parts.append("OPENROUTER_API_KEY is not exported in this shell")
    elif not env.startswith("sk-or-"):
        parts.append(f"OPENROUTER_API_KEY is set but doesn't look like an OpenRouter "
                     f"key (starts with {env[:6]!r}, expected 'sk-or-')")
    kf = os.environ.get("OPENROUTER_API_KEY_FILE", "")
    if kf and not Path(kf).exists():
        parts.append(f"OPENROUTER_API_KEY_FILE points to a missing file ({kf})")
    if not KEY_FILE.exists():
        parts.append(f"no saved key at {KEY_FILE}")
    return "; ".join(parts) or "key sources look fine"


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


def resolve_figure_llm(args_figure_llm=None, main_model=None):
    """Resolve figure LLM: CLI arg -> config file -> main model (fallback)."""
    if args_figure_llm:
        return args_figure_llm
    if CONFIG_FILE.exists():
        try:
            import json as _json
            cfg = _json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            m = cfg.get("figure_llm", "").strip()
            if m:
                return m
        except Exception:
            pass
    return main_model  # fallback: use the main conversion model


def _fetch_models(api_key: str) -> list[dict]:
    """Fetch available models from OpenRouter API. Returns [] on failure."""
    import urllib.request, json
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("data", [])
    except Exception:
        return []


def _select_llm(prompt: str, api_key: str, default: str = "",
                filter_fn=None) -> str:
    """Interactive LLM picker using live OpenRouter model catalogue.

    Args:
        prompt:  question shown before the list
        api_key: for API call
        default: fallback slug if fetch fails or user enters nothing
        filter_fn: callable(model_dict) -> bool, None = keep all

    Returns chosen model slug (str).
    """
    models = _fetch_models(api_key)
    if filter_fn:
        models = [m for m in models if filter_fn(m)]
    # Sort: multimodal first, then by prompt price ascending
    models.sort(key=lambda m: (
        0 if (m.get("architecture") or {}).get("modality", "").startswith("text+image")
        else 1,
        float((m.get("pricing") or {}).get("prompt", 999)),
    ))
    # Cap at 25 to avoid overwhelming
    top = models[:25]

    print(f"\n{prompt}")
    if top:
        print()
        for i, m in enumerate(top, 1):
            slug = m.get("id", "?")
            p = (m.get("pricing") or {})
            cost = float(p.get("prompt", 0)) * 1e3
            ctx = m.get("context_length", "?")
            mm = "🖼️" if ((m.get("architecture") or {}).get("modality", "").startswith("text+image")) else "  "
            print(f"  [{i:2d}] {mm} {slug}  (€{cost:.1f}/1M prompt  |  {ctx:,} ctx)")
        print(f"  [{len(top)+1:2d}] type custom slug")
    else:
        models = []
        print("  (could not fetch model list from OpenRouter)")

    print("Enter number or slug [default: %s]: " % (default or "(none)"), end="")
    ans = input("> ").strip()
    if not ans:
        return default
    # Check if it's a number from the list
    try:
        idx = int(ans) - 1
        if 0 <= idx < len(top):
            return top[idx].get("id", default)
    except ValueError:
        pass
    return ans  # treat as custom slug


def run_setup() -> int:
    """Interactive setup: API key (+ optional Quarto path), saved to ~/.pdf2md/."""
    import json
    print("pdf2md — one-time setup\n")
    existing = resolve_key()
    if existing:
        print(f"Valid OpenRouter API key found at {KEY_FILE}")
        ans = input("Replace? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            key = existing
            print("  (keeping existing key)\n")
        else:
            key = ""
    else:
        key = ""

    if not key:
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
            print("  (no key — set OPENROUTER_API_KEY env var to use pdf2md)\n")

    cfg = {}

    # Key is known now — use it to fetch models
    ak = key or resolve_key()

    main = _select_llm(
        "Select conversion LLM:",
        ak, default=DEFAULT_MODEL,
    )
    cfg["model"] = main
    print(f"  Conversion LLM: {main}\n")

    ans = input("Use a cheaper figure-detection LLM? [y/N] ").strip().lower()
    if ans in ("y", "yes"):
        fig = _select_llm(
            "Select figure-detection LLM (Phase 1):",
            ak, default="google/gemini-2.5-flash",
        )
        cfg["figure_llm"] = fig
        print(f"  Figure LLM: {fig}\n")

    # Quarto auto-detection (needed for PDF output, which is on by default)
    import shutil as _shutil
    quarto = _shutil.which("quarto")
    if quarto:
        print(f"\nQuarto found at: {quarto} (needed for PDF output; --no-render to skip)")
        print("Press Enter to accept, or type a different path (Enter to skip):")
        alt = input("> ").strip()
        if alt:
            quarto = alt
    else:
        print("\nQuarto not found in PATH (needed for PDF output; --no-render to skip).")
        print("Enter path to quarto binary, or press Enter to skip:")
        quarto = input("> ").strip() or None
    if quarto:
        cfg["quarto_path"] = quarto

    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    dm = cfg.get("figure_llm", cfg.get("model", DEFAULT_MODEL))
    print(f"  Conversion LLM: {cfg.get('model', DEFAULT_MODEL)} (set via --main-llm)")
    print(f"  Figure LLM:     {dm}" + (" (same as conversion)" if dm == cfg.get("model", DEFAULT_MODEL) else " (cheaper — Phase 1 only)"))
    if quarto:
        print(f"  Quarto path: {quarto}")
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
    p.add_argument("--main-llm", default=None,
                   help=f"conversion LLM (default: env OPENROUTER_MODEL, config 'model', or {DEFAULT_MODEL})")
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)  # deprecated alias for --main-llm
    p.add_argument("--figure-llm", default=None,
                   help="LLM for Phase 1 figure detection (default: same as --main-llm, "
                        "or 'figure_llm' from config)")
    p.add_argument("--cover-model", default=DEFAULT_COVER_MODEL,
                   help=f"model for cover-metadata extraction (default: {DEFAULT_COVER_MODEL})")
    p.add_argument("--template", type=str, default=None, metavar="TEMPLATE",
                   help="path or URL to a .qmd template file; its YAML frontmatter is injected into the conversion prompt (with --format qmd or gfm)")
    p.add_argument("--no-render", action="store_true",
                   help="skip PDF output (default: produce .qmd + verbatim PDF)")
    p.add_argument("--no-verify", action="store_true", help="skip the content-fidelity verify pass")
    p.add_argument("--json-report", action="store_true", help="write a comprehensive machine-readable <stem>-report.json alongside the output")
    p.add_argument("--postfix", type=int, default=3, metavar="N",
                   help="max iterations of the post-conversion repair loop "
                        "(deterministic fixes → LLM missing-text → vision patches → "
                        "re-verify; default 3)")
    p.add_argument("--no-postfix", action="store_const", const=0, dest="postfix",
                   help=argparse.SUPPRESS)  # deprecated
    p.add_argument("--no-review", action="store_const", const=0, dest="postfix",
                   help="disable the post-conversion review/repair loop")
    p.add_argument("--improve", action="store_true", help="skip conversion, only re-verify and run the repair loop on existing output")
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
    p.add_argument("--strip-chrome", action="store_true", default=False,
                   help="after conversion, strip running headers/footers/page numbers "
                        "from the output")
    # deprecated alias: 1:1 (headers kept) is now the default, so this is a no-op
    # kept only so existing invocations don't break
    p.add_argument("--keep-headers", action="store_true", default=False,
                   help=argparse.SUPPRESS)
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
        log.warning("No usable API key found: %s.", describe_key_sources())
        log.warning("Starting interactive setup — or export OPENROUTER_API_KEY and re-run.")
        run_setup()
        api_key = resolve_key()
        if not api_key:
            log.error("No API key provided (%s). Set OPENROUTER_API_KEY or run "
                      "'pdf2md --setup'.", describe_key_sources())
            return 1
    model = resolve_model(args.main_llm or args.model)
    figure_llm = resolve_figure_llm(args.figure_llm, main_model=model)

    if args.keep_headers:
        log.info("--keep-headers is deprecated: headers are kept by default now "
                 "(use --strip-chrome to remove them)")

    batch = args.path.is_dir()
    events, rich_active = _setup_ui_and_logging(args, batch)

    common = dict(
        api_key=api_key, model=model, figure_llm=figure_llm, cover_model=args.cover_model,
        do_render=not args.no_render, do_verify=not args.no_verify, force=args.force,
        format=args.format, strip_chrome=args.strip_chrome,
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