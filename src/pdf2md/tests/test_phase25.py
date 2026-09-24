import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pdf2md.phase25 import _place_figures_by_caption, _append_unplaced_figures  # noqa: E402


def test_places_figure_at_its_caption():
    qmd = ("# Results\n\nSome analysis text here.\n\n"
           "Figure 10. Comparison of representative vegetation phenology parameters "
           "across sites.\n\nMore discussion follows.\n")
    fig = {"fig_id": "FIG_14", "file": "img-abc.png",
           "caption": "Figure 10. Comparison of representative vegetation phenology "
                      "parameters across sites."}
    out, placed = _place_figures_by_caption(qmd, [fig], "doc-media")
    assert placed == ["FIG_14"]
    assert "![Figure 10. Comparison of representative vegetation phenology parameters " \
           "across sites.](doc-media/img-abc.png)" in out
    assert out.count("Figure 10. Comparison") == 1          # no duplication


def test_cross_reference_not_mistaken_for_caption():
    # a "see Figure 10" sentence must NOT be turned into the image
    qmd = "# Intro\n\nAs shown in Figure 10 the trend is clear.\n\nEnd.\n"
    fig = {"fig_id": "FIG_14", "file": "img-abc.png",
           "caption": "Figure 10. Comparison of representative vegetation phenology "
                      "parameters across many different european sites and seasons."}
    out, placed = _place_figures_by_caption(qmd, [fig], "doc-media")
    assert placed == [] and "img-abc.png" not in out


def test_append_fallback_when_no_caption_in_body():
    qmd = "# Doc\n\nBody with no figure caption at all.\n"
    fig = {"fig_id": "FIG_9", "file": "img-z.png", "caption": "Figure 9. A plot."}
    out, n = _append_unplaced_figures(qmd, [fig], "doc-media")
    assert n == 1 and out.rstrip().endswith("![Figure 9. A plot.](doc-media/img-z.png)")


def test_bare_token_in_a_link_target_resolves_to_the_path_only():
    """A multi-line alt text stops the image-shaped pattern matching, so the token
    reaches the bare-token pass still inside "](...)". Substituting a whole image
    there nests one image inside another's target and Typst fails on the url-encoded
    filename (PA21: "La Albufera" figure)."""
    from pdf2md.phase25 import resolve_leftover_fig_tokens as _resolve_bare_fig_tokens
    figs = [{"fig_id": "FIG_1", "file": "img-abc.jpeg", "caption": "Schematic view."}]
    body = ("![Coastal lagoon\nSand bank\n"
            "Schematic view of La Albufera coastal lagoon (Valencia, Spain).](FIG_1)\n")
    out, n = _resolve_bare_fig_tokens(body, figs, "media", "doc")
    assert n == 1
    assert "](media/img-abc.jpeg)" in out
    assert "](![" not in out                      # no nesting


def test_bare_token_on_its_own_still_becomes_an_image():
    from pdf2md.phase25 import resolve_leftover_fig_tokens as _resolve_bare_fig_tokens
    figs = [{"fig_id": "FIG_2", "file": "img-b.jpeg", "caption": "A plot."}]
    out, n = _resolve_bare_fig_tokens("Text before.\n\nFIG_2\n\nAfter.\n", figs, "media", "doc")
    assert n == 1 and "![A plot.](media/img-b.jpeg)" in out
