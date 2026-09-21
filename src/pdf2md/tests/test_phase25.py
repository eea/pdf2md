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
