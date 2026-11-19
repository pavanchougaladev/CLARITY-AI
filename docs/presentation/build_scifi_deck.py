"""Build the 15-slide CLARITY-3D "Mission Briefing" deck from measured results.

The deck is rebuilt from the pristine 12-slide base every time, so the script is
idempotent. Slide 9 (the old fabricated "ours vs 9 baselines" chart) is rebuilt
with the real 7-matcher benchmark, and three new evidence slides are inserted
after it:

     9  BENCHMARK: 7 MATCHERS x 5 LUNAR PAIRS
    10  LIVE RUN: 358 INLIERS, GRADE A
    11  3D TERRAIN & LANDING HAZARD AI
    12  CRATER AI & VERIFICATION SUITE

Every number comes from a file in this repository:

    scripts/pavan_benchmark_results.json            (matcher benchmark)
    outputs/presentation_sift/quality.json          (UI + CLI certification)
    runs/detect/runs/train_lunar/.../results.csv    (YOLO11n metrics)

Run:  .venv\\Scripts\\python.exe docs\\presentation\\build_scifi_deck.py
"""
from __future__ import annotations

import copy
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DECK = os.path.join(ROOT, "CLARITY-3D_Presentation_SciFi Final.pptx")
BASE_CANDIDATES = [
    os.path.join(HERE, "deck_base.pptx"),
    os.path.join(os.environ.get("TEMP", "."), "opencode", "base_deck.pptx"),
]

CARD_FILL = "0A0D12"
AMBER = "E8A33D"
BLUE = "8FB8CC"
SAGE = "7FB59A"
MUTED = "8A939C"
LINE = "2B3440"
TEXT = "EDEDED"

R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

TOTAL_PAGES = 15
TITLE_FONT = "Century Gothic"
BODY_FONT = "Calibri"

CHROME_NAMES = (
    {"Shape 38", "Shape 39", "Shape 43"}
    | {f"Text {i}" for i in range(40, 54)}
)


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

def _rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def _drop(shape) -> None:
    shape._element.getparent().remove(shape._element)


def _is_chrome(shape) -> bool:
    if shape.name in CHROME_NAMES:
        return True
    return shape.width <= 90000 and shape.height <= 90000  # star field


def strip_to_chrome(slide) -> None:
    """Delete every content shape, keeping stars, header, divider and nav bar."""
    for shape in list(slide.shapes):
        if not _is_chrome(shape):
            _drop(shape)


def _remap_relationships(src_part, dst_part, element) -> None:
    """Point r:id / r:embed / r:link attributes at relationships of the new part."""
    for node in element.iter():
        for attr, old_rid in list(node.attrib.items()):
            if not attr.startswith(R_NS):
                continue
            rel = src_part.rels.get(old_rid)
            if rel is None:
                continue
            if rel.is_external:
                new_rid = dst_part.rels.get_or_add_ext_rel(rel.reltype, rel.target_ref)
            else:
                new_rid = dst_part.rels.get_or_add(rel.reltype, rel.target_part)
            node.attrib[attr] = new_rid


def clone_slide(prs, source):
    """Duplicate `source` (background, stars, chrome, links) as a new slide."""
    new_slide = prs.slides.add_slide(source.slide_layout)

    dst_tree = new_slide.shapes._spTree
    for child in list(dst_tree):
        if child.tag in (
            qn("p:sp"), qn("p:pic"), qn("p:graphicFrame"),
            qn("p:grpSp"), qn("p:cxnSp"),
        ):
            dst_tree.remove(child)

    src_tree = source.shapes._spTree
    for child in list(src_tree):
        if child.tag in (
            qn("p:sp"), qn("p:pic"), qn("p:graphicFrame"),
            qn("p:grpSp"), qn("p:cxnSp"),
        ):
            clone = copy.deepcopy(child)
            _remap_relationships(source.part, new_slide.part, clone)
            dst_tree.append(clone)

    src_cSld = source._element.find(qn("p:cSld"))
    dst_cSld = new_slide._element.find(qn("p:cSld"))
    src_bg = src_cSld.find(qn("p:bg"))
    if src_bg is not None:
        old_bg = dst_cSld.find(qn("p:bg"))
        if old_bg is not None:
            dst_cSld.remove(old_bg)
        dst_cSld.insert(0, copy.deepcopy(src_bg))

    return new_slide


def move_slide(prs, from_index: int, to_index: int) -> None:
    sld_ids = prs.slides._sldIdLst
    entry = list(sld_ids)[from_index]
    sld_ids.remove(entry)
    sld_ids.insert(to_index, entry)


def set_text(shape, text: str, *, size=None, color=None, bold=None,
             font=None, align=None, spacing=None) -> None:
    """Replace the whole text frame with one formatted run (keeps box geometry)."""
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0

    paragraphs = list(tf.paragraphs)
    first = paragraphs[0]
    for extra in paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    for run in list(first.runs):
        run._r.getparent().remove(run._r)

    run = first.add_run()
    run.text = text
    font_obj = run.font
    if size is not None:
        font_obj.size = Pt(size)
    if bold is not None:
        font_obj.bold = bold
    if color is not None:
        font_obj.color.rgb = _rgb(color)
    if font is not None:
        font_obj.name = font
    if align is not None:
        first.alignment = align
    if spacing is not None:
        run._r.get_or_add_rPr().set("spc", str(int(spacing * 100)))


def textbox(slide, x, y, w, h, text, *, size=10, color=TEXT, bold=False,
            font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            spacing=None, line=None):
    box = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = align
    if line is not None:
        para.line_spacing = line
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _rgb(color)
    run.font.name = font
    if spacing is not None:
        run._r.get_or_add_rPr().set("spc", str(int(spacing * 100)))
    return box


def panel(slide, x, y, w, h, *, border=LINE, fill=CARD_FILL, line_w=12700):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Emu(x), Emu(y), Emu(w), Emu(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(fill)
    shape.line.color.rgb = _rgb(border)
    shape.line.width = Emu(line_w)
    shape.shadow.inherit = False
    shape.text_frame.word_wrap = True
    return shape


def stat_card(slide, x, y, w, h, value, label, accent,
              value_pt=21, label_pt=9):
    panel(slide, x, y, w, h, border=accent)
    textbox(slide, x + 40000, y + 45000, w - 80000, 400000, value,
            size=value_pt, color=accent, bold=True, font=TITLE_FONT,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, spacing=1.5)
    textbox(slide, x + 55000, y + 440000, w - 110000, h - 470000, label,
            size=label_pt, color=MUTED, align=PP_ALIGN.CENTER,
            anchor=MSO_ANCHOR.TOP, line=0.95)


def fit_box(path, max_w, max_h):
    with Image.open(path) as img:
        aspect = img.width / img.height
    width = max_w
    height = int(round(width / aspect))
    if height > max_h:
        height = max_h
        width = int(round(height * aspect))
    return width, height


def framed_picture(slide, path, x, y, max_w, max_h):
    w, h = fit_box(path, max_w, max_h)
    inset = 12700
    panel(slide, x - inset, y - inset, w + 2 * inset, h + 2 * inset, border=LINE)
    slide.shapes.add_picture(path, Emu(x), Emu(y), Emu(w), Emu(h))
    return x, y, w, h


def set_header(slide, section, title):
    for shape in slide.shapes:
        if shape.name == "Text 41":
            set_text(shape, section, size=11, color=AMBER, bold=True, spacing=4)
        elif shape.name == "Text 42":
            set_text(shape, title, size=22, color=TEXT, bold=True,
                     font=TITLE_FONT, spacing=3)


# ---------------------------------------------------------------------------
# slide content
# ---------------------------------------------------------------------------

def build_benchmark(slide) -> None:
    """Slide 9 — real 7-matcher benchmark over five Chandrayaan pairs."""
    set_header(slide, "05 · WORK COMPLETED · MEASURED RESULTS",
               "BENCHMARK: 7 MATCHERS × 5 LUNAR PAIRS")

    chart = os.path.join(HERE, "scifi_benchmark.png")
    framed_picture(slide, chart, 457200, 1145000, 5200000, 3590000)

    x1, x2 = 5760720, 7269480
    y1, y2 = 1280160, 2331720
    w, h = 1417320, 914400
    stat_card(slide, x1, y1, w, h, "35 / 35", "engine runs succeeded", AMBER)
    stat_card(slide, x2, y1, w, h, "1.13 px",
              "best mean RMSE · SIFT & NI-LFM", AMBER)
    stat_card(slide, x1, y2, w, h, "864",
              "max mean inliers · LightGlue", BLUE)
    stat_card(slide, x2, y2, w, h, "0.51 s",
              "fastest pair runtime · SIFT", BLUE)

    panel(slide, 5760720, 3383280, 2926080, 1232040, border=SAGE)
    textbox(slide, 5897880, 3460000, 2651760, 1080000,
            "LunarFusion: 285 inliers at 1.93 px in 2.95 s. "
            "All seven engines completed all five pairs. "
            "Scores are means over the whole pair set — per-pair "
            "certification is on the next slide.",
            size=10, color=TEXT, line=0.98)


def build_live_run(slide) -> None:
    """Slide 10 — the certified dashboard run plus the full-resolution CLI run."""
    set_header(slide, "05 · WORK COMPLETED · LIVE VALIDATION",
               "LIVE RUN: 358 INLIERS, GRADE A")

    shot = os.path.join(HERE, "02_registration_gates.png")
    framed_picture(slide, shot, 457200, 1145000, 5100000, 3200000)

    x, w = 5760720, 2926080
    stat_card(slide, x, 1280160, w, 940000, "358",
              "verified inliers · 71.9% grid coverage", AMBER)
    stat_card(slide, x, 2290160, w, 940000, "0.434 px",
              "refined fit RMSE · UI certification run", AMBER)
    stat_card(slide, x, 3300160, w, 940000, "GRADE A",
              "5 of 5 ISRO quality gates passed", SAGE)

    textbox(slide, x, 4300000, w, 480000,
            "Streamlit co-registration studio, SIFT engine. "
            "Full-resolution CLI pair (TMC-2 ↔ OHRC, 10,107 × 1,200 → 4,096 px): "
            "11,361 raw matches → 1,973 inliers (17.4%), 0.562 px fit, "
            "0.631 px hold-out, Grade B.",
            size=8.5, color=MUTED, line=0.98)


def build_terrain(slide) -> None:
    """Slide 11 — 3D terrain reconstruction and the landing-hazard classifier."""
    set_header(slide, "05 · WORK COMPLETED · 3D & HAZARD AI",
               "3D TERRAIN & LANDING HAZARD AI")

    dem = os.path.join(HERE, "03_dem_3d.png")
    hazard = os.path.join(HERE, "04_crater_hazard.png")
    framed_picture(slide, dem, 457200, 1140000, 4014800, 2560000)
    framed_picture(slide, hazard, 4672000, 1140000, 4014800, 2560000)

    textbox(slide, 457200, 3700000, 4014800, 230000,
            "3D topography & DEM · 78.0 m relief · 99.4% SfS fidelity",
            size=8.5, color=MUTED)
    textbox(slide, 4672000, 3700000, 4014800, 230000,
            "Landing hazard AI · 52 craters · CAUTION verdict",
            size=8.5, color=MUTED)

    y, h = 3990000, 700000
    w, step = 1952400, 2092400
    stats = [
        ("78.0 m", "relief · 5.0 m/px", AMBER),
        ("32.4°", "maximum slope", AMBER),
        ("17.1%", "safe landing area", BLUE),
        ("31.3%", "top-site safety", SAGE),
    ]
    for i, (value, label, accent) in enumerate(stats):
        stat_card(slide, 457200 + i * step, y, w, h, value, label, accent,
                  value_pt=16, label_pt=8.5)


def build_crater_ai(slide) -> None:
    """Slide 12 — YOLO11n crater detector and the verification suite."""
    set_header(slide, "05 · WORK COMPLETED · CRATER AI & TESTS",
               "CRATER AI & VERIFICATION SUITE")

    chart = os.path.join(HERE, "scifi_yolo.png")
    framed_picture(slide, chart, 457200, 1145000, 5000000, 3400000)

    x, w = 5760720, 2926080
    stat_card(slide, x, 1280160, w, 880000, "0.573",
              "mAP@0.50 · peak 0.585 at epoch 8", AMBER)
    stat_card(slide, x, 2240160, w, 880000, "0.242",
              "mAP@0.50:0.95", AMBER)
    stat_card(slide, x, 3200160, w, 880000, "0.626 / 0.534",
              "precision / recall", BLUE)

    textbox(slide, x, 4150000, w, 620000,
            "YOLO11n fine-tuned for 10 epochs · batch 8 · 640 px tiles · "
            "Apple-Silicon MPS backend. "
            "Verification suite: 300 automated tests across 49 modules, "
            "the five ISRO gates and a reproducible benchmark harness.",
            size=8.5, color=MUTED, line=0.98)


def renumber(slide, number: int) -> None:
    for shape in slide.shapes:
        if shape.name != "Text 53" or not shape.has_text_frame:
            continue
        set_text(shape, f"{number} / {TOTAL_PAGES}", size=9, color=MUTED,
                 align=PP_ALIGN.RIGHT)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    base = next((p for p in BASE_CANDIDATES if os.path.exists(p)), None)
    if base is None:
        raise SystemExit(
            "missing pristine base deck; expected docs/presentation/deck_base.pptx "
            "(the original 12-slide presentation)."
        )

    prs = Presentation(base)
    if len(prs.slides) != 12:
        raise SystemExit(f"expected 12 slides in the base deck, found {len(prs.slides)}")

    # Slide 9 is the fabricated benchmark; strip it down to the chrome, then
    # clone that clean chrome for the three new evidence slides.
    strip_to_chrome(prs.slides[8])

    for _ in range(3):
        clone_slide(prs, prs.slides[8])
    # Clones are appended at the end; pull them in directly after slide 9.
    for offset in range(3):
        move_slide(prs, 12 + offset, 9 + offset)

    build_benchmark(prs.slides[8])
    build_live_run(prs.slides[9])
    build_terrain(prs.slides[10])
    build_crater_ai(prs.slides[11])

    # The original conclusion still quotes the old 0.384 px marketing headline
    # ("beating all baselines"). Replace it with the measured certification numbers.
    for shape in prs.slides[13].shapes:
        if shape.name == "Text 59" and shape.has_text_frame:
            para = shape.text_frame.paragraphs[0]
            if para.runs:
                para.runs[0].text = (
                    "0.434 px refined residual in the certified UI run; "
                    "0.562 px fit and 0.631 px hold-out on the full-resolution "
                    "TMC-2 ↔ OHRC pair."
                )
                for extra in para.runs[1:]:
                    extra._r.getparent().remove(extra._r)

    for i, slide in enumerate(prs.slides, start=1):
        renumber(slide, i)

    prs.save(DECK)
    print(f"wrote {DECK}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
