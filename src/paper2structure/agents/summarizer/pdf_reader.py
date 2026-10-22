"""Turn a PDF into plain text + figure and table images (no LLM involved here).

Figures in papers come in two flavours and we handle both:
- embedded raster images (photos, micrographs)  -> ``page.get_images()``
- vector drawings (most plots: XRD patterns, curves) -> ``page.cluster_drawings()``
We collect the rectangles of both, merge ones that touch, drop tiny ones (logos, lines),
render each region to a PNG and attach the closest "Fig. N" caption.

Tables are found next to their "Table N" captions (``page.find_tables()``, or a drawn region such
as shaded rows), whatever their ruling style. Each is rendered to a PNG like a figure, and its text
stays in the page text: as a markdown table when that keeps every number, else as it was printed.
Text inside figures (axis ticks) is dropped, text inside tables never is.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pymupdf

_NUMBER = r"(S?\d+|[IVX]+)\b"  # "Table 2", "Table S1", "TABLE II"
_REFERS = (r"(shows?|represents?|presents?|depicts?|illustrates?|gives?|lists?|summari[sz]es?|reveals?|"
           r"compares?|contains?|reports?|indicates?|displays?|demonstrates?|confirms?|provides?|includes?|"
           r"collects?|tabulates?|suggests?)\b")
CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?\s*\d+)", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(table|tab\.)\s*" + _NUMBER, re.IGNORECASE)
# "Fig. 1(a) shows ..." or "Table 2 gives ..." is body text referring to a figure or table, not a caption
REFERENCE_RE = re.compile(r"^\s*(fig(ure)?\.?|table|tab\.)\s*" + _NUMBER + r"\s*(\([a-z,\- ]+\))?\s*" + _REFERS,
                          re.IGNORECASE)
FIGURE_NUMBER_RE = re.compile(r"^\s*fig(?:ure)?\.?\s*(\d+)", re.IGNORECASE)
TABLE_NUMBER_RE = re.compile(r"^\s*(?:table|tab\.)\s*" + _NUMBER, re.IGNORECASE)
# "Fig. 2", "Figures 3", but not "Supplementary Fig. 2" (a different figure)
CITATION_RE = re.compile(r"(?<!supplementary )(?<!supporting )\bfig(?:ure)?s?\.?\s*(\d+)", re.IGNORECASE)
TABLE_CITATION_RE = re.compile(r"(?<!supplementary )(?<!supporting )\b(?:table|tab\.)s?\s*" + _NUMBER, re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
LINE_BREAK_HYPHEN_RE = re.compile(r"([a-z])- ([a-z])")  # "syn- thesized" -> "synthesized"
NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?")
MIN_SIDE_PT = 40  # regions smaller than this (in PDF points, 72 pt = 1 inch) are ignored
MERGE_GAP_PT = 12  # regions closer than this are treated as one figure
TABLE_REACH_PT = 60  # a table starts this close below (or ends this close above) its caption
RENDER_DPI = 200
MAX_CONTEXT_SENTENCES = 8


@dataclass
class ExtractedFigure:
    figure_id: str  # e.g. "p2_f1" (a figure) or "p3_t1" (a table)
    page: int  # 1-based
    caption: str  # best-guess caption ("" if none found)
    image_path: Path
    page_captions: list[str] = field(default_factory=list)  # all captions on this page, to help the LLM
    context: list[str] = field(default_factory=list)  # body-text sentences citing this figure's number
    kind: Literal["figure", "table"] = "figure"


@dataclass
class PaperDocument:
    pages: list[str]  # text per page, figure-internal text (axis ticks etc.) removed
    figures: list[ExtractedFigure]

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[Page {i + 1}]\n{t}" for i, t in enumerate(self.pages))


def _merge_rects(rects: list[pymupdf.Rect], gap: float) -> list[pymupdf.Rect]:
    rects = [pymupdf.Rect(r) for r in rects]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                grown = pymupdf.Rect(rects[i].x0 - gap, rects[i].y0 - gap, rects[i].x1 + gap, rects[i].y1 + gap)
                if grown.intersects(rects[j]):
                    rects[i] = rects[i] | rects[j]
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return rects


def _figure_regions(page: pymupdf.Page) -> list[pymupdf.Rect]:
    rects: list[pymupdf.Rect] = []
    for img in page.get_images(full=True):
        rects.extend(page.get_image_rects(img[0]))
    try:
        rects.extend(page.cluster_drawings())
    except Exception:  # some malformed pages make the clustering fail; images still work
        pass
    page_area = page.rect.width * page.rect.height
    rects = [r & page.rect for r in rects]
    rects = [r for r in rects if r.width >= MIN_SIDE_PT and r.height >= MIN_SIDE_PT]
    rects = [r for r in rects if r.width * r.height < 0.95 * page_area]  # skip full-page backgrounds
    rects = _merge_rects(rects, MERGE_GAP_PT)
    return sorted(rects, key=lambda r: (r.y0, r.x0))


def _closest_caption(region: pymupdf.Rect, captions: list[tuple[pymupdf.Rect, str]]) -> str:
    """Prefer a caption just below the figure in the same column, else the nearest one."""
    best, best_score = "", float("inf")
    for rect, text in captions:
        overlaps_column = min(region.x1, rect.x1) - max(region.x0, rect.x0) > 0
        if rect.y0 >= region.y1 - 5:
            distance = rect.y0 - region.y1  # below
        elif region.contains(rect) or (rect.y0 > (region.y0 + region.y1) / 2 and rect.y1 >= region.y1):
            distance = 0  # caption overlaps the bottom of the figure region
        elif rect.y1 <= region.y0 + 5:
            distance = (region.y0 - rect.y1) + 50  # above: small penalty
        else:
            distance = 100  # beside
        score = distance + (0 if overlaps_column else 200)
        if score < best_score:
            best, best_score = text, score
    return best


def _citing_sentences(body: list[str], caption: str, kind: str = "figure") -> list[str]:
    """Body-text sentences that cite the figure or table number in ``caption`` (e.g. 'Fig. 3b
    shows ...', 'as listed in Table 2')."""
    number_re, citation_re = (TABLE_NUMBER_RE, TABLE_CITATION_RE) if kind == "table" else (FIGURE_NUMBER_RE, CITATION_RE)
    match = number_re.match(caption)
    if not match:
        return []
    number = match.group(1).upper()
    found: list[str] = []
    for paragraph in body:
        for sentence in SENTENCE_SPLIT_RE.split(paragraph):
            if any(m.group(1).upper() == number for m in citation_re.finditer(sentence)) and sentence not in found:
                found.append(sentence)
                if len(found) >= MAX_CONTEXT_SENTENCES:
                    return found
    return found


@dataclass
class _Table:
    rect: pymupdf.Rect
    caption: str
    cells: Optional[object] = None  # the pymupdf Table, when find_tables found it


def _beside(caption: pymupdf.Rect, rect: pymupdf.Rect) -> Optional[float]:
    """How far a table region is from a table caption in the same column (a caption above its
    table is the usual layout, below costs extra); None when they are not adjacent."""
    if min(rect.x1, caption.x1) - max(rect.x0, caption.x0) <= 0:
        return None
    if rect.contains(caption):
        return 0.0
    below = rect.y0 - caption.y1
    if -5 <= below <= TABLE_REACH_PT:
        return max(below, 0.0)
    above = caption.y0 - rect.y1
    if -5 <= above <= TABLE_REACH_PT:
        return max(above, 0.0) + TABLE_REACH_PT
    return None


def _tables(page: pymupdf.Page, captions: list[tuple[pymupdf.Rect, str]],
            regions: list[pymupdf.Rect]) -> list[_Table]:
    """The table beside each table caption: one page.find_tables() locates (ruled, shaded or
    booktabs tables), else a drawn region there. Run only on pages with a table caption."""
    if not captions:
        return []
    try:
        found = list(page.find_tables().tables)
    except Exception:  # noqa: BLE001 - a page the table finder cannot parse: drawn regions still count
        found = []
    candidates = [(pymupdf.Rect(t.bbox), t) for t in found] + [(r, None) for r in regions]
    tables, used = [], set()
    for cap_rect, cap_text in captions:
        scored = [(d, cells is None, k) for k, (rect, cells) in enumerate(candidates)
                  if k not in used and (d := _beside(cap_rect, rect)) is not None]
        if not scored:
            continue
        _, _, k = min(scored)
        used.add(k)
        rect, cells = candidates[k]
        tables.append(_Table(pymupdf.Rect(rect), cap_text, cells))
    return tables


def _mostly_inside(rect: pymupdf.Rect, region: pymupdf.Rect) -> bool:
    inter = rect & region
    return not inter.is_empty and inter.get_area() >= 0.5 * max(rect.get_area(), 1e-6)


def _drawn_table(region: pymupdf.Rect, table: _Table) -> bool:
    """A drawn region that is the table (its rules or shaded rows), not a figure; a region that
    also holds a figure stays one."""
    return _mostly_inside(region, table.rect) or (_mostly_inside(table.rect, region)
                                                  and region.get_area() <= 1.5 * table.rect.get_area())


def _table_text(table: _Table, printed: list[str]) -> str:
    """The table's text for the page: its cells as markdown rows when that keeps every number the
    printed text has (the table finder can merge cells), else the printed text blocks."""
    if table.cells is not None:
        try:
            markdown = table.cells.to_markdown().strip()
        except Exception:  # noqa: BLE001
            markdown = ""
        want, got = Counter(NUMBER_TOKEN_RE.findall(" ".join(printed))), Counter(NUMBER_TOKEN_RE.findall(markdown))
        if markdown and all(got[n] >= c for n, c in want.items()):
            return markdown
    return "\n".join(printed)


def read_pdf(pdf_path: str | Path, figures_dir: str | Path, max_figures: int = 30) -> PaperDocument:
    pdf_path = Path(pdf_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pages: list[str] = []
    body: list[str] = []  # text blocks that are neither captions nor inside figures or tables
    figures: list[ExtractedFigure] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            regions = _figure_regions(page)
            blocks = [(pymupdf.Rect(b[:4]), " ".join(b[4].split())) for b in page.get_text("blocks")]
            blocks = [(rect, clean) for rect, clean in blocks if clean]
            references = [REFERENCE_RE.match(clean) for _, clean in blocks]
            fig_captions = [(r, c) for (r, c), ref in zip(blocks, references) if CAPTION_RE.match(c) and not ref]
            table_captions = [(r, c) for (r, c), ref in zip(blocks, references) if TABLE_CAPTION_RE.match(c) and not ref]
            tables = _tables(page, table_captions, regions)
            regions = [r for r in regions if not any(_drawn_table(r, t) for t in tables)]
            in_table = [next((t for t in tables if _mostly_inside(rect, t.rect)), None) for rect, _ in blocks]
            captions = {c for _, c in fig_captions + table_captions}
            text_parts, written = [], set()
            for (rect, clean), table in zip(blocks, in_table):
                if clean in captions:
                    text_parts.append(clean)
                elif table is not None:  # the whole table once, where its first block is
                    if id(table) not in written:
                        written.add(id(table))
                        text_parts.append(_table_text(table, [c for (r, c), t in zip(blocks, in_table)
                                                              if t is table and c not in captions]))
                elif not any(r.contains(rect) for r in regions):  # drop axis labels/ticks inside plots
                    text_parts.append(clean)
                    body.append(LINE_BREAK_HYPHEN_RE.sub(r"\1\2", clean))
            pages.append("\n".join(text_parts))

            crops = ([(f"p{page.number + 1}_f{k}", "figure", r, _closest_caption(r, fig_captions))
                      for k, r in enumerate(regions, start=1)]
                     + [(f"p{page.number + 1}_t{k}", "table", t.rect, t.caption) for k, t in enumerate(tables, start=1)])
            for fig_id, kind, region, caption in crops:
                if len(figures) >= max_figures:
                    break
                out = figures_dir / f"{fig_id}.png"
                clip = pymupdf.Rect(region.x0 - 4, region.y0 - 4, region.x1 + 4, region.y1 + 4) & page.rect
                page.get_pixmap(clip=clip, dpi=RENDER_DPI).save(out)
                figures.append(ExtractedFigure(figure_id=fig_id, page=page.number + 1, caption=caption, image_path=out,
                                               page_captions=[c for _, c in fig_captions + table_captions], kind=kind))
    for fig in figures:  # citations can be on any page, so this runs after all pages are read
        fig.context = _citing_sentences(body, fig.caption, fig.kind)
    return PaperDocument(pages=pages, figures=figures)
