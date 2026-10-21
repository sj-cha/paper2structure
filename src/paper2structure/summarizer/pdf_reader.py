"""Turn a PDF into plain text + figure images (no LLM involved here).

Figures in papers come in two flavours and we handle both:
- embedded raster images (photos, micrographs)  -> ``page.get_images()``
- vector drawings (most plots: XRD patterns, curves) -> ``page.cluster_drawings()``
We collect the rectangles of both, merge ones that touch, drop tiny ones (logos, lines),
render each region to a PNG and attach the closest "Fig. N" caption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

CAPTION_RE = re.compile(r"^\s*(fig(ure)?\.?\s*\d+)", re.IGNORECASE)
# "Fig. 1(a) shows ..." is body text referring to a figure, not a caption
REFERENCE_RE = re.compile(r"^\s*fig(ure)?\.?\s*\d+\s*(\([a-z,\- ]+\))?\s*(shows|represents|presents|depicts|illustrates)", re.IGNORECASE)
MIN_SIDE_PT = 40  # regions smaller than this (in PDF points, 72 pt = 1 inch) are ignored
MERGE_GAP_PT = 12  # regions closer than this are treated as one figure
RENDER_DPI = 200


@dataclass
class ExtractedFigure:
    figure_id: str  # e.g. "p2_f1"
    page: int  # 1-based
    caption: str  # best-guess caption ("" if none found)
    image_path: Path
    page_captions: list[str] = field(default_factory=list)  # all captions on this page, to help the LLM


@dataclass
class PaperDocument:
    pdf_path: Path
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


def read_pdf(pdf_path: str | Path, figures_dir: str | Path, max_figures: int = 30) -> PaperDocument:
    pdf_path = Path(pdf_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    pages: list[str] = []
    figures: list[ExtractedFigure] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            regions = _figure_regions(page)
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
            text_parts, captions = [], []
            for x0, y0, x1, y1, text, *_ in blocks:
                rect = pymupdf.Rect(x0, y0, x1, y1)
                clean = " ".join(text.split())
                if not clean:
                    continue
                if CAPTION_RE.match(clean) and not REFERENCE_RE.match(clean):
                    captions.append((rect, clean))
                    text_parts.append(clean)
                elif not any(r.contains(rect) for r in regions):  # drop axis labels/ticks inside plots
                    text_parts.append(clean)
            pages.append("\n".join(text_parts))

            for k, region in enumerate(regions, start=1):
                if len(figures) >= max_figures:
                    break
                fig_id = f"p{page.number + 1}_f{k}"
                out = figures_dir / f"{fig_id}.png"
                clip = pymupdf.Rect(region.x0 - 4, region.y0 - 4, region.x1 + 4, region.y1 + 4) & page.rect
                page.get_pixmap(clip=clip, dpi=RENDER_DPI).save(out)
                figures.append(
                    ExtractedFigure(
                        figure_id=fig_id,
                        page=page.number + 1,
                        caption=_closest_caption(region, captions),
                        image_path=out,
                        page_captions=[c for _, c in captions],
                    )
                )
    return PaperDocument(pdf_path=pdf_path, pages=pages, figures=figures)
