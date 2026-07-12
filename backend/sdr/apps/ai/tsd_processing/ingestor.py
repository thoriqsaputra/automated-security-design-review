from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from sdr.apps.workspace.document_processing import convert_pdf_to_markdown
from sdr.core.config import settings
from sdr.apps.ai.tsd_processing.document_models import DiagramBlock, TSDDocument, TSDPage, TextBlock

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF

    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.error(
        "PyMuPDF (fitz) is not installed. TSD ingestion will not function. "
        "Install with: pip install pymupdf"
    )
_MIN_BLOCK_TEXT_LENGTH = 10


def _min_block_text_length() -> int:
    try:
        return max(0, int(getattr(settings, "AI_TSD_MIN_BLOCK_TEXT_LENGTH", _MIN_BLOCK_TEXT_LENGTH)))
    except (TypeError, ValueError):
        return _MIN_BLOCK_TEXT_LENGTH

_MIN_DIAGRAM_BYTES = 512
_CAPTION_SEARCH_RADIUS_PT = 60.0
_CAPTION_LABEL_RE = re.compile(r"^(figure|table|diagram)\s*\d+", re.IGNORECASE)
_HEADING_FONT_SIZE_MULTIPLIER = 1.15
_SUPPORTED_IMAGE_FORMATS = frozenset({"png", "jpeg", "jpg", "gif", "webp"})
_FORMAT_NORMALISATION = {"jpg": "jpeg"}


class TSDIngestor:
    def __init__(
        self,
        min_block_text_length: Optional[int] = None,
        min_diagram_bytes: int = _MIN_DIAGRAM_BYTES,
        caption_search_radius: float = _CAPTION_SEARCH_RADIUS_PT,
        heading_font_multiplier: float = _HEADING_FONT_SIZE_MULTIPLIER,
    ) -> None:
        if not FITZ_AVAILABLE:
            raise RuntimeError(
                "TSDIngestor requires PyMuPDF. Install with: pip install pymupdf"
            )

        self.min_block_text_length = (
            min_block_text_length if min_block_text_length is not None else _min_block_text_length()
        )
        self.min_diagram_bytes = min_diagram_bytes
        self.caption_search_radius = caption_search_radius
        self.heading_font_multiplier = heading_font_multiplier
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def ingest(
        self,
        file_path: str,
        document_name: Optional[str] = None,
    ) -> TSDDocument:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"TSDIngestor.ingest: file not found at '{file_path}'."
            )

        resolved_name = (
            document_name or os.path.splitext(os.path.basename(file_path))[0]
        )

        ingest_started_at = time.perf_counter()
        self.logger.info(
            "TSDIngestor.ingest: starting ingestion for '%s'.",
            resolved_name,
        )

        markdown_image_dir = tempfile.mkdtemp(prefix="tsd-markdown-images-")
        source_pdf_dir = tempfile.mkdtemp(prefix="tsd-source-pdf-")
        source_pdf_path = os.path.join(source_pdf_dir, os.path.basename(file_path))
        shutil.copyfile(file_path, source_pdf_path)
        self.logger.info(
            "TSDIngestor.ingest: converting '%s' to markdown via shared helper. "
            "image_output_dir=%s",
            resolved_name,
            markdown_image_dir,
        )
        markdown_started_at = time.perf_counter()
        markdown_conversion = convert_pdf_to_markdown(
            file_path,
            write_images=True,
            image_output_dir=markdown_image_dir,
        )
        markdown_elapsed = time.perf_counter() - markdown_started_at
        markdown_pages = {
            int(page.get("page_number", index)): page.get("text", "")
            for index, page in enumerate(markdown_conversion.get("pages") or [], start=1)
        }
        self.logger.info(
            "TSDIngestor.ingest: markdown conversion for '%s' completed in %.3fs.",
            resolved_name,
            markdown_elapsed,
        )

        try:
            pdf = fitz.open(file_path)
        except Exception as exc:
            shutil.rmtree(markdown_image_dir, ignore_errors=True)
            shutil.rmtree(source_pdf_dir, ignore_errors=True)
            raise RuntimeError(
                f"TSDIngestor.ingest: PyMuPDF failed to open '{file_path}': {exc}"
            ) from exc

        tsd_document = TSDDocument(
            file_path=file_path,
            document_name=resolved_name,
            total_pages=len(pdf),
            metadata={
                **self._extract_pdf_metadata(pdf),
                "markdown_conversion_method": markdown_conversion.get(
                    "conversion_method"
                ),
                "markdown_images_dir": markdown_image_dir,
                "source_pdf_path": source_pdf_path,
            },
            temp_directories=[markdown_image_dir, source_pdf_dir],
            min_diagram_bytes=self.min_diagram_bytes,
        )

        current_section_heading: Optional[str] = None
        page_scan_started_at = time.perf_counter()
        diagram_extract_elapsed = 0.0

        try:
            for page_index in range(len(pdf)):
                page_number = page_index + 1
                fitz_page = pdf.load_page(page_index)

                tsd_page = self._process_page(
                    fitz_page=fitz_page,
                    page_number=page_number,
                    current_section_heading=current_section_heading,
                    page_markdown_text=markdown_pages.get(page_number, ""),
                    source_pdf_path=source_pdf_path,
                )
                diagram_extract_elapsed += float(
                    tsd_page.metadata.get("diagram_extraction_elapsed", 0.0)
                )

                if tsd_page.heading_blocks:
                    current_section_heading = tsd_page.heading_blocks[-1].text

                tsd_document.pages.append(tsd_page)
                tsd_document.total_text_blocks += len(tsd_page.text_blocks)
                tsd_document.total_diagrams += len(tsd_page.diagrams)

                self.logger.debug(
                    "TSDIngestor.ingest: page %d/%d — %d text block(s), %d diagram(s).",
                    page_number,
                    len(pdf),
                    len(tsd_page.text_blocks),
                    len(tsd_page.diagrams),
                )
        finally:
            pdf.close()

        page_scan_elapsed = time.perf_counter() - page_scan_started_at

        self.logger.info(
            "TSDIngestor.ingest: completed '%s' — "
            "%d page(s), %d text block(s), %d diagram(s). total_elapsed=%.3fs",
            resolved_name,
            tsd_document.total_pages,
            tsd_document.total_text_blocks,
            tsd_document.total_diagrams,
            time.perf_counter() - ingest_started_at,
        )
        self.logger.info(
            "TSDIngestor.ingest: page/block scan for '%s' completed in %.3fs.",
            resolved_name,
            page_scan_elapsed,
        )
        self.logger.info(
            "TSDIngestor.ingest: diagram metadata extraction for '%s' completed in %.3fs.",
            resolved_name,
            diagram_extract_elapsed,
        )

        return tsd_document

    # ------------------------------------------------------------------
    # Page processing
    # ------------------------------------------------------------------

    def _process_page(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
        current_section_heading: Optional[str],
        page_markdown_text: str,
        source_pdf_path: str,
    ) -> TSDPage:
        page_rect = fitz_page.rect
        raw_text = fitz_page.get_text("text") or ""

        tsd_page = TSDPage(
            page_number=page_number,
            raw_text=raw_text,
            markdown_text=page_markdown_text.strip(),
            width_pt=page_rect.width,
            height_pt=page_rect.height,
            section_heading=current_section_heading,
            metadata={"diagram_extraction_elapsed": 0.0},
        )

        text_blocks = self._extract_text_blocks(fitz_page, page_number)
        tsd_page.text_blocks = text_blocks
        self._assign_section_headings(
            tsd_page=tsd_page,
            text_blocks=text_blocks,
            current_section_heading=current_section_heading,
        )

        if not tsd_page.markdown_text.strip():
            tsd_page.markdown_text = tsd_page.all_text

        diagram_started_at = time.perf_counter()
        diagrams = self._extract_diagrams(
            fitz_page,
            page_number,
            tsd_page,
            source_pdf_path=source_pdf_path,
        )
        tsd_page.metadata["diagram_extraction_elapsed"] = (
            time.perf_counter() - diagram_started_at
        )
        tsd_page.diagrams = diagrams

        return tsd_page

    def _assign_section_headings(
        self,
        *,
        tsd_page: TSDPage,
        text_blocks: List[TextBlock],
        current_section_heading: Optional[str],
    ) -> None:
        running_heading = current_section_heading
        for block in text_blocks:
            if block.is_heading:
                running_heading = block.text
            block.section_heading = running_heading
        tsd_page.section_heading = (
            tsd_page.heading_blocks[-1].text if tsd_page.heading_blocks else running_heading
        )

    def _extract_text_blocks(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
    ) -> List[TextBlock]:
        try:
            return self._extract_blocks_from_dict(fitz_page, page_number)
        except Exception as exc:
            self.logger.warning(
                "TSDIngestor._extract_text_blocks: dict extraction failed "
                "for page %d (%s) — falling back to block mode.",
                page_number,
                exc,
            )
            return self._extract_blocks_fallback(fitz_page, page_number)

    def _extract_blocks_from_dict(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
    ) -> List[TextBlock]:
        page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        raw_blocks = page_dict.get("blocks", [])

        all_font_sizes: List[float] = []
        for block in raw_blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0.0)
                    if size > 0:
                        all_font_sizes.append(size)

        median_font_size = _compute_median(all_font_sizes) if all_font_sizes else 12.0

        text_blocks: List[TextBlock] = []
        block_idx = 0

        for block in raw_blocks:
            if block.get("type") != 0:
                continue

            block_lines = []
            span_font_sizes: List[float] = []
            bold_span_count = 0
            total_span_count = 0

            for line in block.get("lines", []):
                line_text_parts = []
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_text_parts.append(span_text)
                    size = span.get("size", 0.0)
                    if size > 0:
                        span_font_sizes.append(size)
                    flags = span.get("flags", 0)
                    if flags & 16:
                        bold_span_count += 1
                    total_span_count += 1
                if line_text_parts:
                    block_lines.append(" ".join(line_text_parts))

            block_text = "\n".join(block_lines).strip()

            if len(block_text) < self.min_block_text_length:
                continue

            bbox = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            dominant_font_size = (
                max(span_font_sizes) if span_font_sizes else median_font_size
            )
            is_bold = (
                bold_span_count > total_span_count // 2
                if total_span_count > 0
                else False
            )
            is_heading = self._is_heading(
                text=block_text,
                font_size=dominant_font_size,
                median_font_size=median_font_size,
                is_bold=is_bold,
            )

            text_blocks.append(
                TextBlock(
                    block_id=f"p{page_number}_b{block_idx}",
                    text=_clean_text(block_text),
                    page_number=page_number,
                    bbox_x0=float(bbox[0]),
                    bbox_y0=float(bbox[1]),
                    bbox_x1=float(bbox[2]),
                    bbox_y1=float(bbox[3]),
                    font_size=dominant_font_size,
                    is_bold=is_bold,
                    is_heading=is_heading,
                )
            )
            block_idx += 1

        text_blocks.sort(key=lambda b: (b.bbox_y0, b.bbox_x0))
        return text_blocks

    def _extract_blocks_fallback(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
    ) -> List[TextBlock]:
        raw_blocks = fitz_page.get_text("blocks") or []
        text_blocks: List[TextBlock] = []

        for block_idx, block in enumerate(raw_blocks):
            if len(block) < 5:
                continue
            block_type = block[6] if len(block) > 6 else 0
            if block_type != 0:
                continue

            block_text = _clean_text(str(block[4]).strip())
            if len(block_text) < self.min_block_text_length:
                continue

            text_blocks.append(
                TextBlock(
                    block_id=f"p{page_number}_b{block_idx}",
                    text=block_text,
                    page_number=page_number,
                    bbox_x0=float(block[0]),
                    bbox_y0=float(block[1]),
                    bbox_x1=float(block[2]),
                    bbox_y1=float(block[3]),
                    font_size=0.0,
                    is_bold=False,
                    is_heading=False,
                )
            )

        text_blocks.sort(key=lambda b: (b.bbox_y0, b.bbox_x0))
        return text_blocks

    def _extract_diagrams(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
        tsd_page: TSDPage,
        source_pdf_path: str,
    ) -> List[DiagramBlock]:
        diagrams: List[DiagramBlock] = []
        image_list = fitz_page.get_images(full=True)

        diagram_idx = 0

        for img_info in image_list:
            xref = img_info[0]

            bbox = self._get_image_bbox(fitz_page, xref)
            caption = self._extract_caption(tsd_page, bbox)
            surrounding_text = self._extract_surrounding_text(tsd_page, bbox)

            diagram_id = f"p{page_number}_d{diagram_idx}"

            diagrams.append(
                DiagramBlock(
                    diagram_id=diagram_id,
                    page_number=page_number,
                    bbox_x0=float(bbox[0]),
                    bbox_y0=float(bbox[1]),
                    bbox_x1=float(bbox[2]),
                    bbox_y1=float(bbox[3]),
                    caption=caption,
                    surrounding_text=surrounding_text,
                    width_pt=float(bbox[2] - bbox[0]),
                    height_pt=float(bbox[3] - bbox[1]),
                    source_pdf_path=source_pdf_path,
                    image_xref=xref,
                )
            )
            diagram_idx += 1

        diagrams.sort(key=lambda d: (d.bbox_y0, d.bbox_x0))
        return diagrams

    def _get_image_bbox(
        self,
        fitz_page: "fitz.Page",
        xref: int,
    ) -> Tuple[float, float, float, float]:
        try:
            rects = fitz_page.get_image_rects(xref)
            if rects:
                r = rects[0]
                return (r.x0, r.y0, r.x1, r.y1)
        except Exception as exc:
            self.logger.debug(
                "TSDIngestor._get_image_bbox: could not get bbox for "
                "xref=%d: %s — using zero bbox.",
                xref,
                exc,
            )
        return (0.0, 0.0, 0.0, 0.0)

    def _extract_caption(
        self,
        tsd_page: TSDPage,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[str]:
        for direction in ("below", "above"):
            nearby_blocks = tsd_page.get_blocks_near_bbox(
                bbox=bbox,
                radius_pt=self.caption_search_radius,
                direction=direction,
            )
            for block in nearby_blocks:
                if block.is_heading:
                    continue
                text = block.text.strip()
                if text and _CAPTION_LABEL_RE.match(text):
                    return text

        for direction in ("below", "above"):
            nearby_blocks = tsd_page.get_blocks_near_bbox(
                bbox=bbox,
                radius_pt=self.caption_search_radius,
                direction=direction,
            )

            for block in nearby_blocks:
                if block.is_heading:
                    continue
                if block.word_count > 20:
                    continue
                if not block.text.strip():
                    continue
                return block.text.strip()

        return None

    def _extract_surrounding_text(
        self,
        tsd_page: TSDPage,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[str]:
        extended_radius = self.caption_search_radius * 2.0
        above_blocks = tsd_page.get_blocks_near_bbox(
            bbox=bbox,
            radius_pt=extended_radius,
            direction="above",
        )
        below_blocks = tsd_page.get_blocks_near_bbox(
            bbox=bbox,
            radius_pt=extended_radius,
            direction="below",
        )

        parts = [
            block.text.strip()
            for block in (*above_blocks, *below_blocks)
            if block.text.strip() and not block.is_heading
        ]

        if not parts:
            return None

        return " ".join(parts)

    def _is_heading(
        self,
        text: str,
        font_size: float,
        median_font_size: float,
        is_bold: bool,
    ) -> bool:
        if not text or not text.strip():
            return False

        stripped = text.strip()
        word_count = len(stripped.split())

        if font_size > 0 and median_font_size > 0:
            if font_size >= median_font_size * self.heading_font_multiplier:
                return True

        if is_bold and word_count <= 12 and not stripped.endswith("."):
            return True

        if _HEADING_PATTERN.match(stripped):
            return True

        return False

    def _extract_pdf_metadata(
        self,
        pdf: "fitz.Document",
    ) -> Dict[str, Any]:
        try:
            raw_meta = pdf.metadata or {}
            return {
                "title": raw_meta.get("title", "").strip(),
                "author": raw_meta.get("author", "").strip(),
                "subject": raw_meta.get("subject", "").strip(),
                "creator": raw_meta.get("creator", "").strip(),
                "producer": raw_meta.get("producer", "").strip(),
                "creation_date": raw_meta.get("creationDate", "").strip(),
                "modification_date": raw_meta.get("modDate", "").strip(),
                "total_pages": len(pdf),
            }
        except Exception as exc:
            self.logger.debug(
                "TSDIngestor._extract_pdf_metadata: failed to extract "
                "metadata: %s — returning minimal dict.",
                exc,
            )
            return {"total_pages": len(pdf)}

_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:\d+\.)+\d*"
    r"|[A-Z]\.(?:\d+\.)*"
    r"|(?:Section|Chapter|Part|Appendix)\s+\d+"
    r")",
    re.IGNORECASE,
)

def _clean_text(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"[^\S\n\t ]+", " ", text)

    ligature_map = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb00": "ff",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
    }
    for ligature, replacement in ligature_map.items():
        cleaned = cleaned.replace(ligature, replacement)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def _compute_median(values: List[float]) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]
    else:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

__all__ = [
    "TextBlock",
    "DiagramBlock",
    "TSDPage",
    "TSDDocument",
    "TSDIngestor",
    "_clean_text",
    "_compute_median",
]
