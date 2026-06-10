# apps/ai/tsd_processing/ingestor.py

"""
TSD Document Ingestor — first stage of the TSD analysis pipeline.

Responsibility:
    Parses a Technical Software Document (TSD) PDF into structured,
    page-aware data objects that carry:
        - Extracted text with block-level bounding boxes
        - Extracted diagram images with page coordinates
        - Section heading detection
        - Block IDs in the format "p{page}_b{idx}" and "p{page}_d{idx}"
          that map directly to CitationAnchor.block_id in the review
          models for click-to-source frontend navigation.

Why block-level granularity?
    The Multi-Agent pipeline needs precise source locations so the
    Mediator's final citations can be traced back to exact PDF regions.
    Page-level granularity is insufficient for the "click-to-source"
    feature — the frontend needs bbox coordinates to scroll and highlight
    the exact evidence region in the PDF viewer.

Dependency chain:
    document_processing.py [2]   (existing PDF extraction foundation)
         ↓
    ingestor.py                  ← YOU ARE HERE
         ↓
    raptor.py                    (needs TSDPage, TextBlock)
    graph_builder.py             (needs TSDPage, TextBlock, DiagramBlock)
         ↓
    retrieval/router.py
         ↓
    analysis_service.py

Tech stack:
    PyMuPDF (fitz) — same library already used in document_processing.py [2]
    base64           — for encoding diagram image bytes
    dataclasses      — for typed, immutable data containers
"""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sdr.apps.workspace.document_processing import convert_pdf_to_markdown

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyMuPDF availability guard
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF

    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False
    logger.error(
        "PyMuPDF (fitz) is not installed. TSD ingestion will not function. "
        "Install with: pip install pymupdf"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum character count for a text block to be considered meaningful.
# Blocks below this threshold are likely page numbers, headers, or artefacts.
_MIN_BLOCK_TEXT_LENGTH = 20

# Minimum image size in bytes — images below this are icons/logos, not diagrams.
# Consistent with VisionAgent._MIN_IMAGE_BYTES in agents/vision.py.
_MIN_DIAGRAM_BYTES = 512

# Maximum caption search radius in points below a diagram bounding box.
# PyMuPDF coordinate system: y increases downward.
_CAPTION_SEARCH_RADIUS_PT = 60.0

# Heading detection: font size multiplier above page median to be a heading.
_HEADING_FONT_SIZE_MULTIPLIER = 1.15

# Supported image formats for diagram extraction.
_SUPPORTED_IMAGE_FORMATS = frozenset({"png", "jpeg", "jpg", "gif", "webp"})
_FORMAT_NORMALISATION = {"jpg": "jpeg"}


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TextBlock:
    """
    A single block of text extracted from one page of the TSD PDF.

    block_id format: "p{page_number}_b{block_index}"
    Example:         "p3_b12"

    This ID is stored in CitationAnchor.block_id in the review models
    to enable click-to-source navigation in the frontend PDF viewer.

    bbox coordinates are in PDF coordinate space (points from bottom-left
    of the page for PyMuPDF's default coordinate system).
    """

    block_id: str  # "p{page}_b{idx}"
    text: str  # cleaned, stripped text content
    page_number: int  # 1-based page number
    bbox_x0: float  # left edge in PDF points
    bbox_y0: float  # top edge in PDF points
    bbox_x1: float  # right edge in PDF points
    bbox_y1: float  # bottom edge in PDF points
    font_size: float = 0.0  # dominant font size in this block
    is_bold: bool = False  # whether dominant font is bold
    is_heading: bool = False  # True if heuristically identified as heading
    section_heading: Optional[str] = None  # resolved page/section heading

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Returns bbox as (x0, y0, x1, y1) tuple for convenience."""
        return (self.bbox_x0, self.bbox_y0, self.bbox_x1, self.bbox_y1)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def is_valid(self) -> bool:
        """Returns True if this block has meaningful text content."""
        return (
            bool(self.block_id)
            and len(self.text.strip()) >= _MIN_BLOCK_TEXT_LENGTH
            and self.page_number > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialises to a plain dict for logging and downstream use."""
        return {
            "block_id": self.block_id,
            "text": self.text,
            "page_number": self.page_number,
            "bbox": {
                "x0": self.bbox_x0,
                "y0": self.bbox_y0,
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
            },
            "font_size": self.font_size,
            "is_bold": self.is_bold,
            "is_heading": self.is_heading,
        }


@dataclass
class DiagramBlock:
    """
    A single diagram or architectural image extracted from one page
    of the TSD PDF.

    diagram_id format: "p{page_number}_d{diagram_index}"
    Example:           "p5_d2"

    This ID maps to:
        - CitationAnchor.block_id in the review models
        - Finding.diagram_id in the review models
        - DiagramInput.diagram_id in agents/vision.py

    Diagram metadata is extracted eagerly during ingest, but image bytes are
    materialized lazily from a preserved source PDF only when downstream
    consumers actually resolve this diagram for Vision or review tasks.
    """

    diagram_id: str  # "p{page}_d{idx}"
    page_number: int  # 1-based page number
    bbox_x0: float  # left edge in PDF points
    bbox_y0: float  # top edge in PDF points
    bbox_x1: float  # right edge in PDF points
    bbox_y1: float  # bottom edge in PDF points
    image_b64: str = ""  # base64-encoded image bytes, populated lazily
    image_format: str = "png"  # "png" | "jpeg" | "gif" | "webp"
    caption: Optional[str] = None  # text immediately below the diagram
    surrounding_text: Optional[str] = None  # nearby text context for Vision agent
    width_pt: float = 0.0  # diagram width in PDF points
    height_pt: float = 0.0  # diagram height in PDF points
    source_pdf_path: Optional[str] = None  # copied source PDF for lazy loads
    image_xref: Optional[int] = None  # PyMuPDF xref inside source_pdf_path

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.bbox_x0, self.bbox_y0, self.bbox_x1, self.bbox_y1)

    def ensure_image_loaded(self, min_diagram_bytes: int = _MIN_DIAGRAM_BYTES) -> bool:
        """
        Loads and caches base64 image data from the preserved source PDF.

        Returns:
            True if image_b64 is available after the call, else False.
        """
        if self.image_b64:
            return True
        if not self.source_pdf_path or self.image_xref is None:
            return False

        started_at = time.perf_counter()

        try:
            pdf = fitz.open(self.source_pdf_path)
        except Exception as exc:
            logger.warning(
                "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d failed "
                "to open source PDF '%s': %s",
                self.diagram_id,
                self.page_number,
                self.source_pdf_path,
                exc,
            )
            return False

        try:
            try:
                base_image = pdf.extract_image(self.image_xref)
            except Exception as exc:
                logger.info(
                    "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d "
                    "elapsed=%.3fs success=false reason=extract_failed error=%s",
                    self.diagram_id,
                    self.page_number,
                    time.perf_counter() - started_at,
                    exc,
                )
                return False

            image_bytes: bytes = base_image.get("image", b"")
            if len(image_bytes) < min_diagram_bytes:
                logger.info(
                    "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d "
                    "elapsed=%.3fs success=false reason=too_small bytes=%d",
                    self.diagram_id,
                    self.page_number,
                    time.perf_counter() - started_at,
                    len(image_bytes),
                )
                return False

            image_format = _FORMAT_NORMALISATION.get(
                str(base_image.get("ext", "png")).lower(), "png"
            )
            if image_format not in _SUPPORTED_IMAGE_FORMATS:
                logger.info(
                    "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d "
                    "elapsed=%.3fs success=false reason=unsupported_format format=%s",
                    self.diagram_id,
                    self.page_number,
                    time.perf_counter() - started_at,
                    image_format,
                )
                return False

            self.image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            self.image_format = image_format
            logger.info(
                "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d "
                "elapsed=%.3fs success=true",
                self.diagram_id,
                self.page_number,
                time.perf_counter() - started_at,
            )
            return True
        finally:
            pdf.close()

    def is_valid(self) -> bool:
        """
        Returns True if this diagram has minimum required fields and
        a non-trivial image size suitable for Vision agent analysis.
        """
        if not self.diagram_id or not self.image_b64 or self.page_number <= 0:
            return False
        try:
            image_bytes = base64.b64decode(self.image_b64)
            return len(image_bytes) >= _MIN_DIAGRAM_BYTES
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagram_id": self.diagram_id,
            "image_format": self.image_format,
            "page_number": self.page_number,
            "bbox": {
                "x0": self.bbox_x0,
                "y0": self.bbox_y0,
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
            },
            "caption": self.caption,
            "width_pt": self.width_pt,
            "height_pt": self.height_pt,
        }


@dataclass
class TSDPage:
    """
    Structured representation of a single page from the TSD PDF.

    Produced by TSDIngestor.ingest() and consumed by:
        - raptor.py       (builds summarisation tree from text_blocks)
        - graph_builder.py (extracts entities/relations from text_blocks)
        - analysis_service.py (retrieves context, passes diagrams to Vision)

    text_blocks are ordered by their vertical position on the page
    (bbox_y0 ascending) so downstream consumers can process content
    in natural reading order.
    """

    page_number: int  # 1-based
    text_blocks: List[TextBlock] = field(default_factory=list)
    diagrams: List[DiagramBlock] = field(default_factory=list)
    section_heading: Optional[str] = None  # dominant heading on this page
    raw_text: str = ""  # full page text (for fallback use)
    markdown_text: str = ""  # page markdown used by retrieval and screening
    width_pt: float = 0.0  # page width in PDF points
    height_pt: float = 0.0  # page height in PDF points
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_text(self) -> str:
        """
        Returns concatenated text from all valid text blocks on this page,
        joined by newlines in reading order.
        """
        if self.markdown_text.strip():
            return self.markdown_text.strip()
        return "\n".join(block.text for block in self.text_blocks if block.is_valid())

    @property
    def heading_blocks(self) -> List[TextBlock]:
        """Returns only the text blocks identified as headings."""
        return [b for b in self.text_blocks if b.is_heading]

    @property
    def has_diagrams(self) -> bool:
        return bool(self.diagrams)

    def get_blocks_near_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        radius_pt: float = 50.0,
    ) -> List[TextBlock]:
        """
        Returns text blocks whose bounding box falls within `radius_pt`
        points of the given bbox. Used to find caption text near diagrams.

        Args:
            bbox:      (x0, y0, x1, y1) reference bounding box.
            radius_pt: Search radius in PDF points.

        Returns:
            List of TextBlock instances within the radius, sorted by y0.
        """
        _, _, _, ref_y1 = bbox
        nearby = []
        for block in self.text_blocks:
            # Look below the diagram (higher y0 in PyMuPDF's top-down coords)
            if ref_y1 <= block.bbox_y0 <= ref_y1 + radius_pt:
                nearby.append(block)
        return sorted(nearby, key=lambda b: b.bbox_y0)


@dataclass
class TSDDocument:
    """
    Complete structured representation of an ingested TSD PDF.

    Produced by TSDIngestor.ingest() and passed to analysis_service.py
    as the top-level input to the Multi-Agent analysis pipeline.
    """

    file_path: str
    document_name: str
    pages: List[TSDPage] = field(default_factory=list)
    total_pages: int = 0
    total_text_blocks: int = 0
    total_diagrams: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    temp_directories: List[str] = field(default_factory=list)
    min_diagram_bytes: int = _MIN_DIAGRAM_BYTES

    @property
    def all_text_blocks(self) -> List[TextBlock]:
        """
        Returns all text blocks across all pages in document order.
        Used by RAPTOR tree builder and graph builder.
        """
        blocks = []
        for page in self.pages:
            blocks.extend(page.text_blocks)
        return blocks

    @property
    def all_diagrams(self) -> List[DiagramBlock]:
        """Returns all diagrams across all pages in document order."""
        diagrams = []
        for page in self.pages:
            diagrams.extend(page.diagrams)
        return diagrams

    @property
    def full_text(self) -> str:
        """
        Returns the complete document text by joining all page texts.
        Used by chunk_text_with_context() [1] for RAPTOR tree building.
        """
        return "\n\n".join(page.all_text for page in self.pages if page.all_text)

    def cleanup_temporary_artifacts(self) -> None:
        for temp_dir in list(self.temp_directories):
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
                logger.info(
                    "TSDDocument.cleanup_temporary_artifacts: removed temp dir %s.",
                    temp_dir,
                )
            except FileNotFoundError:
                logger.debug(
                    "TSDDocument.cleanup_temporary_artifacts: temp dir already gone: %s.",
                    temp_dir,
                )
            except Exception as exc:
                logger.warning(
                    "TSDDocument.cleanup_temporary_artifacts: failed to remove %s: %s",
                    temp_dir,
                    exc,
                )
            finally:
                if temp_dir in self.temp_directories:
                    self.temp_directories.remove(temp_dir)

    def get_block_by_id(self, block_id: str) -> Optional[TextBlock]:
        """
        Looks up a TextBlock by its block_id.
        Used by analysis_service.py to resolve CitationAnchor block_ids
        back to source content for the click-to-source feature.

        Args:
            block_id: "p{page}_b{idx}" format string.

        Returns:
            The matching TextBlock, or None if not found.
        """
        for page in self.pages:
            for block in page.text_blocks:
                if block.block_id == block_id:
                    return block
        return None

    def get_diagram_by_id(self, diagram_id: str) -> Optional[DiagramBlock]:
        """
        Looks up a DiagramBlock by its diagram_id.
        Used by analysis_service.py to resolve Vision agent diagram IDs.

        Args:
            diagram_id: "p{page}_d{idx}" format string.

        Returns:
            The matching DiagramBlock with image bytes materialized on demand,
            or None if not found.
        """
        for page in self.pages:
            for diagram in page.diagrams:
                if diagram.diagram_id == diagram_id:
                    diagram.ensure_image_loaded(self.min_diagram_bytes)
                    return diagram
        return None


# ---------------------------------------------------------------------------
# TSD Ingestor
# ---------------------------------------------------------------------------


class TSDIngestor:
    """
    Parses a TSD PDF into a structured TSDDocument using PyMuPDF.

    Extends the existing document_processing.py [2] foundation with:
        - Block-level text extraction with bounding boxes
        - Heading detection via font size heuristics
        - Diagram metadata extraction with lazy image loading
        - Caption extraction for diagrams
        - Section heading tracking across pages

    Usage:
        ingestor = TSDIngestor()
        tsd_document = ingestor.ingest(file_path, document_name)

    The returned TSDDocument is passed to:
        - RAPTORTreeBuilder (raptor.py)
        - TSDGraphBuilder (graph_builder.py)
        - HybridRetrievalRouter (retrieval/router.py)
        - VisionAgent (agents/vision.py) via analysis_service.py
    """

    def __init__(
        self,
        min_block_text_length: int = _MIN_BLOCK_TEXT_LENGTH,
        min_diagram_bytes: int = _MIN_DIAGRAM_BYTES,
        caption_search_radius: float = _CAPTION_SEARCH_RADIUS_PT,
        heading_font_multiplier: float = _HEADING_FONT_SIZE_MULTIPLIER,
    ) -> None:
        """
        Args:
            min_block_text_length: Minimum characters for a text block to
                                   be included. Filters page numbers and
                                   single-word artefacts.
            min_diagram_bytes:     Minimum raw image size in bytes. Filters
                                   icons and decorative elements.
            caption_search_radius: Points below a diagram bbox to search
                                   for caption text.
            heading_font_multiplier: Font size multiplier above page median
                                     to classify a block as a heading.
        """
        if not FITZ_AVAILABLE:
            raise RuntimeError(
                "TSDIngestor requires PyMuPDF. Install with: pip install pymupdf"
            )

        self.min_block_text_length = min_block_text_length
        self.min_diagram_bytes = min_diagram_bytes
        self.caption_search_radius = caption_search_radius
        self.heading_font_multiplier = heading_font_multiplier
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def ingest(
        self,
        file_path: str,
        document_name: Optional[str] = None,
    ) -> TSDDocument:
        """
        Parses a TSD PDF into a fully structured TSDDocument.

        This is the single public entry point for the TSD ingestion pipeline.
        Called by analysis_service.py before building the RAPTOR tree,
        the GraphRAG index, and running the Multi-Agent debate.

        Pipeline per page:
            1. Extract raw text blocks with bounding boxes via PyMuPDF.
            2. Compute page font size median for heading detection.
            3. Classify each block as heading or body text.
            4. Extract diagram metadata and preserve lazy image references.
            5. Extract captions for each diagram from nearby text blocks.
            6. Build TSDPage and append to TSDDocument.

        Args:
            file_path:     Absolute path to the PDF file on disk.
                           Produced by get_local_file_path() from
                           document_processing.py [2].
            document_name: Human-readable name for the document.
                           Defaults to the filename stem if not provided.

        Returns:
            A fully populated TSDDocument ready for downstream processing.

        Raises:
            FileNotFoundError: If file_path does not exist.
            RuntimeError:      If PyMuPDF fails to open the file.
        """
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
                page_number = page_index + 1  # convert to 1-based
                fitz_page = pdf.load_page(page_index)

                tsd_page = self._process_page(
                    fitz_page=fitz_page,
                    pdf=pdf,
                    page_number=page_number,
                    current_section_heading=current_section_heading,
                    page_markdown_text=markdown_pages.get(page_number, ""),
                    source_pdf_path=source_pdf_path,
                )
                diagram_extract_elapsed += float(
                    tsd_page.metadata.get("diagram_extraction_elapsed", 0.0)
                )

                # Carry the most recent heading forward across pages
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
        pdf: "fitz.Document",
        page_number: int,
        current_section_heading: Optional[str],
        page_markdown_text: str,
        source_pdf_path: str,
    ) -> TSDPage:
        """
        Processes a single PyMuPDF page into a TSDPage dataclass.

        Args:
            fitz_page:              The PyMuPDF page object.
            pdf:                    The open PyMuPDF document for page processing.
            page_number:            1-based page number.
            current_section_heading: Heading carried over from the previous page.

        Returns:
            A populated TSDPage instance.
        """
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

        # --- Extract text blocks ---
        text_blocks = self._extract_text_blocks(fitz_page, page_number)
        tsd_page.text_blocks = text_blocks

        # --- Update section heading from this page's headings ---
        if tsd_page.heading_blocks:
            tsd_page.section_heading = tsd_page.heading_blocks[0].text

        for block in text_blocks:
            block.section_heading = tsd_page.section_heading

        if not tsd_page.markdown_text.strip():
            tsd_page.markdown_text = tsd_page.all_text

        # --- Extract diagrams ---
        diagram_started_at = time.perf_counter()
        diagrams = self._extract_diagrams(
            fitz_page,
            pdf,
            page_number,
            tsd_page,
            source_pdf_path=source_pdf_path,
        )
        tsd_page.metadata["diagram_extraction_elapsed"] = (
            time.perf_counter() - diagram_started_at
        )
        tsd_page.diagrams = diagrams

        return tsd_page

    # ------------------------------------------------------------------
    # Text block extraction
    # ------------------------------------------------------------------

    def _extract_text_blocks(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
    ) -> List[TextBlock]:
        """
        Extracts all meaningful text blocks from a PyMuPDF page.

        Uses PyMuPDF's dict extraction mode to get per-span font metadata
        (size, bold flag) for heading detection. Falls back to the simpler
        block extraction if dict mode fails.

        Args:
            fitz_page:   The PyMuPDF page object.
            page_number: 1-based page number for block_id generation.

        Returns:
            List of TextBlock instances ordered by vertical position (y0).
        """
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
        """
        Primary text extraction path using PyMuPDF's 'dict' mode.
        Provides per-span font size and bold flags for heading detection.
        """
        page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        raw_blocks = page_dict.get("blocks", [])

        # Collect all font sizes on this page to compute the median
        all_font_sizes: List[float] = []
        for block in raw_blocks:
            if block.get("type") != 0:  # 0 = text block in PyMuPDF
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

            # Aggregate all text in this block across lines and spans
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
                    # PyMuPDF bold flag: bit 4 of the flags integer
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

        # Sort by vertical position — natural reading order
        text_blocks.sort(key=lambda b: (b.bbox_y0, b.bbox_x0))
        return text_blocks

    def _extract_blocks_fallback(
        self,
        fitz_page: "fitz.Page",
        page_number: int,
    ) -> List[TextBlock]:
        """
        Fallback text extraction using PyMuPDF's simple block mode.
        No font metadata available — heading detection is disabled.
        Used when dict mode raises an exception.
        """
        raw_blocks = fitz_page.get_text("blocks") or []
        text_blocks: List[TextBlock] = []

        for block_idx, block in enumerate(raw_blocks):
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) < 5:
                continue
            block_type = block[6] if len(block) > 6 else 0
            if block_type != 0:  # skip image blocks
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

    # ------------------------------------------------------------------
    # Diagram extraction
    # ------------------------------------------------------------------

    def _extract_diagrams(
        self,
        fitz_page: "fitz.Page",
        pdf: "fitz.Document",
        page_number: int,
        tsd_page: TSDPage,
        source_pdf_path: str,
    ) -> List[DiagramBlock]:
        """
        Extracts diagram metadata from the PyMuPDF page without eagerly
        materializing image bytes. Image data is loaded later when a
        diagram is explicitly resolved.

        Args:
            fitz_page:   The PyMuPDF page object.
            pdf:         The open PyMuPDF document. Retained for interface
                         compatibility with existing call sites.
            page_number: 1-based page number for diagram_id generation.
            tsd_page:    The partially built TSDPage — used for caption lookup.

        Returns:
            List of DiagramBlock instances ordered by vertical position.
        """
        diagrams: List[DiagramBlock] = []
        image_list = fitz_page.get_images(full=True)

        diagram_idx = 0

        for img_info in image_list:
            xref = img_info[0]

            # Locate the image bbox on the page via get_image_rects()
            bbox = self._get_image_bbox(fitz_page, xref)

            # Extract caption from text blocks immediately below the bbox
            caption = self._extract_caption(tsd_page, bbox)

            # Extract surrounding text — text blocks within 2x caption radius
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

        # Sort by vertical position — natural reading order
        diagrams.sort(key=lambda d: (d.bbox_y0, d.bbox_x0))
        return diagrams

    def _get_image_bbox(
        self,
        fitz_page: "fitz.Page",
        xref: int,
    ) -> Tuple[float, float, float, float]:
        """
        Retrieves the bounding box of an image on the page using
        PyMuPDF's get_image_rects() method.

        Falls back to a zero bbox if the image cannot be located —
        this is non-fatal since the image bytes are still extracted
        and the bbox is only used for caption lookup and click-to-source.

        Args:
            fitz_page: The PyMuPDF page object.
            xref:      The image cross-reference number.

        Returns:
            (x0, y0, x1, y1) bounding box tuple in PDF points.
        """
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
        """
        Extracts a caption for a diagram by searching for short text blocks
        immediately below the diagram's bounding box.

        A block is considered a caption candidate if:
            - It falls within _CAPTION_SEARCH_RADIUS_PT points below the bbox
            - It is short (<=  20 words) — captions are rarely long paragraphs
            - It is not already classified as a heading

        Args:
            tsd_page: The TSDPage containing extracted text blocks.
            bbox:     The diagram's bounding box (x0, y0, x1, y1).

        Returns:
            The caption string, or None if no suitable candidate is found.
        """
        nearby_blocks = tsd_page.get_blocks_near_bbox(
            bbox=bbox,
            radius_pt=self.caption_search_radius,
        )

        for block in nearby_blocks:
            # Skip headings — a heading below a diagram is a new section,
            # not a caption
            if block.is_heading:
                continue
            # Captions are short — skip long paragraphs
            if block.word_count > 20:
                continue
            # Must have actual text
            if not block.text.strip():
                continue
            return block.text.strip()

        return None

    def _extract_surrounding_text(
        self,
        tsd_page: TSDPage,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[str]:
        """
        Extracts text blocks within an extended radius around the diagram
        to provide the Vision agent with architectural context — service
        names, protocol descriptions, and component labels that may not
        appear inside the diagram image itself.

        Uses 2x the caption search radius to cast a wider net than
        caption extraction.

        Args:
            tsd_page: The TSDPage containing extracted text blocks.
            bbox:     The diagram's bounding box (x0, y0, x1, y1).

        Returns:
            Combined surrounding text as a single string, or None if empty.
        """
        extended_radius = self.caption_search_radius * 2.0
        nearby_blocks = tsd_page.get_blocks_near_bbox(
            bbox=bbox,
            radius_pt=extended_radius,
        )

        parts = [
            block.text.strip()
            for block in nearby_blocks
            if block.text.strip() and not block.is_heading
        ]

        if not parts:
            return None

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _is_heading(
        self,
        text: str,
        font_size: float,
        median_font_size: float,
        is_bold: bool,
    ) -> bool:
        """
        Heuristically classifies a text block as a section heading.

        A block is considered a heading if it satisfies ANY of:
            1. Font size exceeds the page median by the configured multiplier
               (default: 15% larger than median).
            2. Font is bold AND the text is short (<= 12 words) AND
               the text does not end with a full stop — headings rarely
               end with periods.
            3. Text matches a common numbered heading pattern
               (e.g. "1.", "2.3", "A.1.2") — common in security standards
               and TSDs.

        Args:
            text:             The block's cleaned text content.
            font_size:        The dominant font size in this block (points).
            median_font_size: The median font size across the page.
            is_bold:          Whether the dominant font is bold.

        Returns:
            True if the block is likely a section heading.
        """
        if not text or not text.strip():
            return False

        stripped = text.strip()
        word_count = len(stripped.split())

        # Rule 1: significantly larger font than page median
        if font_size > 0 and median_font_size > 0:
            if font_size >= median_font_size * self.heading_font_multiplier:
                return True

        # Rule 2: bold + short + no trailing period
        if is_bold and word_count <= 12 and not stripped.endswith("."):
            return True

        # Rule 3: numbered section heading pattern
        # Matches: "1.", "1.2", "2.3.4", "A.", "A.1", "Section 3"
        if _HEADING_PATTERN.match(stripped):
            return True

        return False

    # ------------------------------------------------------------------
    # PDF metadata extraction
    # ------------------------------------------------------------------

    def _extract_pdf_metadata(
        self,
        pdf: "fitz.Document",
    ) -> Dict[str, Any]:
        """
        Extracts document-level metadata from the PDF file.

        Fields extracted (all may be empty strings in poorly-formed PDFs):
            title, author, subject, creator, producer, creation_date,
            modification_date, total_pages.

        Args:
            pdf: The open PyMuPDF document.

        Returns:
            A plain dict of metadata fields. Never raises — returns an
            empty dict with total_pages on any failure.
        """
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


# ---------------------------------------------------------------------------
# Compiled regex — module level for performance
# ---------------------------------------------------------------------------

# Matches numbered and lettered heading patterns common in TSDs and standards:
#   "1."  "1.2"  "2.3.4"  "A."  "A.1"  "Section 3"  "CHAPTER 2"
_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:\d+\.)+\d*"  # numeric: 1. / 1.2 / 2.3.4
    r"|[A-Z]\.(?:\d+\.)*"  # alpha: A. / A.1. / B.2.3.
    r"|(?:Section|Chapter|Part|Appendix)\s+\d+"  # keyword headings
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Module-level pure utility functions
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PyMuPDF by:
        1. Stripping leading/trailing whitespace.
        2. Collapsing multiple consecutive blank lines to a single newline.
        3. Removing non-printable control characters (except newline/tab).
        4. Normalising Unicode ligatures common in PDF fonts
           (e.g. ﬁ → fi, ﬂ → fl).

    Used by both primary and fallback text extraction paths.

    Args:
        text: Raw text string from PyMuPDF.

    Returns:
        Cleaned text string, or empty string if input is falsy.
    """
    if not text:
        return ""

    # Remove non-printable control characters — keep \n and \t
    cleaned = re.sub(r"[^\S\n\t ]+", " ", text)

    # Normalise common PDF ligatures that PyMuPDF doesn't always expand
    ligature_map = {
        "\ufb01": "fi",  # ﬁ
        "\ufb02": "fl",  # ﬂ
        "\ufb03": "ffi",  # ﬃ
        "\ufb04": "ffl",  # ﬄ
        "\ufb00": "ff",  # ﬀ
        "\u2019": "'",  # right single quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "--",  # em dash
    }
    for ligature, replacement in ligature_map.items():
        cleaned = cleaned.replace(ligature, replacement)

    # Collapse 3+ consecutive newlines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def _compute_median(values: List[float]) -> float:
    """
    Computes the median of a list of floats.

    Used for font size median calculation in heading detection.
    Returns 0.0 for an empty list to avoid ZeroDivisionError.

    Args:
        values: List of float values. May be empty.

    Returns:
        The median value, or 0.0 if the list is empty.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]
    else:
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    # Dataclasses — imported by raptor.py, graph_builder.py,
    # analysis_service.py, and agents/vision.py
    "TextBlock",
    "DiagramBlock",
    "TSDPage",
    "TSDDocument",
    # Main ingestor class
    "TSDIngestor",
    # Pure utilities — usable independently in tests
    "_clean_text",
    "_compute_median",
]
