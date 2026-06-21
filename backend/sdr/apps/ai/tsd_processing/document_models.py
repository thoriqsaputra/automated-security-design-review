from __future__ import annotations

import base64
import logging
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sdr.core.config import settings

logger = logging.getLogger(__name__)

_MIN_BLOCK_TEXT_LENGTH = 10
_MIN_DIAGRAM_BYTES = 512
_SUPPORTED_IMAGE_FORMATS = frozenset({"png", "jpeg", "jpg", "gif", "webp"})
_FORMAT_NORMALISATION = {"jpg": "jpeg"}


def _min_block_text_length() -> int:
    try:
        return max(0, int(getattr(settings, "AI_TSD_MIN_BLOCK_TEXT_LENGTH", _MIN_BLOCK_TEXT_LENGTH)))
    except (TypeError, ValueError):
        return _MIN_BLOCK_TEXT_LENGTH


@dataclass
class TextBlock:
    block_id: str
    text: str
    page_number: int
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    font_size: float = 0.0
    is_bold: bool = False
    is_heading: bool = False
    section_heading: Optional[str] = None

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.bbox_x0, self.bbox_y0, self.bbox_x1, self.bbox_y1)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def is_valid(self) -> bool:
        return (
            bool(self.block_id)
            and len(self.text.strip()) >= _min_block_text_length()
            and self.page_number > 0
        )

    def to_dict(self) -> Dict[str, Any]:
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
    diagram_id: str
    page_number: int
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    image_b64: str = ""
    image_format: str = "png"
    caption: Optional[str] = None
    surrounding_text: Optional[str] = None
    width_pt: float = 0.0
    height_pt: float = 0.0
    source_pdf_path: Optional[str] = None
    image_xref: Optional[int] = None

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.bbox_x0, self.bbox_y0, self.bbox_x1, self.bbox_y1)

    def ensure_image_loaded(self, min_diagram_bytes: int = _MIN_DIAGRAM_BYTES) -> bool:
        if self.image_b64:
            return True
        if not self.source_pdf_path or self.image_xref is None:
            return False

        try:
            import fitz  # PyMuPDF
        except Exception:
            logger.warning(
                "DiagramBlock.ensure_image_loaded: PyMuPDF unavailable for diagram_id=%s",
                self.diagram_id,
            )
            return False

        started_at = time.perf_counter()
        try:
            pdf = fitz.open(self.source_pdf_path)
        except Exception as exc:
            logger.warning(
                "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d failed to open source PDF '%s': %s",
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
                    "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d elapsed=%.3fs success=false reason=extract_failed error=%s",
                    self.diagram_id,
                    self.page_number,
                    time.perf_counter() - started_at,
                    exc,
                )
                return False

            image_bytes: bytes = base_image.get("image", b"")
            if len(image_bytes) < min_diagram_bytes:
                logger.info(
                    "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d elapsed=%.3fs success=false reason=too_small bytes=%d",
                    self.diagram_id,
                    self.page_number,
                    time.perf_counter() - started_at,
                    len(image_bytes),
                )
                return False

            image_format = _FORMAT_NORMALISATION.get(str(base_image.get("ext", "png")).lower(), "png")
            if image_format not in _SUPPORTED_IMAGE_FORMATS:
                logger.info(
                    "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d elapsed=%.3fs success=false reason=unsupported_format format=%s",
                    self.diagram_id,
                    self.page_number,
                    time.perf_counter() - started_at,
                    image_format,
                )
                return False

            self.image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            self.image_format = image_format
            logger.info(
                "DiagramBlock.ensure_image_loaded: diagram_id=%s page=%d elapsed=%.3fs success=true",
                self.diagram_id,
                self.page_number,
                time.perf_counter() - started_at,
            )
            return True
        finally:
            pdf.close()

    def is_valid(self) -> bool:
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
    page_number: int
    text_blocks: List[TextBlock] = field(default_factory=list)
    diagrams: List[DiagramBlock] = field(default_factory=list)
    section_heading: Optional[str] = None
    raw_text: str = ""
    markdown_text: str = ""
    width_pt: float = 0.0
    height_pt: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_text(self) -> str:
        if self.markdown_text.strip():
            return self.markdown_text.strip()
        return "\n".join(block.text for block in self.text_blocks if block.is_valid())

    @property
    def heading_blocks(self) -> List[TextBlock]:
        return [b for b in self.text_blocks if b.is_heading]

    def get_blocks_near_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        radius_pt: float = 50.0,
        direction: str = "below",
    ) -> List[TextBlock]:
        ref_x0, ref_y0, ref_x1, ref_y1 = bbox
        want_below = direction in ("below", "both")
        want_above = direction in ("above", "both")
        nearby = []
        for block in self.text_blocks:
            is_below = ref_y1 <= block.bbox_y0 <= ref_y1 + radius_pt
            is_above = ref_y0 - radius_pt <= block.bbox_y1 <= ref_y0
            if (want_below and is_below) or (want_above and is_above):
                nearby.append(block)
        return sorted(nearby, key=lambda b: b.bbox_y0)


@dataclass
class TSDDocument:
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
        blocks = []
        for page in self.pages:
            blocks.extend(page.text_blocks)
        return blocks

    @property
    def all_diagrams(self) -> List[DiagramBlock]:
        diagrams = []
        for page in self.pages:
            diagrams.extend(page.diagrams)
        return diagrams

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.all_text for page in self.pages if page.all_text)

    def cleanup_temporary_artifacts(self) -> None:
        for temp_dir in list(self.temp_directories):
            try:
                shutil.rmtree(temp_dir, ignore_errors=False)
                logger.info("TSDDocument.cleanup_temporary_artifacts: removed temp dir %s.", temp_dir)
            except FileNotFoundError:
                logger.debug("TSDDocument.cleanup_temporary_artifacts: temp dir already gone: %s.", temp_dir)
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
        for page in self.pages:
            for block in page.text_blocks:
                if block.block_id == block_id:
                    return block
        return None

    def get_diagram_by_id(self, diagram_id: str) -> Optional[DiagramBlock]:
        for page in self.pages:
            for diagram in page.diagrams:
                if diagram.diagram_id == diagram_id:
                    diagram.ensure_image_loaded(self.min_diagram_bytes)
                    return diagram
        return None


__all__ = ["TextBlock", "DiagramBlock", "TSDPage", "TSDDocument"]
