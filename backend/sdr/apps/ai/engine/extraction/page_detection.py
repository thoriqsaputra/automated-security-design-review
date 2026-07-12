from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import fitz


logger = logging.getLogger(__name__)

_TOP_LEVEL_V1_RE = re.compile(r"^V1(?:\s|$)")
_APPENDIX_A_RE = re.compile(r"\bappendix\s+a\b", re.IGNORECASE)
_CONTENTS_RE = re.compile(r"\b(?:contents|table of contents)\b", re.IGNORECASE)
_MIN_PLAUSIBLE_REQUIREMENT_RANGE_PAGES = 10


@dataclass(frozen=True)
class TOCEntry:
    level: int
    title: str
    page: int


@dataclass(frozen=True)
class PageInfo:
    page_number: int
    text: str
    lines: Tuple[str, ...]
    heading_lines: Tuple[str, ...]

    @property
    def leading_lines(self) -> Tuple[str, ...]:
        return self.lines[:20]


@dataclass(frozen=True)
class ASVSPageDetectionResult:
    start_page: Optional[int]
    end_page: Optional[int]
    source: str
    matched_anchors: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_page": self.start_page,
            "end_page": self.end_page,
            "source": self.source,
            "matched_anchors": self.matched_anchors,
        }


class ASVSPageRangeDetectionService:
    def __init__(
        self,
        *,
        get_local_file_path: Callable[[Any], AbstractContextManager],
    ) -> None:
        self._get_local_file_path = get_local_file_path

    def detect(self, source_doc: Any) -> ASVSPageDetectionResult:
        with self._get_local_file_path(source_doc.document) as local_path:
            doc = fitz.open(local_path)
            try:
                toc_entries = self._read_toc(doc)
                pages = self._read_pages(doc)
            finally:
                doc.close()

        req_start, req_end, req_anchors, req_source = self._detect_requirement_range(
            pages=pages,
            toc_entries=toc_entries,
        )
        req_anchors, req_source = self._flag_low_confidence_requirement_range(
            req_start=req_start,
            req_end=req_end,
            req_anchors=req_anchors,
            req_source=req_source,
        )
        source = "toc" if req_source == "toc" else "heuristic"
        return ASVSPageDetectionResult(
            start_page=req_start,
            end_page=req_end,
            source=source,
            matched_anchors={
                "requirement_range": req_anchors,
                "toc_entries_used": [entry.title for entry in toc_entries[:20]],
            },
        )

    def _flag_low_confidence_requirement_range(
        self,
        *,
        req_start: Optional[int],
        req_end: Optional[int],
        req_anchors: Dict[str, Any],
        req_source: str,
    ) -> Tuple[Dict[str, Any], str]:
        if (
            req_start is None
            or req_end is None
            or (req_end - req_start) >= _MIN_PLAUSIBLE_REQUIREMENT_RANGE_PAGES
        ):
            return req_anchors, req_source

        req_anchors = dict(req_anchors)
        req_anchors["warning"] = (
            f"detected requirement range spans only {req_end - req_start} page(s), "
            f"below the {_MIN_PLAUSIBLE_REQUIREMENT_RANGE_PAGES}-page plausibility threshold "
            "for an ASVS-style standard; the range is still used as-is but may be clipping "
            "real requirement content."
        )
        logger.warning(
            "ASVSPageRangeDetectionService: low-confidence requirement range "
            "start_page=%s end_page=%s span=%d page(s)",
            req_start,
            req_end,
            req_end - req_start,
        )
        return req_anchors, "heuristic_low_confidence"

    def _read_toc(self, doc: fitz.Document) -> List[TOCEntry]:
        entries: List[TOCEntry] = []
        for raw in doc.get_toc(simple=False):
            if len(raw) < 3:
                continue
            level, title, page = raw[:3]
            if not title or not page:
                continue
            entries.append(TOCEntry(level=int(level), title=str(title).strip(), page=int(page)))
        return entries

    def _read_pages(self, doc: fitz.Document) -> List[PageInfo]:
        pages: List[PageInfo] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            lines, heading_lines = self._extract_lines(page)
            text = "\n".join(lines)
            pages.append(
                PageInfo(
                    page_number=index + 1,
                    text=text,
                    lines=tuple(lines),
                    heading_lines=tuple(heading_lines),
                )
            )
        return pages

    def _extract_lines(self, page: fitz.Page) -> Tuple[List[str], List[str]]:
        data = page.get_text("dict")
        lines: List[str] = []
        heading_lines: List[str] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                normalized = " ".join(text.split())
                lines.append(normalized)
                max_size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                if max_size >= 11.0:
                    heading_lines.append(normalized)
        return lines, heading_lines

    def _detect_requirement_range(
        self,
        *,
        pages: Sequence[PageInfo],
        toc_entries: Sequence[TOCEntry],
    ) -> Tuple[Optional[int], Optional[int], Dict[str, Any], str]:
        toc_entry = self._find_toc_entry(toc_entries, _TOP_LEVEL_V1_RE)
        if toc_entry:
            start_page = max(1, toc_entry.page - 1)
            appendix_page = self._find_appendix_page(pages, toc_entries, start_after=start_page)
            end_page = self._end_page_from_appendix(start_page, appendix_page)
            return start_page, end_page, {
                "method": "toc_v1_plus_appendix",
                "v1_anchor": toc_entry.title,
                "v1_anchor_page": toc_entry.page,
                "appendix_page": appendix_page,
                "end_page": end_page,
            }, "toc"

        start_page = None
        for page in pages:
            if self._is_contents_page(page):
                continue
            candidates = list(page.heading_lines or page.leading_lines)
            if any(_TOP_LEVEL_V1_RE.match(line) for line in candidates):
                start_page = page.page_number
                break
        if start_page is None:
            return None, None, {"method": "not_found"}, "heuristic"

        appendix_page = self._find_appendix_page(pages, toc_entries, start_after=start_page)
        end_page = self._end_page_from_appendix(start_page, appendix_page)
        return start_page, end_page, {
            "method": "heading_plus_appendix",
            "v1_anchor_page": start_page,
            "appendix_page": appendix_page,
            "end_page": end_page,
        }, "heuristic"

    def _find_toc_entry(self, toc_entries: Sequence[TOCEntry], pattern: re.Pattern[str]) -> Optional[TOCEntry]:
        for entry in toc_entries:
            if pattern.search(entry.title):
                return entry
        return None

    def _find_appendix_page(
        self,
        pages: Sequence[PageInfo],
        toc_entries: Sequence[TOCEntry],
        *,
        start_after: int,
    ) -> Optional[int]:
        toc_entry = self._find_toc_entry(toc_entries, _APPENDIX_A_RE)
        if toc_entry and toc_entry.page > start_after:
            return toc_entry.page
        for page in pages:
            if page.page_number <= start_after:
                continue
            if _APPENDIX_A_RE.search(page.text):
                return page.page_number
        return None

    def _end_page_from_appendix(self, start_page: int, appendix_page: Optional[int]) -> Optional[int]:
        if appendix_page is None:
            return None
        end_page = max(start_page, appendix_page - 1)
        logger.info(
            "ASVSPageRangeDetectionService._end_page_from_appendix: start_page=%d "
            "appendix_page=%d end_page=%d span=%d page(s)",
            start_page,
            appendix_page,
            end_page,
            end_page - start_page,
        )
        return end_page

    def _is_contents_page(self, page: PageInfo) -> bool:
        leading_text = "\n".join(page.leading_lines).lower()
        return bool(_CONTENTS_RE.search(leading_text))

__all__ = [
    "ASVSPageDetectionResult",
    "ASVSPageRangeDetectionService",
]
