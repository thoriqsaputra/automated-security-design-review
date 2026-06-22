from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import fitz


logger = logging.getLogger(__name__)

_TOP_LEVEL_V1_RE = re.compile(r"^V1(?:\s|$)")
_APPENDIX_A_RE = re.compile(r"\bappendix\s+a\b", re.IGNORECASE)
_DEFS_TITLE_RE = re.compile(r"application security verification levels", re.IGNORECASE)
_LEVEL_EVAL_RE = re.compile(r"level evaluation", re.IGNORECASE)
_LEVEL_LABEL_RE = re.compile(r"\blevel\s+[123]\b", re.IGNORECASE)
_HOW_TO_USE_RE = re.compile(r"\bhow to use (?:this standard|the asvs)\b", re.IGNORECASE)
_APPLYING_ASVS_RE = re.compile(r"\bapplying asvs in practice\b", re.IGNORECASE)
_ASSESSMENT_RE = re.compile(r"\bassessment and certification\b", re.IGNORECASE)
_CONTENTS_RE = re.compile(r"\b(?:contents|table of contents)\b", re.IGNORECASE)
_REQ_TABLE_RE = re.compile(r"^\d+\.\d+\.\d+$")
_OPTIONAL_MARK_RE = re.compile(r"^o$", re.IGNORECASE)
_NUMERIC_LEVEL_RE = re.compile(r"^[123]$")
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
class TextSpanInfo:
    text: str
    x0: float
    x1: float
    y0: float


@dataclass(frozen=True)
class PositionedLine:
    y0: float
    spans: Tuple[TextSpanInfo, ...]

    @property
    def text(self) -> str:
        return " ".join(span.text for span in self.spans if span.text).strip()


@dataclass(frozen=True)
class RequirementTableSchema:
    mode: str
    level_x0: Optional[float] = None
    l1_x0: Optional[float] = None
    l2_x0: Optional[float] = None
    l3_x0: Optional[float] = None
    after_l3_x0: Optional[float] = None


@dataclass(frozen=True)
class ASVSPageDetectionResult:
    start_page: Optional[int]
    end_page: Optional[int]
    level_definition_start_page: Optional[int]
    level_definition_end_page: Optional[int]
    source: str
    matched_anchors: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level_definition_start_page": self.level_definition_start_page,
            "level_definition_end_page": self.level_definition_end_page,
            "source": self.source,
            "matched_anchors": self.matched_anchors,
        }


@dataclass(frozen=True)
class ASVSRequirementLevelDetectionResult:
    levels: Dict[str, int]
    source: str
    matched_pages: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "levels": dict(self.levels),
            "source": self.source,
            "matched_pages": self.matched_pages,
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

        defs_start, defs_end, defs_anchors, defs_source = self._detect_definition_range(
            pages=pages,
            toc_entries=toc_entries,
        )
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
        source = "toc" if defs_source == "toc" or req_source == "toc" else "heuristic"
        return ASVSPageDetectionResult(
            start_page=req_start,
            end_page=req_end,
            level_definition_start_page=defs_start,
            level_definition_end_page=defs_end,
            source=source,
            matched_anchors={
                "definition_range": defs_anchors,
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

    def _detect_definition_range(
        self,
        *,
        pages: Sequence[PageInfo],
        toc_entries: Sequence[TOCEntry],
    ) -> Tuple[Optional[int], Optional[int], Dict[str, Any], str]:
        toc_entry = self._find_toc_entry(toc_entries, _DEFS_TITLE_RE)
        if toc_entry:
            start_page = toc_entry.page
            if self._should_backtrack_definition_start(pages, start_page):
                start_page -= 1
            end_page = self._toc_section_end_page(toc_entries, toc_entry)
            if end_page is None:
                end_page = self._fallback_definition_end(pages, start_page)
            return start_page, end_page, {
                "method": "toc_section",
                "definition_anchor": toc_entry.title,
                "definition_anchor_page": toc_entry.page,
                "backtracked_start": start_page != toc_entry.page,
                "end_page": end_page,
            }, "toc"

        start_page = None
        for page in pages:
            if self._is_contents_page(page):
                continue
            if _DEFS_TITLE_RE.search(page.text):
                start_page = page.page_number
                break
        if start_page is None:
            for page in pages:
                if self._is_contents_page(page):
                    continue
                if _LEVEL_EVAL_RE.search(page.text):
                    start_page = page.page_number
                    if self._should_backtrack_definition_start(pages, start_page):
                        start_page -= 1
                    break
        if start_page is None:
            return None, None, {"method": "not_found"}, "heuristic"

        end_page = self._fallback_definition_end(pages, start_page)
        return start_page, end_page, {
            "method": "text_heuristic",
            "definition_anchor_page": start_page,
            "end_page": end_page,
        }, "heuristic"

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

    def _toc_section_end_page(self, toc_entries: Sequence[TOCEntry], entry: TOCEntry) -> Optional[int]:
        for candidate in toc_entries:
            if candidate.page <= entry.page:
                continue
            if candidate.level <= entry.level:
                return candidate.page - 1
        return None

    def _should_backtrack_definition_start(
        self,
        pages: Sequence[PageInfo],
        start_page: int,
    ) -> bool:
        if start_page <= 1:
            return False
        previous = pages[start_page - 2]
        current = pages[start_page - 1]
        if _DEFS_TITLE_RE.search(previous.text) or _LEVEL_EVAL_RE.search(previous.text):
            return False
        if _HOW_TO_USE_RE.search(previous.text) or _ASSESSMENT_RE.search(previous.text):
            return False
        if self._is_contents_page(previous):
            return False
        if not previous.lines:
            return False
        if not (_DEFS_TITLE_RE.search(current.text) or _LEVEL_EVAL_RE.search(current.text)):
            return False
        return True

    def _fallback_definition_end(self, pages: Sequence[PageInfo], start_page: int) -> int:
        for page in pages[start_page:]:
            if _HOW_TO_USE_RE.search(page.text) and self._has_level_definition_content(page):
                continue
            if _HOW_TO_USE_RE.search(page.text) or _APPLYING_ASVS_RE.search(page.text) or _ASSESSMENT_RE.search(page.text):
                return max(start_page, page.page_number - 1)
        return start_page

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
        end_page = max(start_page, appendix_page - 2)
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

    def _has_level_definition_content(self, page: PageInfo) -> bool:
        return bool(_LEVEL_LABEL_RE.search(page.text))


class ASVSRequirementLevelDetectionService:
    def __init__(
        self,
        *,
        get_local_file_path: Callable[[Any], AbstractContextManager],
    ) -> None:
        self._get_local_file_path = get_local_file_path

    def detect(
        self,
        source_doc: Any,
        *,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
    ) -> ASVSRequirementLevelDetectionResult:
        with self._get_local_file_path(source_doc.document) as local_path:
            doc = fitz.open(local_path)
            try:
                first_page = max(1, start_page or 1)
                last_page = min(doc.page_count, end_page or doc.page_count)
                levels: Dict[str, int] = {}
                matched_pages = 0
                active_schema: Optional[RequirementTableSchema] = None
                source = "none"
                for page_number in range(first_page, last_page + 1):
                    page = doc.load_page(page_number - 1)
                    lines = self._extract_positioned_lines(page)
                    schema = self._detect_requirement_table_schema(lines) or active_schema
                    if schema is None:
                        continue
                    page_levels = self._extract_levels_from_page(lines, schema)
                    if page_levels:
                        matched_pages += 1
                        source = "pdf_table_geometry"
                        for logical_id, level in page_levels.items():
                            levels[logical_id] = level
                    active_schema = schema
            finally:
                doc.close()

        return ASVSRequirementLevelDetectionResult(
            levels=levels,
            source=source,
            matched_pages=matched_pages,
        )

    def _extract_positioned_lines(self, page: fitz.Page) -> List[PositionedLine]:
        data = page.get_text("dict")
        raw_lines: List[PositionedLine] = []
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                spans: List[TextSpanInfo] = []
                for span in line.get("spans", []):
                    text = " ".join(str(span.get("text", "")).split()).strip()
                    if not text:
                        continue
                    bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    spans.append(
                        TextSpanInfo(
                            text=text,
                            x0=float(bbox[0]),
                            x1=float(bbox[2]),
                            y0=float(bbox[1]),
                        )
                    )
                if spans:
                    raw_lines.append(
                        PositionedLine(
                            y0=min(span.y0 for span in spans),
                            spans=tuple(sorted(spans, key=lambda item: item.x0)),
                        )
                    )
        return self._coalesce_lines_by_y(raw_lines)

    def _coalesce_lines_by_y(self, lines: Sequence[PositionedLine]) -> List[PositionedLine]:
        merged: List[PositionedLine] = []
        current_spans: List[TextSpanInfo] = []
        current_y: Optional[float] = None
        for line in sorted(lines, key=lambda item: (item.y0, item.spans[0].x0)):
            if current_y is None or abs(line.y0 - current_y) <= 1.0:
                current_spans.extend(line.spans)
                current_y = line.y0 if current_y is None else min(current_y, line.y0)
                continue
            merged.append(
                PositionedLine(
                    y0=current_y,
                    spans=tuple(sorted(current_spans, key=lambda item: item.x0)),
                )
            )
            current_spans = list(line.spans)
            current_y = line.y0
        if current_spans and current_y is not None:
            merged.append(
                PositionedLine(
                    y0=current_y,
                    spans=tuple(sorted(current_spans, key=lambda item: item.x0)),
                )
            )
        return merged

    def _detect_requirement_table_schema(
        self,
        lines: Sequence[PositionedLine],
    ) -> Optional[RequirementTableSchema]:
        for line in lines:
            by_upper = {span.text.upper(): span for span in line.spans}
            if {"#", "DESCRIPTION", "LEVEL"}.issubset(by_upper.keys()):
                return RequirementTableSchema(
                    mode="numeric",
                    level_x0=by_upper["LEVEL"].x0,
                )
            if {"#", "DESCRIPTION", "L1", "L2", "L3"}.issubset(by_upper.keys()):
                after_l3 = by_upper.get("CWE") or by_upper.get("NIST §") or by_upper.get("NIST")
                return RequirementTableSchema(
                    mode="matrix",
                    l1_x0=by_upper["L1"].x0,
                    l2_x0=by_upper["L2"].x0,
                    l3_x0=by_upper["L3"].x0,
                    after_l3_x0=after_l3.x0 if after_l3 else by_upper["L3"].x0 + 40.0,
                )
        return None

    def _extract_levels_from_page(
        self,
        lines: Sequence[PositionedLine],
        schema: RequirementTableSchema,
    ) -> Dict[str, int]:
        levels: Dict[str, int] = {}
        for line in lines:
            logical_id = self._extract_requirement_id(line.text)
            if not logical_id:
                continue
            level = self._infer_level_from_line(line, schema)
            if level is not None:
                levels[logical_id] = level
        return levels

    def _extract_requirement_id(self, text: str) -> Optional[str]:
        first_token = (text or "").strip().split(" ", 1)[0]
        return first_token if _REQ_TABLE_RE.match(first_token) else None

    def _infer_level_from_line(
        self,
        line: PositionedLine,
        schema: RequirementTableSchema,
    ) -> Optional[int]:
        if schema.mode == "numeric":
            if schema.level_x0 is None:
                return None
            for span in line.spans:
                if span.x0 + 2.0 < schema.level_x0:
                    continue
                if _NUMERIC_LEVEL_RE.match(span.text):
                    return int(span.text)
            return None

        marks: Set[int] = set()
        for span in line.spans:
            if _OPTIONAL_MARK_RE.match(span.text):
                if schema.l1_x0 is not None and schema.l2_x0 is not None and schema.l1_x0 - 6.0 <= span.x0 < schema.l2_x0 - 6.0:
                    marks.add(1)
                elif schema.l2_x0 is not None and schema.l3_x0 is not None and schema.l2_x0 - 6.0 <= span.x0 < schema.l3_x0 - 6.0:
                    marks.add(2)
                elif schema.l3_x0 is not None:
                    upper_bound = schema.after_l3_x0 if schema.after_l3_x0 is not None else schema.l3_x0 + 40.0
                    if schema.l3_x0 - 6.0 <= span.x0 < upper_bound:
                        marks.add(3)
        return max(marks) if marks else None


__all__ = [
    "ASVSPageDetectionResult",
    "ASVSPageRangeDetectionService",
    "ASVSRequirementLevelDetectionResult",
    "ASVSRequirementLevelDetectionService",
]
