"""Deterministic TSD content filtering for security-review indexes."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sdr.core.config import settings

from sdr.apps.ai.tsd_processing.ingestor import DiagramBlock, TextBlock, TSDDocument, TSDPage


_SECURITY_TERMS = (
    "architecture",
    "component",
    "service",
    "api",
    "endpoint",
    "interface",
    "integration",
    "webhook",
    "external",
    "database",
    "data store",
    "storage",
    "cache",
    "queue",
    "worker",
    "batch",
    "deployment",
    "infrastructure",
    "network",
    "kubernetes",
    "container",
    "auth",
    "authentication",
    "authorization",
    "authorisation",
    "identity",
    "session",
    "oauth",
    "oidc",
    "jwt",
    "sso",
    "mfa",
    "encrypt",
    "encryption",
    "tls",
    "mtls",
    "certificate",
    "secret",
    "vault",
    "credential",
    "data flow",
    "dfd",
    "sequence",
    "diagram",
    "assumption",
    "constraint",
    "scope",
    "out of scope",
    "admin console",
    "management console",
    "file upload",
    "mobile",
    "browser",
    "frontend",
)

_EXPLICIT_SCOPE_PATTERNS = (
    re.compile(r"\bno\s+(mobile|android|ios|browser|web\s*ui|frontend|session|file\s+upload|external|third[-\s]party)\b", re.I),
    re.compile(r"\bwithout\s+(mobile|android|ios|browser|web\s*ui|frontend|session|file\s+upload|external|third[-\s]party)\b", re.I),
    re.compile(r"\b(api[-\s]?only|backend[-\s]?only|service[-\s]?only)\b", re.I),
    re.compile(r"\bout\s+of\s+scope\b", re.I),
    re.compile(r"\bnot\s+(in\s+scope|supported|implemented|applicable)\b", re.I),
)

_ADMIN_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("revision_history", re.compile(r"\b(revision|version|change)\s+(history|log)\b|\bchangelog\b|\brevision\s+date\b", re.I)),
    ("approval_signature", re.compile(r"\b(approval|approved\s+by|signature|sign[-\s]?off|reviewed\s+by)\b", re.I)),
    ("document_control", re.compile(r"\b(document\s+(control|owner|metadata|status)|classification:\s*|prepared\s+by|authored\s+by|distribution\s+list)\b", re.I)),
    ("table_of_contents", re.compile(r"\btable\s+of\s+contents\b|\.{4,}\s*\d{1,3}\b", re.I)),
    ("glossary", re.compile(r"\b(glossary|acronyms?|abbreviations?)\b", re.I)),
    ("legal_notice", re.compile(r"\b(confidential|copyright|all\s+rights\s+reserved|legal\s+disclaimer|proprietary)\b", re.I)),
    ("references", re.compile(r"\b(references|bibliography)\b", re.I)),
)

_OCR_GARBAGE_RE = re.compile(r"^[\W\d_]+$")
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d{1,4}(\s+of\s+\d{1,4})?$", re.I)


@dataclass(frozen=True)
class ContentFilterDecision:
    include: bool
    content_class: str
    security_signal_score: int = 0
    drop_reason: Optional[str] = None
    matched_terms: Tuple[str, ...] = ()


@dataclass
class FilteredTSDDocumentView:
    document_name: str
    pages: List[TSDPage] = field(default_factory=list)
    included_block_ids: List[str] = field(default_factory=list)
    excluded_block_ids: List[str] = field(default_factory=list)
    decisions_by_block_id: Dict[str, ContentFilterDecision] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.all_text for page in self.pages if page.all_text.strip())


def content_filter_enabled() -> bool:
    return bool(getattr(settings, "AI_TSD_CONTENT_FILTER_ENABLED", True))


def build_filtered_tsd_view(tsd_document: TSDDocument) -> FilteredTSDDocumentView:
    """Return a lightweight filtered view without mutating the ingested document."""
    if not content_filter_enabled():
        valid_blocks = [b for b in tsd_document.all_text_blocks if b.is_valid()]
        return FilteredTSDDocumentView(
            document_name=tsd_document.document_name,
            pages=list(tsd_document.pages),
            included_block_ids=[b.block_id for b in valid_blocks],
            stats={
                "filter_enabled": False,
                "original_pages": len(tsd_document.pages),
                "included_pages": len(tsd_document.pages),
                "excluded_pages": 0,
                "original_blocks": len(tsd_document.all_text_blocks),
                "included_blocks": len(valid_blocks),
                "excluded_blocks": 0,
                "excluded_by_class": {},
            },
        )

    min_score = _configured_min_score()
    filtered_pages: List[TSDPage] = []
    included_block_ids: List[str] = []
    excluded_block_ids: List[str] = []
    decisions_by_block_id: Dict[str, ContentFilterDecision] = {}
    excluded_by_class: Counter[str] = Counter()
    original_valid_blocks = 0

    for page in tsd_document.pages:
        page_context = " ".join(
            part
            for part in (
                page.section_heading or "",
                _first_line(page.markdown_text),
                _first_line(page.raw_text),
            )
            if part
        )
        page_admin_class = _admin_class(page_context)
        page_terms = _matched_security_terms(page_context)
        page_has_scope_signal = _has_explicit_scope_signal(page_context)
        included_blocks: List[TextBlock] = []

        for block in page.text_blocks:
            if not block.is_valid():
                continue
            original_valid_blocks += 1
            decision = decide_text_block(
                block,
                page_heading=page.section_heading,
                page_admin_class=page_admin_class,
                page_terms=page_terms,
                page_has_scope_signal=page_has_scope_signal,
                min_score=min_score,
            )
            decisions_by_block_id[block.block_id] = decision
            if decision.include:
                included_blocks.append(block)
                included_block_ids.append(block.block_id)
            else:
                excluded_block_ids.append(block.block_id)
                excluded_by_class[decision.content_class] += 1

        diagram_context = _diagram_context(page.diagrams)
        include_diagrams = bool(diagram_context and _should_include_text(diagram_context, None, min_score).include)
        filtered_text = _build_filtered_page_text(page, included_blocks, diagram_context if include_diagrams else "")
        if filtered_text.strip():
            filtered_pages.append(
                replace(
                    page,
                    text_blocks=included_blocks,
                    diagrams=list(page.diagrams) if include_diagrams or included_blocks else [],
                    markdown_text=filtered_text,
                    raw_text=filtered_text,
                )
            )

    stats = {
        "filter_enabled": True,
        "filter_mode": getattr(settings, "AI_TSD_CONTENT_FILTER_MODE", "conservative"),
        "filter_min_score": min_score,
        "original_pages": len(tsd_document.pages),
        "included_pages": len(filtered_pages),
        "excluded_pages": max(0, len(tsd_document.pages) - len(filtered_pages)),
        "original_blocks": len(tsd_document.all_text_blocks),
        "original_valid_blocks": original_valid_blocks,
        "included_blocks": len(included_block_ids),
        "excluded_blocks": len(excluded_block_ids),
        "excluded_by_class": dict(sorted(excluded_by_class.items())),
    }
    return FilteredTSDDocumentView(
        document_name=tsd_document.document_name,
        pages=filtered_pages,
        included_block_ids=included_block_ids,
        excluded_block_ids=excluded_block_ids,
        decisions_by_block_id=decisions_by_block_id,
        stats=stats,
    )


def iter_filtered_scope_parts(tsd_document: TSDDocument) -> Iterable[str]:
    view = build_filtered_tsd_view(tsd_document)
    if getattr(tsd_document, "document_name", None):
        yield str(tsd_document.document_name)
    for page in view.pages:
        if getattr(page, "section_heading", None):
            yield str(page.section_heading)
        if page.all_text.strip():
            yield page.all_text
        for diagram in getattr(page, "diagrams", []) or []:
            if getattr(diagram, "caption", None):
                yield str(diagram.caption)
            if getattr(diagram, "surrounding_text", None):
                yield str(diagram.surrounding_text)


def decide_text_block(
    block: TextBlock,
    *,
    page_heading: Optional[str] = None,
    page_admin_class: Optional[str] = None,
    page_terms: Sequence[str] = (),
    page_has_scope_signal: bool = False,
    min_score: Optional[int] = None,
) -> ContentFilterDecision:
    text = block.text or ""
    heading = block.section_heading or page_heading
    effective_min_score = _configured_min_score() if min_score is None else min_score
    return _should_include_text(
        text,
        heading,
        effective_min_score,
        inherited_admin_class=page_admin_class,
        inherited_terms=page_terms,
        inherited_scope_signal=page_has_scope_signal,
    )


def _should_include_text(
    text: str,
    heading: Optional[str],
    min_score: int,
    *,
    inherited_admin_class: Optional[str] = None,
    inherited_terms: Sequence[str] = (),
    inherited_scope_signal: bool = False,
) -> ContentFilterDecision:
    normalized = _normalize(text)
    heading_normalized = _normalize(heading or "")
    combined = " ".join(part for part in (heading_normalized, normalized) if part)
    if not combined:
        return ContentFilterDecision(False, "empty", 0, "empty")
    if _PAGE_NUMBER_RE.match(combined) or _OCR_GARBAGE_RE.match(combined):
        return ContentFilterDecision(False, "noise", 0, "page_number_or_ocr_garbage")

    explicit_scope = inherited_scope_signal or _has_explicit_scope_signal(combined)
    terms = tuple(dict.fromkeys([*_matched_security_terms(combined), *inherited_terms]))
    score = len(terms) + (2 if explicit_scope else 0)

    admin_class = _admin_class(combined) or inherited_admin_class
    if admin_class in {"revision_history", "approval_signature", "document_control", "table_of_contents", "glossary", "legal_notice"}:
        if explicit_scope:
            return ContentFilterDecision(True, "explicit_scope", score, matched_terms=terms)
        return ContentFilterDecision(False, admin_class, score, f"excluded_{admin_class}", terms)
    if admin_class == "references" and score < min_score:
        return ContentFilterDecision(False, admin_class, score, "references_without_security_signal", terms)

    if explicit_scope:
        return ContentFilterDecision(True, "explicit_scope", score, matched_terms=terms)
    if score >= min_score:
        return ContentFilterDecision(True, "security_relevant", score, matched_terms=terms)
    return ContentFilterDecision(False, "low_signal", score, "below_security_signal_threshold", terms)


def _configured_min_score() -> int:
    try:
        return max(0, int(getattr(settings, "AI_TSD_CONTENT_FILTER_MIN_SCORE", 1)))
    except (TypeError, ValueError):
        return 1


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _first_line(value: str) -> str:
    for line in (value or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _matched_security_terms(text: str) -> Tuple[str, ...]:
    normalized = _normalize(text)
    return tuple(term for term in _SECURITY_TERMS if term in normalized)


def _has_explicit_scope_signal(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _EXPLICIT_SCOPE_PATTERNS)


def _admin_class(text: str) -> Optional[str]:
    for content_class, pattern in _ADMIN_PATTERNS:
        if pattern.search(text or ""):
            return content_class
    return None


def _diagram_context(diagrams: Sequence[DiagramBlock]) -> str:
    parts: List[str] = []
    for diagram in diagrams or []:
        if getattr(diagram, "caption", None):
            parts.append(str(diagram.caption))
        if getattr(diagram, "surrounding_text", None):
            parts.append(str(diagram.surrounding_text))
    return "\n".join(part for part in parts if part.strip())


def _build_filtered_page_text(page: TSDPage, blocks: Sequence[TextBlock], diagram_context: str = "") -> str:
    if not blocks and not diagram_context.strip():
        return ""
    parts: List[str] = []
    if page.section_heading:
        parts.append(str(page.section_heading).strip())
    parts.extend((block.text or "").strip() for block in blocks if (block.text or "").strip())
    if diagram_context.strip():
        parts.append(diagram_context.strip())
    return "\n\n".join(part for part in parts if part)
