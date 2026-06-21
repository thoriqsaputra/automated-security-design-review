"""Deterministic TSD content filtering for security-review indexes."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sdr.core.config import settings

from sdr.apps.ai.client import get_embeddings
from sdr.apps.ai.tsd_processing.document_models import DiagramBlock, TextBlock, TSDDocument, TSDPage

logger = logging.getLogger(__name__)


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
    ("list_of_figures", re.compile(r"\b(list\s+of\s+(figures|diagrams|images)|daftar\s+gambar)\b", re.I)),
    ("list_of_tables", re.compile(r"\b(list\s+of\s+tables|daftar\s+tabel)\b", re.I)),
    ("glossary", re.compile(r"\b(glossary|acronyms?|abbreviations?)\b", re.I)),
    ("legal_notice", re.compile(r"\b(confidential|copyright|all\s+rights\s+reserved|legal\s+disclaimer|proprietary)\b", re.I)),
    ("references", re.compile(r"\b(references|bibliography)\b", re.I)),
)

_OCR_GARBAGE_RE = re.compile(r"^[\W\d_]+$")
_PAGE_NUMBER_RE = re.compile(r"^(page\s*)?\d{1,4}(\s+of\s+\d{1,4})?$", re.I)
_LEADER_DOT_RE = re.compile(r"[.\u2024\u2025\u2026\u2027\u2219\u22ef\u00b7]{4,}")
_OUTLINE_ENTRY_RE = re.compile(
    r"\b(?:figure|fig\.?|diagram|table|appendix|lampiran|gambar|tabel)\s*[a-z0-9.-]*\b.*"
    r"(?:[.\u2024\u2025\u2026\u2027\u2219\u22ef\u00b7]{4,}|\s{3,})\s*\d{1,4}\b",
    re.I,
)
_EMBEDDING_GATE_EXEMPLARS = (
    "The system architecture consists of multiple services that communicate over internal APIs.",
    "Users authenticate to the application using a username and password or single sign-on.",
    "Session tokens are issued after login and expire after a period of inactivity.",
    "User passwords are hashed and salted before being stored in the database.",
    "Sensitive data is encrypted before being written to persistent storage.",
    "Customer data is replicated across multiple availability zones for redundancy.",
    "The application exchanges data with external third-party systems over the network.",
    "The service is deployed inside a container orchestration platform such as Kubernetes.",
    "Secrets and credentials are stored in a dedicated secrets management system.",
    "Administrators can access a separate console to manage system configuration.",
    "Users can upload files which are processed and stored by the backend service.",
    "This feature is not available to mobile or browser-based clients.",
)
_UNICODE_PUNCT_TRANSLATION = str.maketrans({
    "\u2024": ".",
    "\u2025": ".",
    "\u2026": ".",
    "\u2027": ".",
    "\u2219": ".",
    "\u22ef": ".",
    "\u00b7": ".",
})


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
    original_valid_blocks = 0

    # Pass 1 — lexical decisions only, grouped by page (logic unchanged).
    page_block_decisions: List[Tuple[TSDPage, List[Tuple[TextBlock, ContentFilterDecision]]]] = []
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
        block_decisions: List[Tuple[TextBlock, ContentFilterDecision]] = []

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
            block_decisions.append((block, decision))

        page_block_decisions.append((page, block_decisions))

    # Rescue step — semantic second chance for blocks dropped purely for
    # lacking literal keyword overlap. Never overrides admin-class vetoes,
    # noise, or other deliberate exclusions.
    embedding_rescued_blocks = 0
    if _embedding_gate_enabled():
        rescue_candidates = [
            (block.block_id, block, decision)
            for _page, block_decisions in page_block_decisions
            for block, decision in block_decisions
            if not decision.include and decision.drop_reason == "below_security_signal_threshold"
        ]
        rescued_decisions = _rescue_low_signal_blocks(rescue_candidates)
        if rescued_decisions:
            embedding_rescued_blocks = len(rescued_decisions)
            page_block_decisions = [
                (
                    page,
                    [
                        (block, rescued_decisions.get(block.block_id, decision))
                        for block, decision in block_decisions
                    ],
                )
                for page, block_decisions in page_block_decisions
            ]

    # Pass 2 — assemble the filtered view from the (possibly rescued) decisions.
    filtered_pages: List[TSDPage] = []
    included_block_ids: List[str] = []
    excluded_block_ids: List[str] = []
    decisions_by_block_id: Dict[str, ContentFilterDecision] = {}
    excluded_by_class: Counter[str] = Counter()

    for page, block_decisions in page_block_decisions:
        included_blocks: List[TextBlock] = []
        for block, decision in block_decisions:
            decisions_by_block_id[block.block_id] = decision
            if decision.include:
                included_blocks.append(block)
                included_block_ids.append(block.block_id)
            else:
                excluded_block_ids.append(block.block_id)
                excluded_by_class[decision.content_class] += 1

        diagram_context = _filtered_diagram_context(page.diagrams, min_score=min_score)
        include_diagrams = bool(diagram_context)
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
        "embedding_rescued_blocks": embedding_rescued_blocks,
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

    own_admin_class = _admin_class(combined)
    admin_class = own_admin_class or inherited_admin_class
    if admin_class in {
        "revision_history",
        "approval_signature",
        "document_control",
        "table_of_contents",
        "list_of_figures",
        "list_of_tables",
        "glossary",
        "legal_notice",
    }:
        if explicit_scope:
            return ContentFilterDecision(True, "explicit_scope", score, matched_terms=terms)
        if own_admin_class is None and score >= min_score:
            # admin_class came only from page-level inheritance (e.g. a noisy
            # first line) — don't let it veto a block with its own independent
            # security signal.
            return ContentFilterDecision(True, "security_relevant", score, matched_terms=terms)
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


def _embedding_gate_enabled() -> bool:
    return bool(getattr(settings, "AI_TSD_CONTENT_FILTER_EMBEDDING_GATE_ENABLED", True))


def _embedding_similarity_threshold() -> float:
    try:
        return float(getattr(settings, "AI_TSD_CONTENT_FILTER_EMBEDDING_SIMILARITY_THRESHOLD", 0.55))
    except (TypeError, ValueError):
        return 0.55


def _embedding_batch_size() -> int:
    try:
        return max(1, int(getattr(settings, "AI_TSD_CONTENT_FILTER_EMBEDDING_BATCH_SIZE", 32)))
    except (TypeError, ValueError):
        return 32


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def _embed_texts_batched(texts: Sequence[str]) -> List[List[float]]:
    batch_size = _embedding_batch_size()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        try:
            batch_vectors = get_embeddings(texts=batch) or []
        except Exception as exc:
            logger.warning(
                "content_filter._embed_texts_batched: embedding call failed for batch of %d text(s): %s",
                len(batch),
                exc,
            )
            batch_vectors = []
        if len(batch_vectors) != len(batch):
            batch_vectors = [[] for _ in batch]
        vectors.extend(batch_vectors)
    return vectors


def _rescue_low_signal_blocks(
    candidates: List[Tuple[str, TextBlock, ContentFilterDecision]],
) -> Dict[str, ContentFilterDecision]:
    if not candidates:
        return {}

    exemplar_vectors = [v for v in _embed_texts_batched(_EMBEDDING_GATE_EXEMPLARS) if v]
    if not exemplar_vectors:
        return {}

    candidate_vectors = _embed_texts_batched([block.text or "" for _, block, _ in candidates])
    threshold = _embedding_similarity_threshold()
    rescued: Dict[str, ContentFilterDecision] = {}

    for (block_id, _block, decision), vector in zip(candidates, candidate_vectors):
        if not vector:
            continue
        best_similarity = max(_cosine_similarity(vector, exemplar) for exemplar in exemplar_vectors)
        if best_similarity >= threshold:
            rescued[block_id] = replace(decision, include=True, content_class="embedding_relevant")

    return rescued


def _normalize(value: str) -> str:
    value = (value or "").translate(_UNICODE_PUNCT_TRANSLATION)
    return re.sub(r"\s+", " ", value.strip().lower())


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
    normalized = _normalize(text)
    if _looks_like_outline_entry(normalized):
        return "table_of_contents"
    for content_class, pattern in _ADMIN_PATTERNS:
        if pattern.search(normalized):
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


def _filtered_diagram_context(diagrams: Sequence[DiagramBlock], *, min_score: int) -> str:
    parts: List[str] = []
    for diagram in diagrams or []:
        for candidate in (
            str(getattr(diagram, "caption", "") or "").strip(),
            str(getattr(diagram, "surrounding_text", "") or "").strip(),
        ):
            if not candidate:
                continue
            if _should_include_text(candidate, None, min_score).include:
                parts.append(candidate)
    return "\n".join(parts)


def _looks_like_outline_entry(text: str) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    if _LEADER_DOT_RE.search(normalized) and re.search(r"\d{1,4}\b", normalized):
        return True
    return bool(_OUTLINE_ENTRY_RE.search(normalized))


def _build_filtered_page_text(page: TSDPage, blocks: Sequence[TextBlock], diagram_context: str = "") -> str:
    if not blocks and not diagram_context.strip():
        return ""
    parts: List[str] = []
    if page.section_heading and _admin_class(page.section_heading) is None:
        parts.append(str(page.section_heading).strip())
    parts.extend((block.text or "").strip() for block in blocks if (block.text or "").strip())
    if diagram_context.strip():
        parts.append(diagram_context.strip())
    return "\n\n".join(part for part in parts if part)
