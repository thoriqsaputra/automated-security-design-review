from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from sdr.core.config import settings

from sdr.apps.standards.utils import build_parameter_analysis_text

from sdr.apps.ai.engine.preparation.contract_synthesizer import ContractSynthesizer
from sdr.apps.ai.engine.classification.domain_classification import DOMAIN_KEYWORDS, classify_requirement_domain
from sdr.apps.ai.engine.dto import DebateInput
from sdr.apps.ai.retrieval.searchers.graph import _extract_keywords

logger = logging.getLogger(__name__)


class DebateInputFactory:
    def __init__(self, contract_synthesizer: Optional[ContractSynthesizer] = None) -> None:
        self.contract_synthesizer = contract_synthesizer or ContractSynthesizer()

    def build_contract(
        self,
        *,
        parameter_text: str,
        parameter_section: str,
        parent_description: str = "",
    ) -> dict:
        return self.contract_synthesizer.synthesize(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            parent_description=parent_description,
        )

    def build_retrieval_query_details(self, parameter, contract: Optional[dict] = None) -> dict:
        def _to_text(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, (list, tuple)):
                parts = [str(item).strip() for item in value if str(item).strip()]
                return " ".join(parts).strip()
            return str(value).strip()

        def _to_text_list(value) -> list:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return [str(item).strip() for item in value if str(item).strip()]
            text = _to_text(value)
            return [text] if text else []

        parent = getattr(parameter, "parent", None)
        parameter_text = build_parameter_analysis_text(parameter).strip()
        contract_domain = _to_text((contract or {}).get("domain"))
        if contract_domain:
            classification = None
            domain = contract_domain
        else:
            classification = classify_requirement_domain(
                child_requirement=parameter_text,
                parent_title=(getattr(parent, "title", "") or "").strip(),
                parent_description=(getattr(parent, "description", "") or "").strip(),
                extra_parts=[_to_text((contract or {}).get("then"))],
            )
            domain = classification.primary_domain or "general"
        domain_keywords = list(DOMAIN_KEYWORDS.get(domain, DOMAIN_KEYWORDS["general"]))
        return {
            "parent_title": (getattr(parent, "title", "") or "").strip(),
            "parent_description": (getattr(parent, "description", "") or "").strip(),
            "child_requirement": parameter_text,
            "contract_then": _to_text((contract or {}).get("then")),
            "contract_not_sufficient": _to_text_list((contract or {}).get("not_sufficient")),
            "domain_keywords": domain_keywords,
            "domain_signal": domain,
            "primary_domain": classification.primary_domain if classification else domain,
            "secondary_domains": classification.secondary_domains if classification else [],
            "domain_classification_reason": classification.reason if classification else "contract_domain",
            "matched_domain_terms": classification.matched_terms if classification else [],
            "generated_domain_keywords": domain_keywords,
            "retry_queries": [],
        }

    def build_debate_input(
        self,
        *,
        parameter,
        category,
        retrieval_result,
        tsd_document,
        killed_assumptions: List[Dict[str, Any]],
        contract: Optional[dict] = None,
        retrieval_query_details: Optional[dict] = None,
    ) -> DebateInput:
        parameter_text = build_parameter_analysis_text(parameter).strip()
        parameter_section = parameter.parent.title if parameter.parent else "General"
        if contract is None:
            contract = self.build_contract(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                parent_description=(parameter.parent.description if parameter.parent else "") or "",
            )
        if retrieval_query_details is None:
            retrieval_query_details = self.build_retrieval_query_details(parameter, contract)
        retrieval_metadata = dict(getattr(retrieval_result, "evidence_metadata", {}) or {})
        if retrieval_metadata:
            retrieval_query_details = {
                **retrieval_query_details,
                "retrieval_evidence_metadata": retrieval_metadata,
            }
        supplemental_block_limit = max(0, int(getattr(settings, "AI_DEBATE_CONTEXT_SUPPLEMENTAL_BLOCK_LIMIT", 0)))
        chunk_block_ids = getattr(retrieval_result, "context_chunk_block_ids", []) or []
        debate_context_chunks = self.build_xml_context_chunks(
            retrieval_result.context_chunks or [],
            retrieval_metadata=retrieval_metadata,
            tsd_document=tsd_document,
            source_block_ids=getattr(retrieval_result, "source_block_ids", []) or [],
            include_source_blocks=supplemental_block_limit > 0,
            query_text=parameter_text,
            chunk_block_ids=chunk_block_ids,
        )
        context_chunk_map = self.build_context_chunk_map(
            retrieval_result.context_chunks or [],
            retrieval_metadata=retrieval_metadata,
            tsd_document=tsd_document,
            source_block_ids=getattr(retrieval_result, "source_block_ids", []) or [],
            include_source_blocks=supplemental_block_limit > 0,
            source_block_limit=supplemental_block_limit,
            query_text=parameter_text,
            chunk_block_ids=chunk_block_ids,
        )
        logger.info(
            "DebateInputFactory.build_debate_input: parameter id=%s retrieval_chunks=%d source_block_ids=%d prompt_chunks=%d chunk_map_entries=%d",
            getattr(parameter, "id", None),
            len(retrieval_result.context_chunks or []),
            len(getattr(retrieval_result, "source_block_ids", []) or []),
            len(debate_context_chunks),
            len(context_chunk_map),
        )
        return DebateInput(
            parameter=parameter,
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            contract=contract,
            hunter_plan={},
            retrieval_query_details=retrieval_query_details,
            killed_assumptions=list(killed_assumptions),
            context_chunks=debate_context_chunks,
            context_chunk_map=context_chunk_map,
        )

    def build_context_chunk_map(
        self,
        context_chunks: list,
        retrieval_metadata: Optional[dict] = None,
        tsd_document=None,
        source_block_ids: Optional[list] = None,
        include_source_blocks: bool = True,
        source_block_limit: Optional[int] = None,
        query_text: str = "",
        chunk_block_ids: Optional[List[List[str]]] = None,
    ) -> dict:
        chunk_map = {}
        evidence_quality = (retrieval_metadata or {}).get("evidence_quality") or {}
        block_source_map = (retrieval_metadata or {}).get("block_source_map") or {}
        for idx, chunk in enumerate(context_chunks, start=1):
            evidence_kind = self.classify_context_chunk_text(chunk)
            chunk_id = f"graph_summary_{idx}" if evidence_kind == "graph_summary" else f"chunk_{idx}"
            real_block_ids = (
                chunk_block_ids[idx - 1] if chunk_block_ids and idx - 1 < len(chunk_block_ids) else []
            )
            real_block_ids = [bid for bid in (real_block_ids or []) if bid]
            if real_block_ids:
                source_location = self.resolve_chunk_source_location(real_block_ids[0], tsd_document)
            else:
                source_location = self.resolve_chunk_source_location(chunk_id, tsd_document)
            chunk_map[chunk_id] = {
                "source": "retrieval_context",
                "section": source_location.get("section") or "unknown",
                "text": chunk,
                "evidence_kind": evidence_kind,
                "citation_grade": bool(real_block_ids),
                "evidence_quality": evidence_quality,
                **({"block_ids": real_block_ids} if real_block_ids else {}),
                **source_location,
            }
        if include_source_blocks:
            limit = max(0, int(source_block_limit)) if source_block_limit is not None else None
            candidate_limit = max(
                limit or 0,
                int(getattr(settings, "AI_DEBATE_SUPPLEMENTAL_BLOCK_CANDIDATE_LIMIT", 32)),
            )
            keywords = [kw.lower() for kw in _extract_keywords(query_text or "")]
            resolved: List[Dict[str, Any]] = []
            for idx, block_id in enumerate(source_block_ids or []):
                if len(resolved) >= candidate_limit:
                    break
                if not block_id or block_id in chunk_map or "_d" in block_id:
                    continue
                source_location = self.resolve_chunk_source_location(block_id, tsd_document)
                text = ""
                provenance = (block_source_map.get(block_id) if isinstance(block_source_map, dict) else None) or {}
                try:
                    if tsd_document is not None and hasattr(tsd_document, "get_block_window"):
                        window = tsd_document.get_block_window(
                            block_id,
                            before=int(getattr(settings, "AI_DEBATE_SUPPLEMENTAL_BLOCK_WINDOW_BEFORE", 1)),
                            after=int(getattr(settings, "AI_DEBATE_SUPPLEMENTAL_BLOCK_WINDOW_AFTER", 1)),
                            char_budget=int(getattr(settings, "AI_DEBATE_SUPPLEMENTAL_BLOCK_CHAR_BUDGET", 2200)),
                        )
                        if window:
                            text = window["text"]
                            wx0, wy0, wx1, wy1 = window["bbox"]
                            # Override with the UNION bbox spanning every merged
                            # block, not just the originally matched block's single
                            # line — otherwise the PDF viewer highlights only one
                            # line of a multi-sentence quoted passage.
                            source_location = {
                                **source_location,
                                "page": window["page_number"],
                                "page_number": window["page_number"],
                                "bbox": {"x0": wx0, "y0": wy0, "x1": wx1, "y1": wy1},
                                "bbox_x0": wx0,
                                "bbox_y0": wy0,
                                "bbox_x1": wx1,
                                "bbox_y1": wy1,
                            }
                    elif tsd_document is not None:
                        block = tsd_document.get_block_by_id(block_id)
                        text = getattr(block, "text", "") or ""
                except Exception:
                    logger.warning(
                        "DebateInputFactory.build_context_chunk_map: failed to resolve block_id=%s",
                        block_id,
                        exc_info=True,
                    )
                    text = ""
                if not text:
                    continue
                lowered_text = text.lower()
                score = sum(1 for keyword in keywords if keyword in lowered_text)
                resolved.append(
                    {
                        "block_id": block_id,
                        "score": score,
                        "order": idx,
                        "text": text,
                        "source_location": source_location,
                        "provenance": provenance,
                    }
                )
            resolved.sort(key=lambda item: (-item["score"], item["order"]))
            if limit is not None:
                resolved = resolved[:limit]
            for item in resolved:
                source_location = item["source_location"]
                provenance = item["provenance"]
                chunk_map[item["block_id"]] = {
                    "source": "retrieval_grounded_source",
                    "section": source_location.get("section") or "unknown",
                    "text": item["text"],
                    "evidence_kind": "grounded_text_block",
                    "citation_grade": True,
                    "evidence_quality": evidence_quality,
                    "retrieval_origin": provenance.get("retrieval_origin"),
                    "retrieval_origin_label": provenance.get("retrieval_origin_label"),
                    **source_location,
                }
        return chunk_map

    def resolve_chunk_source_location(self, chunk_id: str, tsd_document) -> Dict[str, Any]:
        if not chunk_id or tsd_document is None:
            return {}
        block = None
        try:
            if "_d" in chunk_id and hasattr(tsd_document, "get_diagram_by_id"):
                block = tsd_document.get_diagram_by_id(chunk_id)
            elif hasattr(tsd_document, "get_block_by_id"):
                block = tsd_document.get_block_by_id(chunk_id)
        except Exception:
            logger.warning(
                "DebateInputFactory.resolve_chunk_source_location: failed to resolve chunk_id=%s",
                chunk_id,
                exc_info=True,
            )
            return {}
        if block is None:
            return {}
        page_number = getattr(block, "page_number", None)
        bbox = {
            "x0": getattr(block, "bbox_x0", None),
            "y0": getattr(block, "bbox_y0", None),
            "x1": getattr(block, "bbox_x1", None),
            "y1": getattr(block, "bbox_y1", None),
        }
        return {
            "page": page_number,
            "page_number": page_number,
            "bbox": bbox,
            "bbox_x0": bbox["x0"],
            "bbox_y0": bbox["y0"],
            "bbox_x1": bbox["x1"],
            "bbox_y1": bbox["y1"],
            "section": getattr(block, "section_heading", None),
        }

    def classify_context_chunk_text(self, chunk: str) -> str:
        text = (chunk or "").strip()
        lowered = text.lower()
        if not text:
            return "empty"
        if text.startswith("--- VECTOR RESULT"):
            return "baseline_requirement"
        if text.startswith("--- GRAPH RESULT") or text.startswith("--- GRAPH PATH") or lowered.startswith("graph node:"):
            return "graph_summary"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(text) < 120 and len(lines) <= 2:
            return "heading_only"
        if re.search(
            r"\b(use|uses|using|implemented|configured|enabled|enforced|validated|verified|required|requires|oauth|oidc|token|jwt|pkce|jwks|mfa|rbac|encrypt|encrypted)\b",
            lowered,
        ):
            return "implementation_or_scope_context"
        return "weak_context"

    def build_xml_context_chunks(
        self,
        context_chunks: list,
        retrieval_metadata: Optional[dict] = None,
        tsd_document=None,
        source_block_ids: Optional[list] = None,
        include_source_blocks: bool = False,
        query_text: str = "",
        chunk_block_ids: Optional[List[List[str]]] = None,
    ) -> list:
        source_block_limit = None
        if include_source_blocks:
            source_block_limit = max(0, int(getattr(settings, "AI_DEBATE_CONTEXT_SUPPLEMENTAL_BLOCK_LIMIT", 0)))
        chunk_map = self.build_context_chunk_map(
            context_chunks,
            retrieval_metadata=retrieval_metadata,
            tsd_document=tsd_document,
            source_block_ids=source_block_ids,
            include_source_blocks=include_source_blocks,
            source_block_limit=source_block_limit,
            query_text=query_text,
            chunk_block_ids=chunk_block_ids,
        )
        xml_chunks = []
        for chunk_id, payload in chunk_map.items():
            attrs = [
                f'id="{chunk_id}"',
                f'source="{payload.get("source", "unknown")}"',
                f'citable="{"true" if payload.get("citation_grade") else "false"}"',
                f'section="{payload.get("section", "unknown")}"',
            ]
            if payload.get("page_number") is not None:
                attrs.append(f'page_number="{payload["page_number"]}"')
            if payload.get("bbox_x0") is not None:
                attrs.append(f'bbox_x0="{payload["bbox_x0"]}"')
                attrs.append(f'bbox_y0="{payload["bbox_y0"]}"')
                attrs.append(f'bbox_x1="{payload["bbox_x1"]}"')
                attrs.append(f'bbox_y1="{payload["bbox_y1"]}"')

            attr_str = " ".join(attrs)
            xml_chunks.append(
                "\n".join(
                    [
                        f"<CONTEXT_CHUNK {attr_str}>",
                        payload.get("text", ""),
                        "</CONTEXT_CHUNK>",
                    ]
                )
            )
        return xml_chunks
