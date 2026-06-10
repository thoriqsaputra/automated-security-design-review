from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from sdr.apps.ai.client import chat_completion
from .domain_classification import classify_requirement_domain

logger = logging.getLogger(__name__)


class ContractSynthesizer:
    """Rule-first contract synthesis with optional LLM fallback."""

    _FALLBACK_TEMPERATURE = 0.0
    _FALLBACK_MAX_TOKENS = 600
    _CONFIDENCE_LABELS = {
        "very_low": 0.1,
        "very low": 0.1,
        "low": 0.3,
        "medium": 0.5,
        "moderate": 0.5,
        "high": 0.8,
        "very_high": 0.9,
        "very high": 0.9,
    }

    def synthesize(
        self,
        parameter_text: str,
        parameter_section: str,
        parent_description: str = "",
        child_context: str = "",
    ) -> Dict[str, Any]:
        text = (parameter_text or "").strip()
        section = (parameter_section or "General").strip()
        parent_desc = (parent_description or "").strip()
        child_ctx = (child_context or "").strip()
        domain, confidence = self._infer_domain(text, section, parent_desc)
        rule_contract = self._build_rule_contract(
            parameter_text=text,
            parameter_section=section,
            domain=domain,
            confidence=confidence,
        )
        if confidence >= 0.55:
            return rule_contract
        llm_contract = self._llm_fallback(text, section, parent_desc, child_ctx, domain)
        if llm_contract:
            return llm_contract
        return rule_contract

    def _infer_domain(self, *parts: str) -> tuple[str, float]:
        child = parts[0] if len(parts) > 0 else ""
        parent_title = parts[1] if len(parts) > 1 else ""
        parent_desc = parts[2] if len(parts) > 2 else ""
        classification = classify_requirement_domain(
            child_requirement=child,
            parent_title=parent_title,
            parent_description=parent_desc,
        )
        term_count = len(classification.matched_terms.get(classification.primary_domain, []))
        if classification.primary_domain == "general" or term_count == 0:
            return "general", 0.4
        confidence = min(0.95, 0.5 + 0.08 * term_count)
        return classification.primary_domain, confidence

    def _build_rule_contract(
        self,
        parameter_text: str,
        parameter_section: str,
        domain: str,
        confidence: float,
    ) -> Dict[str, Any]:
        lowered = parameter_text.lower()
        domain_then = {
            "architecture_network": "Explicit network architecture/security-control evidence is required.",
            "iam_access_control": "Explicit IAM and access-control implementation evidence is required.",
            "data_crypto_privacy": "Explicit cryptography/data-protection evidence is required.",
            "business_logic_concurrency": "Explicit concurrency-control implementation evidence is required.",
            "transaction_integrity": "Explicit transaction integrity and atomicity evidence is required.",
            "general": "Only explicit, cited evidence may satisfy this requirement.",
        }
        not_sufficient = [
            "Aspirational language without implementation details.",
            "General security claims without control-specific evidence.",
            "Mentions of related concepts without explicit requirement match.",
        ]
        if domain in {"business_logic_concurrency", "transaction_integrity"}:
            not_sufficient.extend(
                [
                    "Requirement restatements without implementation evidence.",
                    "Generic claims about reliability or safety without concrete concurrency controls.",
                    "IAM-only claims without transaction/concurrency safeguards.",
                ]
            )
        return {
            "given": f"A TSD excerpt relevant to section '{parameter_section}'.",
            "when": f"Evaluating requirement: {parameter_text}",
            "then": domain_then.get(domain, domain_then["general"]),
            "not_sufficient": not_sufficient,
            "in_scope": not any(token in lowered for token in ["not applicable", "out of scope"]),
            "specific_enough": len(parameter_text or "") >= 12,
            "domain": domain,
            "confidence": round(confidence, 2),
            "synth_mode": "rule",
        }

    def _llm_fallback(
        self,
        parameter_text: str,
        parameter_section: str,
        parent_description: str,
        child_context: str,
        domain: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            system_prompt = (
                "Generate a strict JSON contract with keys: given, when, then, not_sufficient, "
                "in_scope, specific_enough, domain, confidence, synth_mode."
            )
            user_prompt = (
                f"Section: {parameter_section}\n"
                f"Parent description: {parent_description}\n"
                f"Child context: {child_context}\n"
                f"Requirement: {parameter_text}\n"
                f"Domain hint: {domain}\n"
                "Return JSON only."
            )
            response = chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                component="contract_synthesizer",
                temperature=self._FALLBACK_TEMPERATURE,
                max_tokens=self._FALLBACK_MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            if response.error or not response.content:
                return None
            try:
                parsed = json.loads((response.content or "").strip())
            except json.JSONDecodeError:
                logger.warning("ContractSynthesizer._llm_fallback: JSON decode error. Attempting repair.")
                repair_prompt = (
                    "The following JSON has syntax errors. Fix it and return ONLY valid JSON.\n\n"
                    f"{(response.content or '').strip()}"
                )
                repair_resp = chat_completion(
                    messages=[{"role": "user", "content": repair_prompt}],
                    component="fallback",
                    temperature=0.0,
                    max_tokens=self._FALLBACK_MAX_TOKENS,
                    response_format={"type": "json_object"}
                )
                if repair_resp.error:
                    return None
                try:
                    parsed = json.loads((repair_resp.content or "").strip())
                except json.JSONDecodeError:
                    return None
            if not isinstance(parsed, dict):
                return None
            required = {"given", "when", "then", "not_sufficient"}
            if not required.issubset(parsed.keys()):
                return None
            parsed["domain"] = parsed.get("domain") or domain or "general"
            parsed["confidence"] = self._coerce_confidence(parsed.get("confidence", 0.5))
            parsed["synth_mode"] = "llm_fallback"
            return parsed
        except Exception:
            logger.exception("ContractSynthesizer._llm_fallback failed")
            return None

    def _coerce_confidence(self, value: Any) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in self._CONFIDENCE_LABELS:
                return self._CONFIDENCE_LABELS[normalized]
            try:
                return max(0.0, min(1.0, float(normalized)))
            except ValueError:
                logger.warning(
                    "ContractSynthesizer._coerce_confidence: unknown confidence value %r; using default",
                    value,
                )
                return 0.5
        return 0.5
