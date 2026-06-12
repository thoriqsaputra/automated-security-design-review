from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sdr.apps.ai.prompts.agent_prompt import (
    VISION_ARCHITECT_SYSTEM_PROMPT,
    VISION_AUDITOR_SYSTEM_PROMPT,
    VISION_CRITIC_SYSTEM_PROMPT,
    build_vision_architect_prompt,
    build_vision_auditor_prompt,
    build_vision_critic_prompt,
)
from .base import (
    BaseAgent,
    VisionResult,
    VERDICT_MET,
    VERDICT_NOT_MET,
    VERDICT_NA,
    VALID_VERDICTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagram input dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiagramInput:

    diagram_id: str  # "p{page}_d{idx}" — matches ingestor output [2]
    image_b64: str  # base64-encoded PNG/JPEG image bytes [2]
    page_number: int  # 1-based page number in the TSD PDF
    caption: Optional[str] = None  # caption or title if extractable
    surrounding_text: Optional[str] = None  # text immediately around the diagram
    image_format: str = "png"  # image format for Bedrock content block [4]

    # Bounding box in PDF coordinate space — enables click-to-source [3]
    bbox_x0: Optional[float] = None
    bbox_y0: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None

    def is_valid(self) -> bool:
        """
        Returns True if this diagram has the minimum required fields
        to be passed to the Vision agent.
        """
        return bool(self.diagram_id) and bool(self.image_b64) and self.page_number > 0

    def decode_image_bytes(self) -> bytes:
        """
        Decodes the base64 image string to raw bytes for the Bedrock
        multimodal content block. Raises ValueError on invalid base64.
        """
        try:
            return base64.b64decode(self.image_b64)
        except Exception as exc:
            raise ValueError(
                f"DiagramInput.decode_image_bytes: failed to decode "
                f"base64 for diagram_id='{self.diagram_id}': {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Vision Agent
# ---------------------------------------------------------------------------


class VisionAgent(BaseAgent):
    """
    Concrete implementation of the Vision agent.

    The Vision agent uses Claude's multimodal capability via the Bedrock
    Converse API [4] to audit architectural diagrams embedded in TSDs.

    It runs once per (diagram, parameter) pair. A single TSD may contain
    multiple diagrams; analysis_service.py is responsible for iterating
    over diagrams and calling run() for each relevant (diagram, parameter)
    combination.

    Key design decisions:
    - Image bytes are sent via Bedrock's native multimodal content block
      format — not as base64 text in the prompt string. This is the correct
      approach for the Converse API [4].
    - The Vision agent is deliberately NOT part of the Hunter/Critic/Mediator
      debate chain. Its findings are stored as separate FINDING_TYPE_DIAGRAM
      records alongside the text-based findings [3].
    - Diagram findings with verdict "na" are still persisted — they document
      that the diagram was audited and found to be out of scope, which is
      valuable audit trail information.

    The Vision agent never raises — all errors are captured in
    VisionResult.error so analysis_service.py can continue to the
    next diagram/parameter pair.
    """

    system_prompt: str = VISION_AUDITOR_SYSTEM_PROMPT
    model_component: str = "vision"

    # Vision responses tend to be more descriptive — allow more tokens
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.9

    # Minimum image dimensions to send to Vision — smaller images are
    # likely icons or logos, not architectural diagrams [2]
    _MIN_IMAGE_BYTES = 512

    def _build_user_prompt(
        self,
        parameter_text: str,
        parameter_section: str,
        diagram: DiagramInput,
    ) -> str:
        """
        Delegates to build_vision_architect_prompt() from agent_prompts.py.
        The image itself is passed separately via _call_llm(image_b64=...).
        The text prompt provides context about the parameter and diagram.
        """
        return build_vision_architect_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            diagram_caption=diagram.caption,
            surrounding_text=diagram.surrounding_text,
        )

    def run(
        self,
        parameter_text: str,
        parameter_section: str,
        diagram: DiagramInput,
    ) -> VisionResult:
        """
        Executes the Vision agent for a single (diagram, parameter) pair.

        Pipeline:
            1. Validate inputs — guard against empty parameter or invalid diagram.
            2. Validate image data — check base64 decoding and minimum size.
            3. Build the text portion of the user-turn prompt.
            4. Call the LLM via _call_llm() with the image bytes attached [4].
            5. Parse the JSON response via _parse_json_response().
            6. Extract and validate all fields with shared helpers.
            7. Return a fully populated VisionResult.

        Args:
            parameter_text:    Full requirement text from CategoryParameterChild [3].
            parameter_section: Parent section title from CategoryParameterParent [3].
            diagram:           DiagramInput instance produced by the TSD ingestor [2].

        Returns:
            VisionResult — never raises. Check .error field for failures.
        """
        # ------------------------------------------------------------------
        # 1. Input validation
        # ------------------------------------------------------------------
        if not parameter_text or not parameter_text.strip():
            msg = "parameter_text is empty — cannot audit diagram against blank requirement."
            self.logger.error("VisionAgent.run: %s", msg)
            return self._vision_error(msg)

        if not diagram.is_valid():
            msg = (
                f"DiagramInput is invalid — diagram_id='{diagram.diagram_id}' "
                f"page={diagram.page_number} image_b64 present={bool(diagram.image_b64)}."
            )
            self.logger.error("VisionAgent.run: %s", msg)
            return self._vision_error(msg)

        # ------------------------------------------------------------------
        # 2. Validate and decode image data
        # ------------------------------------------------------------------
        try:
            image_bytes = diagram.decode_image_bytes()
        except ValueError as exc:
            msg = str(exc)
            self.logger.error("VisionAgent.run: %s", msg)
            return self._vision_error(msg)

        if len(image_bytes) < self._MIN_IMAGE_BYTES:
            msg = (
                f"Image for diagram_id='{diagram.diagram_id}' is too small "
                f"({len(image_bytes)} bytes < {self._MIN_IMAGE_BYTES} minimum) — "
                f"likely an icon or logo, not an architectural diagram."
            )
            self.logger.warning("VisionAgent.run: %s", msg)
            return VisionResult(
                verdict=VERDICT_NA,
                confidence=0.9,
                reasoning=(
                    "Diagram skipped: image is too small to be an architectural "
                    "diagram. Likely an icon or decorative element."
                ),
                logic_summary=(
                    "Diagram skipped: image is too small to be an architectural "
                    "diagram. Likely an icon or decorative element."
                ),
                visual_elements_cited=[],
                missing_controls=[],
                raw_response=None,
                error=None,
            )

        # ------------------------------------------------------------------
        # 3. Architect pass
        # ------------------------------------------------------------------
        architect_prompt = self._build_user_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            diagram=diagram,
        )

        self.logger.info(
            "VisionAgent.run: auditing diagram_id='%s' (p.%d) "
            "against parameter '%s...'",
            diagram.diagram_id,
            diagram.page_number,
            parameter_text[:60],
        )

        architect_response = self._call_vision_stage(
            user_prompt=architect_prompt,
            image_bytes=image_bytes,
            image_format=self._validate_image_format(diagram.image_format),
            stage_name="architect",
            system_prompt=VISION_ARCHITECT_SYSTEM_PROMPT,
        )
        if architect_response is None:
            return self._vision_error(
                f"Architect pass failed for diagram_id='{diagram.diagram_id}'."
            )
        architect_result = self._extract_architect_result(architect_response)

        # ------------------------------------------------------------------
        # 4. Security auditor pass
        # ------------------------------------------------------------------
        auditor_prompt = build_vision_auditor_prompt(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            diagram_caption=diagram.caption,
            surrounding_text=diagram.surrounding_text,
            architect_result=architect_result,
        )
        auditor_response = self._call_vision_stage(
            user_prompt=auditor_prompt,
            image_bytes=image_bytes,
            image_format=self._validate_image_format(diagram.image_format),
            stage_name="auditor",
            system_prompt=VISION_AUDITOR_SYSTEM_PROMPT,
        )
        if auditor_response is None:
            return self._vision_error(
                f"Security auditor pass failed for diagram_id='{diagram.diagram_id}'."
            )

        verdict = self._validate_verdict(
            auditor_response.get("verdict"),
            fallback=VERDICT_NA,
        )
        confidence = self._clamp_confidence(
            auditor_response.get("confidence"),
            default=0.5,
        )
        reasoning = self._extract_vision_reasoning(auditor_response)
        reasoning_fields = self._extract_reasoning_fields(
            auditor_response,
            reasoning_fallback=reasoning,
        )
        visual_elements_cited = self._extract_string_list(
            auditor_response.get("visual_elements_cited", []),
            field_name="visual_elements_cited",
        )
        missing_controls = self._extract_missing_controls(
            auditor_response.get("missing_controls", []),
            verdict=verdict,
        )
        visual_evidence = self._extract_visual_evidence(
            auditor_response.get("visual_evidence", [])
        )
        ambiguous_elements = self._extract_string_list(
            auditor_response.get("ambiguous_elements", []),
            field_name="ambiguous_elements",
        )
        missing_information = self._extract_string_list(
            auditor_response.get("missing_information", []),
            field_name="missing_information",
        )
        ambiguity_reason = self._extract_text_field(
            auditor_response,
            "ambiguity_reason",
            default="",
            max_chars=400,
        )

        # ------------------------------------------------------------------
        # 7. Post-parse consistency checks
        # ------------------------------------------------------------------

        # missing_controls should only be populated for not_met findings
        if verdict != VERDICT_NOT_MET and missing_controls:
            self.logger.debug(
                "VisionAgent.run: clearing missing_controls for "
                "verdict='%s' on diagram_id='%s'.",
                verdict,
                diagram.diagram_id,
            )
            missing_controls = []

        # Warn if met but no visual elements cited — hallucination risk
        if verdict == VERDICT_MET and not visual_elements_cited:
            self.logger.warning(
                "VisionAgent.run: verdict='met' but no visual_elements_cited "
                "for diagram_id='%s' parameter '%s...' — low confidence finding.",
                diagram.diagram_id,
                parameter_text[:60],
            )

        # In ambiguous or under-specified diagrams, prefer NA over forced non-compliance.
        if verdict == VERDICT_NOT_MET and (ambiguous_elements or missing_information):
            self.logger.info(
                "VisionAgent.run: switching verdict to 'na' due to ambiguity for diagram_id='%s'.",
                diagram.diagram_id,
            )
            verdict = VERDICT_NA
            missing_controls = []

        # ------------------------------------------------------------------
        # 8. Optional critic pass
        # ------------------------------------------------------------------
        critic_payload = None
        if self._should_run_vision_critic(
            verdict=verdict,
            confidence=confidence,
            visual_evidence=visual_evidence,
            ambiguous_elements=ambiguous_elements,
            missing_information=missing_information,
        ):
            critic_prompt = build_vision_critic_prompt(
                parameter_text=parameter_text,
                parameter_section=parameter_section,
                architect_result=architect_result,
                auditor_result={
                    "verdict": verdict,
                    "confidence": confidence,
                    "reasoning": reasoning_fields["logic_summary"],
                    "visual_evidence": visual_evidence,
                    "ambiguous_elements": ambiguous_elements,
                    "missing_information": missing_information,
                    "ambiguity_reason": ambiguity_reason,
                },
            )
            critic_response = self._call_vision_stage(
                user_prompt=critic_prompt,
                image_bytes=image_bytes,
                image_format=self._validate_image_format(diagram.image_format),
                stage_name="critic",
                system_prompt=VISION_CRITIC_SYSTEM_PROMPT,
            )
            if critic_response:
                critic_payload = self._extract_critic_payload(critic_response)
                if critic_payload.get("outcome") == "overturn":
                    verdict = self._validate_verdict(
                        critic_payload.get("revised_verdict"),
                        fallback=verdict,
                    )
                    confidence = self._clamp_confidence(
                        critic_payload.get("revised_confidence"),
                        default=confidence,
                    )

        # ------------------------------------------------------------------
        # 9. Log and return
        # ------------------------------------------------------------------
        self.logger.info(
            "VisionAgent.run: verdict='%s' confidence=%.2f "
            "visual_elements=%d missing_controls=%d visual_evidence=%d "
            "for diagram_id='%s' parameter '%s...'",
            verdict,
            confidence,
            len(visual_elements_cited),
            len(missing_controls),
            len(visual_evidence),
            diagram.diagram_id,
            parameter_text[:60],
        )

        return VisionResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning_fields["reasoning"],
            assumptions=reasoning_fields["assumptions"],
            logic_summary=reasoning_fields["logic_summary"],
            cot_trace=reasoning_fields["cot_trace"],
            visual_elements_cited=visual_elements_cited,
            missing_controls=missing_controls,
            visual_evidence=visual_evidence,
            ambiguous_elements=ambiguous_elements,
            missing_information=missing_information,
            ambiguity_reason=ambiguity_reason,
            architect_summary=architect_result,
            auditor_reasoning=reasoning_fields["logic_summary"],
            critic_result=critic_payload,
            raw_response=None,
            error=None,
        )

    # ------------------------------------------------------------------
    # Vision-specific private helpers
    # ------------------------------------------------------------------

    def _extract_vision_reasoning(self, parsed: dict) -> str:
        """
        Extracts and sanitises the reasoning field from the Vision agent's
        parsed response. The reasoning should describe what is visually
        observed in the diagram and how it relates to the parameter.

        Falls back to a generic message if the field is missing or empty
        so downstream code always has a non-null string to persist in
        Finding.vision_reasoning [3].
        """
        raw = parsed.get("reasoning") or ""
        reasoning = str(raw).strip()
        if not reasoning:
            self.logger.debug(
                "VisionAgent._extract_vision_reasoning: reasoning field "
                "missing or empty in LLM response."
            )
            return "No reasoning provided by the Vision agent."
        return reasoning

    def _extract_string_list(
        self,
        raw: object,
        field_name: str,
    ) -> List[str]:
        """
        Extracts a list of non-empty strings from a raw LLM response field.
        Used for both visual_elements_cited and missing_controls.

        Deduplicates while preserving insertion order.
        Returns an empty list on any type mismatch.

        Args:
            raw:        The raw value at the field key in the parsed JSON.
            field_name: Key name — used only in log messages for clarity.

        Returns:
            A deduplicated list of non-empty stripped strings.
        """
        if not isinstance(raw, list):
            self.logger.debug(
                "VisionAgent._extract_string_list: '%s' is not a list "
                "(got %s) — returning [].",
                field_name,
                type(raw).__name__,
            )
            return []

        result: List[str] = []
        seen: set = set()

        for item in raw:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                result.append(cleaned)
                seen.add(cleaned)

        return result

    def _extract_visual_evidence(self, raw: object) -> List[dict]:
        if not isinstance(raw, list):
            return []

        evidence: List[dict] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            element = str(item.get("element", "")).strip()
            why_relevant = str(item.get("why_relevant", "")).strip()
            bbox_raw = item.get("bbox")
            bbox = self._normalize_bbox(bbox_raw)
            if not element and not why_relevant:
                continue
            evidence.append(
                {
                    "element": element,
                    "bbox": bbox,
                    "why_relevant": why_relevant,
                }
            )
        return evidence

    def _normalize_bbox(self, raw: object) -> Optional[List[float]]:
        if not isinstance(raw, list) or len(raw) != 4:
            return None
        normalized: List[float] = []
        for value in raw:
            number = self._safe_float(value)
            if number is None:
                return None
            normalized.append(max(0.0, min(1.0, number)))
        return normalized

    def _call_vision_stage(
        self,
        user_prompt: str,
        image_bytes: bytes,
        image_format: str,
        stage_name: str,
        system_prompt: str,
    ) -> Optional[dict]:
        original_prompt = self.system_prompt
        self.system_prompt = system_prompt
        try:
            response = self._call_llm(
                user_prompt=user_prompt,
                image_b64=image_bytes,
                image_format=image_format,
                top_p=self.top_p,
            )
        finally:
            self.system_prompt = original_prompt

        if response.error:
            self.logger.error("VisionAgent.%s pass error: %s", stage_name, response.error)
            return None
        parsed = self._parse_json_response(response)
        if parsed is None:
            self.logger.error("VisionAgent.%s pass parse error.", stage_name)
            return None
        return parsed

    def _extract_architect_result(self, parsed: dict) -> dict:
        return {
            "diagram_title": self._extract_text_field(parsed, "diagram_title", default=""),
            "components": self._extract_string_list(parsed.get("components", []), "components"),
            "connections": self._extract_string_list(parsed.get("connections", []), "connections"),
            "data_flows": self._extract_string_list(parsed.get("data_flows", []), "data_flows"),
            "trust_boundaries": self._extract_string_list(parsed.get("trust_boundaries", []), "trust_boundaries"),
            "visible_labels": self._extract_string_list(parsed.get("visible_labels", []), "visible_labels"),
            "visible_security_controls": self._extract_string_list(
                parsed.get("visible_security_controls", []), "visible_security_controls"
            ),
            "unclear_regions": self._extract_string_list(parsed.get("unclear_regions", []), "unclear_regions"),
            "visual_evidence": self._extract_visual_evidence(parsed.get("visual_evidence", [])),
            "notes": self._extract_text_field(parsed, "notes", default="", max_chars=600),
        }

    def _should_run_vision_critic(
        self,
        verdict: str,
        confidence: float,
        visual_evidence: List[dict],
        ambiguous_elements: List[str],
        missing_information: List[str],
    ) -> bool:
        if verdict == VERDICT_MET:
            return True
        if confidence < 0.75:
            return True
        if not visual_evidence:
            return True
        if ambiguous_elements or missing_information:
            return True
        return False

    def _extract_critic_payload(self, parsed: dict) -> dict:
        outcome = str(parsed.get("outcome", "")).strip().lower()
        if outcome not in {"uphold", "overturn"}:
            outcome = "uphold"
        return {
            "outcome": outcome,
            "reasoning": self._extract_text_field(parsed, "reasoning", default="", max_chars=1000),
            "hallucinated_or_unsupported_claims": self._extract_string_list(
                parsed.get("hallucinated_or_unsupported_claims", []),
                field_name="hallucinated_or_unsupported_claims",
            ),
            "revised_verdict": self._validate_verdict(parsed.get("revised_verdict"), fallback=VERDICT_NA),
            "revised_confidence": self._clamp_confidence(parsed.get("revised_confidence"), default=0.5),
        }

    def _extract_missing_controls(
        self,
        raw: object,
        verdict: str,
    ) -> List[str]:
        """
        Extracts the missing_controls list from the Vision agent's response.

        Missing controls are only meaningful for not_met findings — if the
        verdict is met or na, an empty list is returned regardless of what
        the LLM produced. The post-parse consistency check in run() enforces
        this as a second line of defence.

        Deduplicates while preserving insertion order.
        Returns an empty list on any type mismatch or empty input.

        Args:
            raw:     The raw value at "missing_controls" in the parsed JSON.
            verdict: The validated verdict string — used to gate the extraction.

        Returns:
            A deduplicated list of non-empty stripped strings, or [] for
            met/na verdicts.
        """
        if verdict != VERDICT_NOT_MET:
            self.logger.debug(
                "VisionAgent._extract_missing_controls: skipping extraction "
                "for verdict='%s' — missing_controls only apply to not_met.",
                verdict,
            )
            return []

        return self._extract_string_list(raw, field_name="missing_controls")

    def _validate_image_format(self, image_format: str) -> str:
        """
        Validates and normalises the image format string for the Bedrock
        multimodal content block.

        Bedrock Converse API [4] accepts: "png", "jpeg", "gif", "webp".
        Defaults to "png" for any unrecognised format — PNG is the safest
        lossless format for architectural diagrams.

        Args:
            image_format: Raw format string from DiagramInput.image_format.

        Returns:
            A lowercase format string accepted by the Bedrock API [4].
        """
        _BEDROCK_SUPPORTED_FORMATS = frozenset({"png", "jpeg", "jpg", "gif", "webp"})
        _FORMAT_NORMALISATION = {
            "jpg": "jpeg",  # Bedrock uses "jpeg" not "jpg"
            "jpeg": "jpeg",
            "png": "png",
            "gif": "gif",
            "webp": "webp",
        }

        normalised = image_format.strip().lower() if image_format else "png"

        if normalised not in _BEDROCK_SUPPORTED_FORMATS:
            self.logger.warning(
                "VisionAgent._validate_image_format: unsupported format '%s' "
                "— defaulting to 'png'.",
                image_format,
            )
            return "png"

        return _FORMAT_NORMALISATION.get(normalised, "png")


# ---------------------------------------------------------------------------
# Batch audit helper — called by analysis_service.py
# ---------------------------------------------------------------------------


def audit_diagrams_for_parameter(
    agent: VisionAgent,
    parameter_text: str,
    parameter_section: str,
    diagrams: List[DiagramInput],
) -> List[tuple[DiagramInput, VisionResult]]:
    if not diagrams:
        return []

    results: List[tuple[DiagramInput, VisionResult]] = []

    for diagram in diagrams:
        if not diagram.is_valid():
            logger.warning(
                "audit_diagrams_for_parameter: skipping invalid "
                "DiagramInput diagram_id='%s' page=%s for parameter '%s...'",
                diagram.diagram_id,
                diagram.page_number,
                parameter_text[:60],
            )
            continue

        vision_result = agent.run(
            parameter_text=parameter_text,
            parameter_section=parameter_section,
            diagram=diagram,
        )
        results.append((diagram, vision_result))

    logger.info(
        "audit_diagrams_for_parameter: completed %d/%d diagram audit(s) "
        "for parameter '%s...'",
        len(results),
        len(diagrams),
        parameter_text[:60],
    )

    return results


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "DiagramInput",
    "VisionAgent",
    "audit_diagrams_for_parameter",
]
