from __future__ import annotations

import json
from types import SimpleNamespace

from sdr.apps.ai.client.base import AIProvider, AIResponse
from sdr.apps.ai.services.analysis.ingestion_service import (
    IngestionService,
    _build_tsd_screening_sample,
)
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument, TSDPage


def _page(page_number: int, heading: str, text: str) -> TSDPage:
    return TSDPage(
        page_number=page_number,
        section_heading=heading,
        markdown_text=text,
        raw_text=text,
    )


def _document(pages: list[TSDPage]) -> TSDDocument:
    return TSDDocument(
        file_path="/tmp/example.pdf",
        document_name="Example TSD",
        pages=pages,
        total_pages=len(pages),
    )


def _response(payload: dict) -> AIResponse:
    return AIResponse(
        content=json.dumps(payload),
        model="test-model",
        provider=AIProvider.LOCAL,
    )


def _service() -> IngestionService:
    return IngestionService(ingestor=SimpleNamespace())


def test_screening_sample_skips_front_matter_and_includes_architecture_pages():
    doc = _document(
        [
            _page(1, "Table of Contents", "Contents\n1 Revision History\n2 Architecture Overview"),
            _page(2, "Revision History", "Version Date Author Change approval status"),
            _page(3, "Glossary", "API means application programming interface"),
            _page(
                4,
                "Architecture Overview",
                "The web application uses an API Gateway, Auth Service, and Customer Database.",
            ),
            _page(
                5,
                "Deployment",
                "The service is deployed in containers behind a network load balancer.",
            ),
        ]
    )

    sample, metadata = _build_tsd_screening_sample(doc)

    assert "Architecture Overview" in sample
    assert "Auth Service" in sample
    assert "Customer Database" in sample
    assert metadata["skipped_front_matter_pages"] >= 2
    assert 4 in metadata["sampled_pages"]


def test_screening_allows_low_confidence_non_tsd(monkeypatch):
    doc = _document([_page(1, "Overview", "This document has limited architecture detail.")])

    monkeypatch.setattr(
        "sdr.apps.ai.services.analysis.ingestion_service.chat_completion",
        lambda **kwargs: _response(
            {
                "is_tsd": False,
                "confidence": 0.55,
                "document_type": "unknown",
                "reasoning": "The evidence is ambiguous.",
            }
        ),
    )

    is_valid, reason = _service()._screen_tsd(doc)

    assert is_valid is True
    assert reason is None


def test_screening_rejects_clear_high_confidence_non_tsd(monkeypatch):
    doc = _document([_page(1, "Terms", "This legal contract defines commercial payment terms.")])

    monkeypatch.setattr(
        "sdr.apps.ai.services.analysis.ingestion_service.chat_completion",
        lambda **kwargs: _response(
            {
                "is_tsd": False,
                "confidence": 0.96,
                "document_type": "legal contract",
                "reasoning": "The document is a contract, not a software design.",
            }
        ),
    )

    is_valid, reason = _service()._screen_tsd(doc)

    assert is_valid is False
    assert "contract" in reason


def test_screening_accepts_tsd_evidence_after_toc(monkeypatch):
    doc = _document(
        [
            _page(1, "Table of Contents", "Contents " * 500),
            _page(2, "Revision History", "Version author approved " * 200),
            _page(
                8,
                "API Architecture",
                "The API Gateway routes requests to Auth Service and Order Service over HTTPS.",
            ),
        ]
    )
    captured_prompt = {}

    def fake_completion(**kwargs):
        captured_prompt["prompt"] = kwargs["messages"][1]["content"]
        return _response(
            {
                "is_tsd": True,
                "confidence": 0.88,
                "document_type": "technical software design",
                "reasoning": "The excerpts describe software APIs and services.",
            }
        )

    monkeypatch.setattr(
        "sdr.apps.ai.services.analysis.ingestion_service.chat_completion",
        fake_completion,
    )

    is_valid, reason = _service()._screen_tsd(doc)

    assert is_valid is True
    assert reason is None
    assert "API Gateway routes requests" in captured_prompt["prompt"]
    assert "first 3000" not in captured_prompt["prompt"]


def test_screening_api_error_allows_document(monkeypatch):
    doc = _document([_page(1, "Policy", "This may be ambiguous.")])

    monkeypatch.setattr(
        "sdr.apps.ai.services.analysis.ingestion_service.chat_completion",
        lambda **kwargs: AIResponse(
            content="",
            model="test-model",
            provider=AIProvider.LOCAL,
            error="timeout",
        ),
    )

    is_valid, reason = _service()._screen_tsd(doc)

    assert is_valid is True
    assert reason is None
