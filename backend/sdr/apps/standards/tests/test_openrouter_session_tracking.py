from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from sdr.apps.ai.client.base import AIProvider, AIResponse
from sdr.apps.ai.client.manager import AIServiceManager
from sdr.apps.ai.client.openrouter.service import OpenRouterAIService
from sdr.apps.ai.client.session import (
    build_standard_ingestion_session_id,
    build_tsd_analysis_session_id,
    get_current_request_metadata,
    merge_request_metadata,
    request_metadata_context,
)
from sdr.apps.ai.engine import pipeline as analysis_pipeline


class OpenRouterSessionTrackingTests(TestCase):
    def test_merge_request_metadata_preserves_active_session_id(self):
        with request_metadata_context({"session_id": "tsd_analysis_review_42", "job_type": "tsd_analysis"}):
            merged = merge_request_metadata({"session_id": "override", "attempt": "2"})

        self.assertEqual(merged["session_id"], "tsd_analysis_review_42")
        self.assertEqual(merged["job_type"], "tsd_analysis")
        self.assertEqual(merged["attempt"], "2")

    def test_ai_service_manager_injects_context_metadata(self):
        captured = {}

        class FakeService:
            default_model = "fake-model"

            def chat_completion(self, **kwargs):
                captured.update(kwargs)
                return AIResponse(
                    content="ok",
                    model=kwargs["model"],
                    provider=AIProvider.OPENROUTER,
                )

        manager = AIServiceManager.__new__(AIServiceManager)

        with request_metadata_context({"session_id": "standard_ingestion_job_11", "job_id": "11"}):
            manager._invoke_service(
                FakeService(),
                AIProvider.OPENROUTER,
                None,
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "metadata": {"session_id": "override", "attempt": "1"},
                },
            )

        self.assertEqual(captured["metadata"]["session_id"], "standard_ingestion_job_11")
        self.assertEqual(captured["metadata"]["job_id"], "11")
        self.assertEqual(captured["metadata"]["attempt"], "1")

    def test_openrouter_service_sends_metadata_in_extra_body(self):
        captured = {}

        class FakeLimiter:
            def acquire(self):
                return None

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    usage=None,
                )

        service = OpenRouterAIService.__new__(OpenRouterAIService)
        service.api_key = "test-key"
        service.default_model = "test-model"
        service.fast_model = "fast-model"
        service.rate_limiter = FakeLimiter()
        service.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        response = service.chat_completion(
            messages=[{"role": "user", "content": "hello"}],
            model="test-model",
            metadata={"session_id": "tsd_analysis_review_9", "job_id": "9"},
        )

        self.assertIsNone(response.error)
        self.assertEqual(captured["extra_body"]["metadata"]["session_id"], "tsd_analysis_review_9")
        self.assertEqual(captured["extra_body"]["metadata"]["job_id"], "9")

    def test_run_tsd_analysis_sets_job_session_context(self):
        observed = {}

        class FakeSummary:
            def to_dict(self):
                return {"ok": True}

        class FakePipeline:
            def run(self, review):
                observed.update(get_current_request_metadata())
                return FakeSummary()

        with patch.object(analysis_pipeline, "TSDAnalysisPipeline", FakePipeline):
            review = SimpleNamespace(id=7)
            summary = analysis_pipeline.run_tsd_analysis(review)

        self.assertEqual(summary.to_dict(), {"ok": True})
        self.assertEqual(observed["session_id"], build_tsd_analysis_session_id(7))
        self.assertEqual(observed["job_type"], "tsd_analysis")
        self.assertEqual(observed["job_id"], "7")

    def test_session_id_builders_use_job_scoped_names(self):
        self.assertEqual(build_standard_ingestion_session_id(15), "standard_ingestion_job_15")
        self.assertEqual(build_tsd_analysis_session_id(22), "tsd_analysis_review_22")
