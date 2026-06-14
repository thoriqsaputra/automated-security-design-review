from sdr.apps.ai.retrieval import HybridRetrievalRouter, RetrievalCandidate, RetrievalResult
from sdr.apps.ai.retrieval.graph.communities import GraphCommunityService
from sdr.apps.ai.retrieval.routing.router import HybridRetrievalRouter as CanonicalRouter


def test_retrieval_package_exports_canonical_symbols():
    assert HybridRetrievalRouter is CanonicalRouter
    assert RetrievalCandidate is not None
    assert RetrievalResult is not None
    assert GraphCommunityService is not None
