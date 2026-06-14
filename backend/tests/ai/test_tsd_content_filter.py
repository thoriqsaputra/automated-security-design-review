from __future__ import annotations

import base64

from sdr.apps.ai.engine.dto import RetrievalIndexes
from sdr.apps.ai.engine.preparation.retrieval_service import RetrievalService
from sdr.apps.ai.retrieval.searchers.graph import GraphSearcher
from sdr.apps.ai.tsd_processing.content_filter import build_filtered_tsd_view
from sdr.apps.ai.tsd_processing.document_models import TSDDocument as CanonicalTSDDocument
from sdr.apps.ai.tsd_processing.graph_builder import GraphEntity, TSDGraphBuilder
from sdr.apps.ai.tsd_processing.graph_builder import GraphRelation, TSDGraph
from sdr.apps.ai.tsd_processing.ingestor import DiagramBlock, TSDDocument, TSDPage, TextBlock
from sdr.apps.ai.tsd_processing.prepared_view import prepare_tsd_view
from sdr.apps.ai.tsd_processing.raptor import RAPTORTreeBuilder


def _block(block_id: str, text: str, page_number: int = 1, heading: str | None = None) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        text=text,
        page_number=page_number,
        bbox_x0=0,
        bbox_y0=0,
        bbox_x1=100,
        bbox_y1=20,
        section_heading=heading,
    )


def _page(page_number: int, heading: str, blocks: list[TextBlock]) -> TSDPage:
    for block in blocks:
        block.page_number = page_number
        block.section_heading = block.section_heading or heading
    return TSDPage(
        page_number=page_number,
        text_blocks=blocks,
        section_heading=heading,
        markdown_text="\n".join(block.text for block in blocks),
        raw_text="\n".join(block.text for block in blocks),
    )


def _diagram(diagram_id: str, page_number: int) -> DiagramBlock:
    return DiagramBlock(
        diagram_id=diagram_id,
        page_number=page_number,
        bbox_x0=0,
        bbox_y0=0,
        bbox_x1=10,
        bbox_y1=10,
        image_b64=base64.b64encode(b"x" * 600).decode("ascii"),
    )


def _document(pages: list[TSDPage]) -> TSDDocument:
    return TSDDocument(
        file_path="/tmp/example.pdf",
        document_name="Example TSD",
        pages=pages,
        total_pages=len(pages),
        total_text_blocks=sum(len(page.text_blocks) for page in pages),
    )


def test_content_filter_excludes_admin_and_glossary_but_keeps_security_sections(settings_override):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    doc = _document(
        [
            _page(1, "Revision History", [_block("p1_b1", "Revision history version date author approved by security team")]),
            _page(2, "Glossary", [_block("p2_b1", "API application programming interface JWT json web token")]),
            _page(3, "Architecture Overview", [_block("p3_b1", "The API Gateway calls Auth Service and writes customer records to User Database")]),
            _page(4, "Authentication", [_block("p4_b1", "Users authenticate with OIDC and JWT tokens over HTTPS TLS connections")]),
        ]
    )

    view = build_filtered_tsd_view(doc)

    assert "p1_b1" in view.excluded_block_ids
    assert "p2_b1" in view.excluded_block_ids
    assert "p3_b1" in view.included_block_ids
    assert "p4_b1" in view.included_block_ids
    assert view.stats["excluded_by_class"]["revision_history"] == 1
    assert view.stats["excluded_by_class"]["glossary"] == 1


def test_content_filter_keeps_explicit_negative_scope_statements(settings_override):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    doc = _document(
        [
            _page(1, "Out of Scope", [_block("p1_b1", "This system is API-only with no mobile app and no browser frontend")]),
        ]
    )

    view = build_filtered_tsd_view(doc)

    assert view.included_block_ids == ["p1_b1"]
    assert "no mobile app" in view.full_text.lower()


def test_raptor_leaf_nodes_use_filtered_text_and_preserve_block_ids(settings_override, monkeypatch):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    doc = _document(
        [
            _page(1, "Revision History", [_block("p1_b1", "Revision history version date author approved by security team")]),
            _page(2, "API Architecture", [_block("p2_b1", "The Merchant API calls Payment Service over HTTPS TLS")]),
            _page(3, "Database", [_block("p3_b1", "Payment Service writes transaction records to the Ledger Database")]),
            _page(4, "Authentication", [_block("p4_b1", "The API requires JWT authentication through the Auth Service")]),
        ]
    )
    builder = RAPTORTreeBuilder(max_depth=0)
    monkeypatch.setattr(builder, "_embed_all_nodes", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_synthesise_root", lambda *args, **kwargs: None)

    tree = builder.build(doc)
    leaf_text = "\n".join(node.text for node in tree.get_leaf_nodes())
    leaf_block_ids = [block_id for node in tree.get_leaf_nodes() for block_id in node.source_block_ids]

    assert "Revision history" not in leaf_text
    assert "Merchant API calls Payment Service" in leaf_text
    assert "p1_b1" not in leaf_block_ids
    assert {"p2_b1", "p3_b1", "p4_b1"}.issubset(set(leaf_block_ids))
    assert tree.build_stats["content_filter_included_blocks"] == 3
    assert tree.build_stats["content_filter_excluded_blocks"] == 1


def test_graphrag_extraction_receives_only_filtered_pages(settings_override, monkeypatch):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    doc = _document(
        [
            _page(1, "Approval", [_block("p1_b1", "Approval table approved by author reviewer security sign off")]),
            _page(2, "API Architecture", [_block("p2_b1", "The Public API calls Billing Service over HTTPS")]),
        ]
    )
    builder = TSDGraphBuilder(graph_extraction_mode="llm")
    seen_pages: list[int] = []

    def _fake_extract(page: TSDPage):
        seen_pages.append(page.page_number)
        assert [block.block_id for block in page.text_blocks] == ["p2_b1"]
        return [
            GraphEntity(
                entity_id="public_api",
                name="Public API",
                entity_type="api",
                source_pages=[page.page_number],
                source_block_ids=[block.block_id for block in page.text_blocks],
            )
        ], []

    monkeypatch.setattr(builder, "_extract_from_page", _fake_extract)
    monkeypatch.setattr(builder, "_embed_graph_entities", lambda *args, **kwargs: None)

    graph = builder.build(doc)

    assert seen_pages == [2]
    assert graph.entities["public_api"].source_block_ids == ["p2_b1"]
    assert graph.build_stats["content_filter_excluded_blocks"] == 1


def test_prefilter_profile_ignores_excluded_glossary_terms_but_keeps_negative_scope(settings_override):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    doc = _document(
        [
            _page(1, "Glossary", [_block("p1_b1", "Mobile means Android and iOS application clients")]),
            _page(2, "Scope", [_block("p2_b1", "The platform is API-only with no mobile app or browser frontend")]),
        ]
    )
    indexes = RetrievalIndexes.model_construct(raptor_tree=None, tsd_graph=None)
    profile = RetrievalService()._build_document_capability_profile(indexes, tsd_document=doc)

    assert profile["signals"]["mobile"]["present"] is False
    assert profile["signals"]["explicit_non_mobile_profile"]["present"] is True


def test_content_filter_excludes_unicode_dotted_figure_list_entries(settings_override):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    dotted = "Figure 1. Use Case Activity Diagram……………………………………………………..…………………… 7"
    doc = _document(
        [
            _page(1, "List of Figures", [_block("p1_b1", dotted)]),
            _page(2, "Authentication Architecture", [_block("p2_b1", "The API Gateway sends JWT tokens to the Auth Service over TLS")]),
        ]
    )

    view = build_filtered_tsd_view(doc)

    assert "p1_b1" in view.excluded_block_ids
    assert "p2_b1" in view.included_block_ids
    assert view.stats["excluded_by_class"]["table_of_contents"] == 1


def test_raptor_excludes_list_of_figures_page_from_leaf_nodes(settings_override, monkeypatch):
    settings_override(AI_TSD_CONTENT_FILTER_ENABLED=True, AI_TSD_CONTENT_FILTER_MIN_SCORE=1)
    dotted = "Figure 2. Class Diagram……………………………………………………………………………………………… 18"
    doc = _document(
        [
            _page(1, "List of Figures", [_block("p1_b1", dotted)]),
            _page(2, "Architecture Overview", [_block("p2_b1", "Client requests pass through API Gateway to Auth Service and Ledger Database")]),
            _page(3, "Session Controls", [_block("p3_b1", "Sessions use JWT tokens and TLS-protected transport between services")]),
            _page(4, "Monitoring", [_block("p4_b1", "Audit logs are forwarded to monitoring and alerting services")]),
        ]
    )
    builder = RAPTORTreeBuilder(max_depth=0)
    monkeypatch.setattr(builder, "_embed_all_nodes", lambda *args, **kwargs: None)
    monkeypatch.setattr(builder, "_synthesise_root", lambda *args, **kwargs: None)

    tree = builder.build(doc)
    leaf_text = "\n".join(node.text for node in tree.get_leaf_nodes())

    assert "Class Diagram" not in leaf_text
    assert "API Gateway to Auth Service" in leaf_text
    assert tree.build_stats["content_filter_excluded_blocks"] == 1


def test_retrieval_service_build_indexes_prepares_filtered_view_once(monkeypatch):
    doc = _document(
        [
            _page(1, "Architecture", [_block("p1_b1", "API Gateway calls Auth Service over TLS")]),
            _page(2, "Data", [_block("p2_b1", "Auth Service writes session state to Redis Cache")]),
        ]
    )


def test_tsd_document_all_diagrams_flattens_page_diagrams():
    doc = CanonicalTSDDocument(
        file_path="/tmp/example.pdf",
        document_name="Example TSD",
        pages=[
            TSDPage(page_number=1, diagrams=[_diagram("p1_d1", 1), _diagram("p1_d2", 1)]),
            TSDPage(page_number=2, diagrams=[_diagram("p2_d1", 2)]),
        ],
    )

    assert [diagram.diagram_id for diagram in doc.all_diagrams] == ["p1_d1", "p1_d2", "p2_d1"]


def test_retrieval_service_infers_diagram_ids_from_document_pages():
    doc = CanonicalTSDDocument(
        file_path="/tmp/example.pdf",
        document_name="Example TSD",
        pages=[
            TSDPage(page_number=1, diagrams=[_diagram("p1_d1", 1)]),
            TSDPage(page_number=3, diagrams=[_diagram("p3_d1", 3)]),
        ],
    )
    service = RetrievalService()

    diagram_ids, metadata = service._infer_diagram_block_ids(
        source_block_ids=["p1_b1"],
        tsd_document=doc,
        max_diagrams=3,
    )

    assert diagram_ids == ["p1_d1"]
    assert metadata == []
    prepared = prepare_tsd_view(doc)
    prepared_calls = []
    linked = []

    class _StubRaptorBuilder:
        def build(self, tsd_document, progress_callback=None, prepared_view=None):
            assert tsd_document is doc
            assert prepared_view is prepared
            return type(
                "StubTree",
                (),
                {
                    "is_empty": lambda self: False,
                    "total_nodes": 1,
                    "build_seconds": 0.0,
                },
            )()

    class _StubGraphBuilder:
        def build(self, tsd_document, progress_callback=None, prepared_view=None):
            assert tsd_document is doc
            assert prepared_view is prepared
            return type(
                "StubGraph",
                (),
                {
                    "is_empty": lambda self: False,
                    "total_entities": 1,
                    "total_relations": 0,
                    "build_stats": {},
                },
            )()

    monkeypatch.setattr(
        "sdr.apps.ai.engine.preparation.retrieval_service.prepare_tsd_view",
        lambda tsd_document: prepared_calls.append(tsd_document) or prepared,
    )

    service = RetrievalService(
        raptor_builder=_StubRaptorBuilder(),
        graph_builder=_StubGraphBuilder(),
        router=object(),
        linker=type("StubLinker", (), {"link": lambda self, graph, tree: linked.append((graph, tree))})(),
    )

    indexes = service.build_indexes(doc)

    assert indexes.raptor_tree is not None
    assert indexes.tsd_graph is not None
    assert prepared_calls == [doc]
    assert len(linked) == 1


def test_graph_search_relation_embedding_uses_public_formatter(monkeypatch):
    relation = GraphRelation(
        source_entity_id="api_gateway",
        target_entity_id="auth_service",
        relation_type="calls",
        description="API Gateway calls Auth Service",
        protocol="HTTPS",
        confidence=0.9,
    )
    graph = TSDGraph(document_name="Example")
    searcher = GraphSearcher()

    monkeypatch.setattr(
        "sdr.apps.ai.tsd_processing.graph_builder.TSDGraphBuilder._relation_embedding_text",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("private helper should not be used")),
    )
    monkeypatch.setattr(
        "sdr.apps.ai.retrieval.searchers.graph.get_embeddings",
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )

    searcher._ensure_relation_embeddings(graph=graph, relations=[relation])

    assert relation.has_embedding is True
    assert relation.embedding == [0.1, 0.2, 0.3]
