from __future__ import annotations

from sdr.apps.ai.services.analysis.dto import RetrievalIndexes
from sdr.apps.ai.services.analysis.retrieval_service import RetrievalService
from sdr.apps.ai.tsd_processing.content_filter import build_filtered_tsd_view
from sdr.apps.ai.tsd_processing.graph_builder import GraphEntity, TSDGraphBuilder
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument, TSDPage, TextBlock
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
