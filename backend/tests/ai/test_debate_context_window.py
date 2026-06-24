from __future__ import annotations

from sdr.apps.ai.engine.debate.debate_input_factory import DebateInputFactory
from sdr.apps.ai.tsd_processing.document_models import TextBlock, TSDDocument, TSDPage


def _block(block_id, text, y0, section_heading="Authentication"):
    return TextBlock(
        block_id=block_id,
        text=text,
        page_number=1,
        bbox_x0=0.0,
        bbox_y0=y0,
        bbox_x1=100.0,
        bbox_y1=y0 + 10.0,
        section_heading=section_heading,
    )


def _document_with_blocks(blocks):
    page = TSDPage(page_number=1, text_blocks=blocks)
    return TSDDocument(file_path="x.pdf", document_name="x", pages=[page])


def test_get_block_window_text_merges_neighbors_in_same_section():
    blocks = [
        _block("b1", "The gateway authenticates all inbound requests.", 0),
        _block("b2", "It validates the bearer token signature first.", 10),
        _block("b3", "Then it checks the token against the revocation list.", 20),
    ]
    document = _document_with_blocks(blocks)

    merged = document.get_block_window_text("b2", before=1, after=1, char_budget=2200)

    assert "validates the bearer token signature" in merged
    assert "authenticates all inbound requests" in merged
    assert "revocation list" in merged


def test_get_block_window_text_stops_at_section_boundary():
    blocks = [
        _block("b1", "Authentication section closing remark.", 0, section_heading="Authentication"),
        _block("b2", "Authorization section opening remark.", 10, section_heading="Authorization"),
    ]
    document = _document_with_blocks(blocks)

    merged = document.get_block_window_text("b2", before=1, after=0, char_budget=2200)

    assert merged == "Authorization section opening remark."


def test_get_block_window_text_respects_char_budget():
    blocks = [
        _block("b1", "A" * 2000, 0),
        _block("b2", "TARGET " * 5, 10),
        _block("b3", "C" * 2000, 20),
    ]
    document = _document_with_blocks(blocks)

    merged = document.get_block_window_text("b2", before=1, after=1, char_budget=50)

    assert "TARGET" in merged
    assert "A" * 2000 not in merged
    assert "C" * 2000 not in merged


def test_get_block_window_text_falls_back_for_unknown_block():
    document = _document_with_blocks([_block("b1", "Solo block text.", 0)])

    assert document.get_block_window_text("missing") == ""


def test_build_context_chunk_map_uses_merged_window_for_supplemental_blocks():
    blocks = [
        _block("b1", "The gateway authenticates all inbound requests.", 0),
        _block("b2", "It validates the bearer token signature first.", 10),
        _block("b3", "Then it checks the token against the revocation list.", 20),
    ]
    document = _document_with_blocks(blocks)
    factory = DebateInputFactory()

    chunk_map = factory.build_context_chunk_map(
        context_chunks=[],
        retrieval_metadata={},
        tsd_document=document,
        source_block_ids=["b2"],
        include_source_blocks=True,
    )

    assert "b2" in chunk_map
    merged_text = chunk_map["b2"]["text"]
    assert "validates the bearer token signature" in merged_text
    assert "authenticates all inbound requests" in merged_text
    assert "revocation list" in merged_text
    assert chunk_map["b2"]["citation_grade"] is True


def test_build_context_chunk_map_ranks_blocks_by_query_relevance_before_truncating():
    blocks = [
        _block("b1", "The SMS gateway rotates its signing key nightly.", 0, section_heading="Messaging"),
        _block("b2", "The billing export job runs every quarter.", 10, section_heading="Billing"),
        _block("b3", "All forms require a valid CAPTCHA and anti-CSRF token before submission.", 20, section_heading="Forms"),
    ]
    document = _document_with_blocks(blocks)
    factory = DebateInputFactory()

    chunk_map = factory.build_context_chunk_map(
        context_chunks=[],
        retrieval_metadata={},
        tsd_document=document,
        source_block_ids=["b1", "b2", "b3"],
        include_source_blocks=True,
        source_block_limit=1,
        query_text="Does the application use anti-CSRF tokens with CAPTCHA on forms?",
    )

    assert "b3" in chunk_map
    assert "b1" not in chunk_map
    assert "b2" not in chunk_map


def test_build_context_chunk_map_preserves_original_order_without_query_text():
    blocks = [
        _block("b1", "First unrelated block.", 0, section_heading="Other"),
        _block("b2", "Second unrelated block.", 10, section_heading="Other"),
    ]
    document = _document_with_blocks(blocks)
    factory = DebateInputFactory()

    chunk_map = factory.build_context_chunk_map(
        context_chunks=[],
        retrieval_metadata={},
        tsd_document=document,
        source_block_ids=["b1", "b2"],
        include_source_blocks=True,
        source_block_limit=1,
    )

    assert "b1" in chunk_map
    assert "b2" not in chunk_map


def test_get_block_window_bbox_spans_all_merged_blocks():
    blocks = [
        _block("b1", "The gateway authenticates all inbound requests.", 0),
        _block("b2", "It validates the bearer token signature first.", 10),
        _block("b3", "Then it checks the token against the revocation list.", 20),
    ]
    document = _document_with_blocks(blocks)

    window = document.get_block_window("b2", before=1, after=1, char_budget=2200)

    assert window["bbox"] == (0.0, 0.0, 100.0, 30.0)
    assert window["block_ids"] == ["b1", "b2", "b3"]


def test_build_context_chunk_map_bbox_spans_merged_window_not_just_target_block():
    blocks = [
        _block("b1", "The gateway authenticates all inbound requests.", 0),
        _block("b2", "It validates the bearer token signature first.", 10),
        _block("b3", "Then it checks the token against the revocation list.", 20),
    ]
    document = _document_with_blocks(blocks)
    factory = DebateInputFactory()

    chunk_map = factory.build_context_chunk_map(
        context_chunks=[],
        retrieval_metadata={},
        tsd_document=document,
        source_block_ids=["b2"],
        include_source_blocks=True,
    )

    payload = chunk_map["b2"]
    # b2's own bbox alone is (0, 10, 100, 20) — the fix must widen this to
    # cover b1+b2+b3 so the PDF viewer highlight covers the full quoted text.
    assert payload["bbox_y0"] == 0.0
    assert payload["bbox_y1"] == 30.0
    assert payload["bbox"] == {"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 30.0}
