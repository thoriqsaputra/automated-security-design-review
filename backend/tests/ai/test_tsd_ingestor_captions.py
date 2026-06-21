from __future__ import annotations

from sdr.apps.ai.tsd_processing.ingestor import TextBlock, TSDIngestor, TSDPage


def _block(block_id: str, text: str, bbox_y0: float, bbox_y1: float) -> TextBlock:
    return TextBlock(
        block_id=block_id,
        text=text,
        page_number=1,
        bbox_x0=0,
        bbox_y0=bbox_y0,
        bbox_x1=100,
        bbox_y1=bbox_y1,
    )


# Diagram bbox spans y0=100 (top) to y1=200 (bottom).
_DIAGRAM_BBOX = (0.0, 100.0, 100.0, 200.0)


def test_get_blocks_near_bbox_direction_above_below_both():
    above_block = _block("above", "Figure 1: Above caption", bbox_y0=70, bbox_y1=90)
    below_block = _block("below", "Figure 1: Below caption", bbox_y0=210, bbox_y1=230)
    page = TSDPage(page_number=1, text_blocks=[above_block, below_block])

    above_only = page.get_blocks_near_bbox(_DIAGRAM_BBOX, radius_pt=50.0, direction="above")
    below_only = page.get_blocks_near_bbox(_DIAGRAM_BBOX, radius_pt=50.0, direction="below")
    both = page.get_blocks_near_bbox(_DIAGRAM_BBOX, radius_pt=50.0, direction="both")

    assert [b.block_id for b in above_only] == ["above"]
    assert [b.block_id for b in below_only] == ["below"]
    assert [b.block_id for b in both] == ["above", "below"]


def test_extract_caption_falls_back_to_block_above_diagram():
    above_block = _block("above", "Figure 1: Architecture overview", bbox_y0=70, bbox_y1=90)
    page = TSDPage(page_number=1, text_blocks=[above_block])
    ingestor = TSDIngestor()

    caption = ingestor._extract_caption(page, _DIAGRAM_BBOX)

    assert caption == "Figure 1: Architecture overview"


def test_extract_surrounding_text_includes_blocks_above_and_below():
    above_block = _block("above", "The diagram below shows the data flow.", bbox_y0=70, bbox_y1=90)
    below_block = _block("below", "Figure 1: Architecture overview", bbox_y0=210, bbox_y1=230)
    page = TSDPage(page_number=1, text_blocks=[above_block, below_block])
    ingestor = TSDIngestor()

    surrounding_text = ingestor._extract_surrounding_text(page, _DIAGRAM_BBOX)

    assert surrounding_text == "The diagram below shows the data flow. Figure 1: Architecture overview"
