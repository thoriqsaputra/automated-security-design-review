from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from sdr.apps.ai.tsd_processing.content_filter import FilteredTSDDocumentView, build_filtered_tsd_view
from sdr.apps.ai.tsd_processing.document_models import TextBlock, TSDDocument, TSDPage


@dataclass(frozen=True)
class PreparedTSDView:
    tsd_document: TSDDocument
    filtered_view: FilteredTSDDocumentView
    pages_with_text: List[TSDPage]
    valid_blocks: List[TextBlock]
    stats: Dict[str, Any]


def prepare_tsd_view(tsd_document: TSDDocument) -> PreparedTSDView:
    filtered_view = build_filtered_tsd_view(tsd_document)
    pages_with_text = [page for page in filtered_view.pages if page.all_text.strip()]
    valid_blocks = [
        block
        for page in filtered_view.pages
        for block in page.text_blocks
        if block.is_valid()
    ]
    return PreparedTSDView(
        tsd_document=tsd_document,
        filtered_view=filtered_view,
        pages_with_text=pages_with_text,
        valid_blocks=valid_blocks,
        stats=dict(filtered_view.stats),
    )


__all__ = ["PreparedTSDView", "prepare_tsd_view"]
