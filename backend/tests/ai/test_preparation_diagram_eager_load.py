from __future__ import annotations

import base64

from sdr.apps.ai.tsd_processing.document_models import DiagramBlock, TSDDocument, TSDPage
from sdr.apps.designs.preparation_store import _eager_load_diagram_images, serialize_tsd_document


def _diagram_with_lazy_bytes(diagram_id, image_bytes):
    diagram = DiagramBlock(
        diagram_id=diagram_id,
        page_number=1,
        bbox_x0=0.0,
        bbox_y0=0.0,
        bbox_x1=100.0,
        bbox_y1=100.0,
    )

    def _fake_ensure_image_loaded(min_diagram_bytes=512):
        diagram.image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return True

    diagram.ensure_image_loaded = _fake_ensure_image_loaded
    return diagram


def test_eager_load_diagram_images_resolves_every_diagram_before_serialization():
    diagram_a = _diagram_with_lazy_bytes("d1", b"x" * 600)
    diagram_b = _diagram_with_lazy_bytes("d2", b"y" * 700)
    page = TSDPage(page_number=1, diagrams=[diagram_a, diagram_b])
    document = TSDDocument(file_path="x.pdf", document_name="x", pages=[page])

    assert diagram_a.image_b64 == ""
    assert diagram_b.image_b64 == ""

    _eager_load_diagram_images(document)

    assert diagram_a.image_b64 != ""
    assert diagram_b.image_b64 != ""

    payload = serialize_tsd_document(document)
    serialized_diagrams = payload["pages"][0]["diagrams"]
    assert serialized_diagrams[0]["image_b64"] == diagram_a.image_b64
    assert serialized_diagrams[1]["image_b64"] == diagram_b.image_b64


def test_eager_load_diagram_images_tolerates_failed_resolution():
    diagram = DiagramBlock(diagram_id="d1", page_number=1, bbox_x0=0.0, bbox_y0=0.0, bbox_x1=10.0, bbox_y1=10.0)

    def _raise(min_diagram_bytes=512):
        raise RuntimeError("source PDF unavailable")

    diagram.ensure_image_loaded = _raise
    page = TSDPage(page_number=1, diagrams=[diagram])
    document = TSDDocument(file_path="x.pdf", document_name="x", pages=[page])

    _eager_load_diagram_images(document)

    assert diagram.image_b64 == ""
