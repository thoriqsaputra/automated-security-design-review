from sdr.apps.ai.engine.extraction.services import _annotate_chunks_with_chapter_context


def _chunk(idx: int, total: int, body: str) -> dict:
    return {"text": f"--- DOCUMENT CHUNK {idx} OF {total} ---\n\n{body}"}


def test_first_chunk_with_its_own_heading_is_left_untouched():
    chunks = [_chunk(1, 2, "## V1 Chapter One\n\nSome V1 content.")]

    result = _annotate_chunks_with_chapter_context(chunks)

    assert "[CONTEXT:" not in result[0]["text"]


def test_chunk_opening_mid_chapter_gets_carried_forward_context():
    chunks = [
        _chunk(1, 2, "## V1 Chapter One\n\n" + ("filler text. " * 50)),
        _chunk(2, 2, "## V1.9 Sub Section\n\nOrphaned sub-section content, no top-level heading here."),
    ]

    result = _annotate_chunks_with_chapter_context(chunks)

    assert "[CONTEXT:" not in result[0]["text"]
    assert 'JSON section key: "V1 Chapter One"' in result[1]["text"]
    assert result[1]["text"].index("[CONTEXT:") < result[1]["text"].index("## V1.9 Sub Section")


def test_active_chapter_flips_once_a_new_top_level_heading_is_seen():
    chunks = [
        _chunk(1, 3, "## V1 Chapter One\n\n" + ("filler text. " * 50)),
        _chunk(
            2,
            3,
            "## V1.9 Sub Section\n\n"
            + ("filler text. " * 50)
            + "\n\n## V2 Chapter Two\n\nV2 intro content.",
        ),
        _chunk(3, 3, "## V2.1 Sub Section\n\nMore orphaned content under V2."),
    ]

    result = _annotate_chunks_with_chapter_context(chunks)

    assert '"V1 Chapter One"' in result[1]["text"]
    assert 'JSON section key: "V2 Chapter Two"' in result[2]["text"]
    assert 'never fuse the chapter letter "V2" onto it' in result[2]["text"]


def test_chunk_that_starts_with_its_own_new_heading_is_not_annotated():
    chunks = [
        _chunk(1, 2, "## V1 Chapter One\n\n" + ("filler text. " * 50)),
        _chunk(2, 2, "## V2 Chapter Two\n\nV2 content starts immediately."),
    ]

    result = _annotate_chunks_with_chapter_context(chunks)

    assert "[CONTEXT:" not in result[1]["text"]
