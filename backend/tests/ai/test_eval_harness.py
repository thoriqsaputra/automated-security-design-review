from __future__ import annotations

from types import SimpleNamespace

from sdr.apps.ai.evaluations.metrics import calculate_context_precision
from sdr.apps.ai.evaluations.judges import judge_faithfulness_deterministic
from sdr.apps.ai.retrieval.postprocessing.quote_grounding import is_quote_grounded


def test_calculate_context_precision_single_id_backward_compatible():
    assert calculate_context_precision("b2", ["b1", "b2", "b3"]) == 0.5
    assert calculate_context_precision("missing", ["b1", "b2"]) == 0.0


def test_calculate_context_precision_best_rank_wins_across_multiple_expected_ids():
    # "b3" is at rank 1, "b1" is at rank 3 — best (lowest-rank) precision should win.
    retrieved = ["b3", "b2", "b1"]
    assert calculate_context_precision(["b1", "b3"], retrieved) == 1.0


def test_calculate_context_precision_all_missing_returns_zero():
    assert calculate_context_precision(["x", "y"], ["b1", "b2", "b3"]) == 0.0


def test_build_real_review_dataset_skips_na_and_citation_less_findings():
    from sdr.apps.ai.evaluations.dataset_generator import build_real_review_dataset

    met_with_citations = SimpleNamespace(
        id=1,
        review_id=42,
        category_id=7,
        child_parameter_id=99,
        met_status="met",
        requirement_text="Verify that MFA is enforced.",
        citations=[
            SimpleNamespace(block_id="blk-1", quoted_text="MFA is required for all users."),
            SimpleNamespace(block_id="blk-2", quoted_text="A second factor is mandatory."),
        ],
    )
    # "na" findings are excluded by the SQL filter itself (met_status.in_([MET, NOT_MET]))
    # before rows ever reach this function — so the fake DB never returns one here.
    no_citations_finding = SimpleNamespace(
        id=3,
        review_id=42,
        category_id=7,
        child_parameter_id=101,
        met_status="not_met",
        requirement_text="Verify that encryption is used.",
        citations=[],
    )

    class FakeQuery:
        def __init__(self, rows):
            self._rows = rows

        def filter(self, *args, **kwargs):
            return self

        def all(self):
            return self._rows

    class FakeDB:
        def query(self, model):
            return FakeQuery([met_with_citations, no_citations_finding])

    dataset = build_real_review_dataset(FakeDB(), review_id=42)

    assert len(dataset) == 1
    item = dataset[0]
    assert item["source"] == "real_review"
    assert item["finding_id"] == 1
    assert item["block_ids"] == ["blk-1", "blk-2"]
    assert item["question"] == "Verify that MFA is enforced."
    assert "MFA is required for all users." in item["ground_truth_context"]
    assert "A second factor is mandatory." in item["ground_truth_context"]


def test_judge_faithfulness_deterministic_matches_is_quote_grounded():
    grounded_quote = "MFA is required for all users."
    ungrounded_quote = "Passwords must be stored in plaintext."
    blocks = {"blk-1": "All users must authenticate using MFA is required for all users in this system."}

    score = judge_faithfulness_deterministic([grounded_quote, ungrounded_quote], blocks)

    assert score == 0.5
    assert is_quote_grounded(grounded_quote, blocks["blk-1"]) is True
    assert is_quote_grounded(ungrounded_quote, blocks["blk-1"]) is False


def test_judge_faithfulness_deterministic_no_quotes_is_fully_faithful():
    assert judge_faithfulness_deterministic([], {"blk-1": "anything"}) == 1.0
