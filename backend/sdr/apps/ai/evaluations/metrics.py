from typing import List, Sequence, Union


def calculate_context_precision(
    expected_block_ids: Union[str, Sequence[str]],
    retrieved_block_id_groups: Sequence[Sequence[str]],
) -> float:
    """
    Calculates Context Precision based on the ranking of the best-ranked
    retrieved chunk that contains an expected block. If a chunk at rank k
    (1-indexed) contains any expected block id among its full source block
    ids, the precision is 1/k for the lowest such k. If none are found, 0.0.

    `retrieved_block_id_groups` is rank-ordered: one entry per retrieved
    chunk, each entry being that chunk's full list of source block ids (a
    RAPTOR-grouped chunk can legitimately span many block ids, not just one).

    Accepts a single block id (legacy callers) or a list of block ids (a
    finding can cite multiple blocks).
    """
    if isinstance(expected_block_ids, str):
        expected_block_ids = [expected_block_ids]
    expected_set = set(expected_block_ids)

    for rank, group in enumerate(retrieved_block_id_groups, start=1):
        # Backward-compatible with a flat List[str] (one block id per rank),
        # as well as the grouped List[List[str]] form (one chunk's full set
        # of source block ids per rank).
        group_ids = [group] if isinstance(group, str) else group
        if expected_set & set(group_ids):
            return 1.0 / rank

    return 0.0
