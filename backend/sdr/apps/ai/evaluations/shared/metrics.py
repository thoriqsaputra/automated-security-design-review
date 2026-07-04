from typing import Dict, List, Sequence, Tuple, Union


def calculate_set_retrieval_precision_recall(
    expected_ids: Sequence[str],
    retrieved_ids: Sequence[str],
) -> Tuple[float, float]:
    """
    Precision/recall of a ranked (or unranked) id list against an expected id set.
    Order-independent — use `calculate_context_precision` for rank-sensitive MRR.

    precision = |retrieved ∩ expected| / |retrieved|
    recall    = |retrieved ∩ expected| / |expected|
    """
    expected_set = set(expected_ids)
    retrieved_set = set(retrieved_ids)
    if not retrieved_set:
        precision = 0.0
    else:
        precision = len(expected_set & retrieved_set) / len(retrieved_set)
    if not expected_set:
        recall = 0.0
    else:
        recall = len(expected_set & retrieved_set) / len(expected_set)
    return precision, recall


def calculate_binary_confusion(
    labels: Sequence[str],
    preds: Sequence[Union[str, None]],
    *,
    positive_label: str = "met",
    ignore_label: str = "na",
) -> Dict[str, float]:
    """
    Binary confusion matrix (positive_label vs everything else), skipping any
    pair where the true label is `ignore_label` or the prediction is None.
    """
    tp = fp = fn = tn = 0
    for true, pred in zip(labels, preds):
        if true == ignore_label or pred is None:
            continue
        t_pos = true == positive_label
        p_pos = pred == positive_label
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


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
