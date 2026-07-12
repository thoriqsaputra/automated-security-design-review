from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, List, Optional

from sdr.apps.ai.retrieval.core.candidates import RetrievalCandidate
from sdr.apps.ai.tsd_processing.raptor import RAPTORNode, RAPTORTree

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

_BM25_SATURATION_K = 10.0

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except Exception:
    BM25_AVAILABLE = False


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


class KeywordSearcher:
    def search(
        self,
        query_text: str,
        tree: Optional[RAPTORTree],
        top_k: int = 20,
        allowed_levels: Optional[List[int]] = None,
    ) -> List[RetrievalCandidate]:
        if not query_text or not query_text.strip() or tree is None or tree.is_empty():
            return []

        all_nodes: List[RAPTORNode] = tree.get_all_nodes()
        if allowed_levels is not None:
            allowed = set(allowed_levels)
            all_nodes = [n for n in all_nodes if n.level in allowed]
        nodes = [n for n in all_nodes if n.text]
        if not nodes:
            return []

        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        corpus_tokens = [_tokenize(n.text) for n in nodes]
        if BM25_AVAILABLE:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
        else:
            scores = self._fallback_scores(query_tokens, corpus_tokens)

        scores = [max(0.0, s) / (max(0.0, s) + _BM25_SATURATION_K) for s in scores]

        paired = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        results: List[RetrievalCandidate] = []
        for node, score in paired[:top_k]:
            results.append(
                RetrievalCandidate(
                    id=node.node_id,
                    source_type="bm25",
                    text=node.text,
                    score=float(score),
                    block_ids=list(node.source_block_ids),
                    metadata={
                        "level": node.level,
                        "page_numbers": list(node.page_numbers),
                        "section_heading": node.section_heading,
                        "sensitivity": "internal",
                    },
                    token_count=node.token_estimate,
                )
            )
        return results

    def _fallback_scores(self, query_tokens: List[str], docs: Iterable[List[str]]) -> List[float]:
        doc_list = list(docs)
        n_docs = len(doc_list)
        doc_freq = Counter()
        for tokens in doc_list:
            for token in set(tokens):
                doc_freq[token] += 1

        scores: List[float] = []
        for tokens in doc_list:
            tf = Counter(tokens)
            dl = max(1, len(tokens))
            score = 0.0
            for query_token in query_tokens:
                if tf[query_token] == 0:
                    continue
                idf = math.log(1 + (n_docs - doc_freq[query_token] + 0.5) / (doc_freq[query_token] + 0.5))
                score += idf * (tf[query_token] / dl)
            scores.append(score)
        return scores


__all__ = ["KeywordSearcher"]
