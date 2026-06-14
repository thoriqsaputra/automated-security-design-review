from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RetrievalSnapshotBuilder:
    def __init__(self, *, workflow_repository) -> None:
        self.workflow_repository = workflow_repository
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def save(self, review, indexes) -> Optional[Dict[str, Any]]:
        snapshot = self.build_snapshot(indexes)
        if snapshot is None:
            return None
        try:
            self.workflow_repository.save_retrieval_snapshot(review.id, snapshot=snapshot)
            review.retrieval_snapshot_json = snapshot
            self.logger.info(
                "RetrievalSnapshotBuilder.save: review_id=%s status=%s raptor_status=%s graph_status=%s",
                review.id,
                snapshot.get("status"),
                ((snapshot.get("raptor") or {}).get("status") if isinstance(snapshot.get("raptor"), dict) else None),
                ((snapshot.get("graph") or {}).get("status") if isinstance(snapshot.get("graph"), dict) else None),
            )
        except Exception as exc:
            self.logger.exception(
                "RetrievalSnapshotBuilder.save: failed for review_id=%s: %s",
                review.id,
                exc,
            )
        return snapshot

    def build_snapshot(self, indexes) -> Optional[Dict[str, Any]]:
        raptor_snapshot = self.serialize_raptor_snapshot(getattr(indexes, "raptor_tree", None))
        graph_snapshot = self.serialize_graph_snapshot(getattr(indexes, "tsd_graph", None))
        if raptor_snapshot is None and graph_snapshot is None:
            return None
        return {
            "status": "ready",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "raptor": raptor_snapshot,
            "graph": graph_snapshot,
        }

    def serialize_raptor_snapshot(self, raptor_tree) -> Optional[Dict[str, Any]]:
        if not raptor_tree or getattr(raptor_tree, "is_empty", lambda: True)():
            return {"status": "unavailable", "total_nodes": 0, "max_level": 0, "root_node_id": None, "nodes": []}
        nodes = []
        for node in getattr(raptor_tree, "get_all_nodes", lambda: [])():
            nodes.append(
                {
                    "id": node.node_id,
                    "parent_id": self.find_raptor_parent_id(raptor_tree, node.node_id),
                    "level": int(getattr(node, "level", 0) or 0),
                    "section_heading": getattr(node, "section_heading", None),
                    "text_preview": self.summarize_log_text(getattr(node, "text", ""), limit=280),
                    "page_numbers": list(getattr(node, "page_numbers", []) or []),
                    "source_block_count": len(getattr(node, "source_block_ids", []) or []),
                    "child_count": len(getattr(node, "children", []) or []),
                }
            )
        return {
            "status": "ready",
            "total_nodes": int(getattr(raptor_tree, "total_nodes", len(nodes)) or len(nodes)),
            "max_level": int(getattr(raptor_tree, "max_level", 0) or 0),
            "root_node_id": getattr(getattr(raptor_tree, "root_node", None), "node_id", None),
            "nodes": nodes,
        }

    def find_raptor_parent_id(self, raptor_tree, node_id: str) -> Optional[str]:
        for candidate in getattr(raptor_tree, "get_all_nodes", lambda: [])():
            for child in getattr(candidate, "children", []) or []:
                if getattr(child, "node_id", None) == node_id:
                    return getattr(candidate, "node_id", None)
        return None

    def serialize_graph_snapshot(self, tsd_graph) -> Optional[Dict[str, Any]]:
        if not tsd_graph or getattr(tsd_graph, "is_empty", lambda: True)():
            return {"status": "unavailable", "total_entities": 0, "total_relations": 0, "nodes": [], "edges": []}
        nodes = []
        for entity in (getattr(tsd_graph, "entities", {}) or {}).values():
            degree = in_degree = out_degree = 0
            graph_obj = getattr(tsd_graph, "graph", None)
            if graph_obj is not None:
                try:
                    degree = int(graph_obj.degree(entity.entity_id))
                    in_degree = int(graph_obj.in_degree(entity.entity_id))
                    out_degree = int(graph_obj.out_degree(entity.entity_id))
                except Exception:
                    degree = in_degree = out_degree = 0
            nodes.append(
                {
                    "id": entity.entity_id,
                    "label": entity.name,
                    "entity_type": entity.entity_type,
                    "source_pages": list(entity.source_pages or []),
                    "source_block_count": len(entity.source_block_ids or []),
                    "degree": degree,
                    "in_degree": in_degree,
                    "out_degree": out_degree,
                }
            )
        edges = []
        graph_obj = getattr(tsd_graph, "graph", None)
        if graph_obj is not None:
            for source, target, data in graph_obj.edges(data=True):
                relation = data.get("relation") if isinstance(data, dict) else None
                relation_obj = data.get("relation_obj") if isinstance(data, dict) else None
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "relation_type": relation,
                        "confidence": float(getattr(relation_obj, "confidence", 0.0) or 0.0),
                        "protocol": getattr(relation_obj, "protocol", None),
                        "is_encrypted": getattr(relation_obj, "is_encrypted", None),
                        "requires_auth": getattr(relation_obj, "requires_auth", None),
                        "source_pages": list(getattr(relation_obj, "source_pages", []) or []),
                    }
                )
        return {
            "status": "ready",
            "total_entities": int(getattr(tsd_graph, "total_entities", len(nodes)) or len(nodes)),
            "total_relations": int(getattr(tsd_graph, "total_relations", len(edges)) or len(edges)),
            "nodes": nodes,
            "edges": edges,
        }

    def summarize_log_text(self, value: Optional[str], *, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 3)].rstrip()}..."
