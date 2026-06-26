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
                "RetrievalSnapshotBuilder.save: review_id=%s status=%s raptor_status=%s",
                review.id,
                snapshot.get("status"),
                ((snapshot.get("raptor") or {}).get("status") if isinstance(snapshot.get("raptor"), dict) else None),
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
        if raptor_snapshot is None:
            return None
        return {
            "status": "ready",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "raptor": raptor_snapshot,
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

    def summarize_log_text(self, value: Optional[str], *, limit: int = 240) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 3)].rstrip()}..."
