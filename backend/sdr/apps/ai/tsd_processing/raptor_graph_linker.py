from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Set


class RaptorGraphLinker:
    def link(self, tsd_graph, raptor_tree) -> None:
        if tsd_graph is None or raptor_tree is None or raptor_tree.is_empty():
            return
        raptor_node_to_entities: Dict[str, Set[str]] = defaultdict(set)
        entity_to_raptor_node_ids: Dict[str, Set[str]] = defaultdict(set)
        for node in raptor_tree.get_all_nodes():
            node_block_ids = set(getattr(node, "source_block_ids", []) or [])
            if not node_block_ids:
                continue
            for entity_id, entity in (tsd_graph.entities or {}).items():
                entity_block_ids = set(getattr(entity, "source_block_ids", []) or [])
                if entity_block_ids and entity_block_ids.intersection(node_block_ids):
                    raptor_node_to_entities[node.node_id].add(entity_id)
                    entity_to_raptor_node_ids[entity_id].add(node.node_id)
        tsd_graph.raptor_node_to_entities = dict(raptor_node_to_entities)
        tsd_graph.entity_to_raptor_node_ids = dict(entity_to_raptor_node_ids)


__all__ = ["RaptorGraphLinker"]
