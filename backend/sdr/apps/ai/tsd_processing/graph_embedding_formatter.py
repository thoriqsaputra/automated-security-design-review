from __future__ import annotations

from typing import Any, List

_MAX_ENTITY_EMBED_TEXT_CHARS = 512
_MAX_RELATION_EMBED_TEXT_CHARS = 768


def build_entity_embedding_text(entity: Any) -> str:
    text = (
        f"Entity {entity.name} type {entity.entity_type}. "
        f"Description {entity.description or 'N/A'}."
    ).strip()
    return text[:_MAX_ENTITY_EMBED_TEXT_CHARS]


def build_relation_embedding_text(relation: Any) -> str:
    flags: List[str] = []
    if relation.protocol:
        flags.append(f"protocol={relation.protocol}")
    if relation.is_encrypted is not None:
        flags.append(f"encrypted={relation.is_encrypted}")
    if relation.requires_auth is not None:
        flags.append(f"auth={relation.requires_auth}")
    text = (
        f"Relation {relation.source_entity_id} {relation.relation_type} {relation.target_entity_id}. "
        f"Description {relation.description or 'N/A'}. "
        f"{' '.join(flags)}"
    ).strip()
    return text[:_MAX_RELATION_EMBED_TEXT_CHARS]


__all__ = ["build_entity_embedding_text", "build_relation_embedding_text"]
