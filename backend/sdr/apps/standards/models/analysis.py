from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Integer, ForeignKey, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref
from sdr.core.database import Base
from .base import StandardsBigIntBase


class CategoryDiagramRequirementEmbedding(Base, StandardsBigIntBase):
    __tablename__ = "standards_categorydiagramrequirementembedding"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    diagram_requirement_id: Mapped[int] = mapped_column(
        ForeignKey("standards_categorydiagramrequirement.id", ondelete="CASCADE"),
    )

    model_name: Mapped[str] = mapped_column(String(128))
    model_dim: Mapped[int] = mapped_column(Integer, default=1024)

    embedding = mapped_column(Vector(1024))

    content_hash: Mapped[str] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    diagram_requirement = relationship(
        "CategoryDiagramRequirement",
        backref=backref("embeddings", passive_deletes=True),
    )

    __table_args__ = (
        Index(
            "idx_categorydiagramrequirementembedding_req_active",
            "diagram_requirement_id",
            "is_active",
        ),
        Index(
            "idx_categorydiagramrequirementembedding_content_hash",
            "content_hash",
        ),
        Index(
            "diagram_requirement_embedding_hnsw_idx",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
