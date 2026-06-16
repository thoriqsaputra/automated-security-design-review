"""add diagram requirement embeddings

Revision ID: 88b0f6f0f0e1
Revises: 188246f8f1c3
Create Date: 2026-06-15 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = "88b0f6f0f0e1"
down_revision = "188246f8f1c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "standards_categorydiagramrequirementembedding",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("diagram_requirement_id", sa.BigInteger(), nullable=False),
        sa.Column("model_name", sa.String(length=128), nullable=False),
        sa.Column("model_dim", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["diagram_requirement_id"],
            ["standards_categorydiagramrequirement.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_diagramreqembedding_req_id",
        "standards_categorydiagramrequirementembedding",
        ["diagram_requirement_id"],
        unique=False,
    )
    op.create_index(
        "idx_categorydiagramrequirementembedding_req_active",
        "standards_categorydiagramrequirementembedding",
        ["diagram_requirement_id", "is_active"],
        unique=False,
    )
    op.create_index(
        "idx_categorydiagramrequirementembedding_content_hash",
        "standards_categorydiagramrequirementembedding",
        ["content_hash"],
        unique=False,
    )
    op.execute(
        "CREATE INDEX diagram_requirement_embedding_hnsw_idx "
        "ON standards_categorydiagramrequirementembedding "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS diagram_requirement_embedding_hnsw_idx")
    op.drop_index(
        "idx_categorydiagramrequirementembedding_content_hash",
        table_name="standards_categorydiagramrequirementembedding",
    )
    op.drop_index(
        "idx_categorydiagramrequirementembedding_req_active",
        table_name="standards_categorydiagramrequirementembedding",
    )
    op.drop_index(
        "idx_diagramreqembedding_req_id",
        table_name="standards_categorydiagramrequirementembedding",
    )
    op.drop_table("standards_categorydiagramrequirementembedding")
