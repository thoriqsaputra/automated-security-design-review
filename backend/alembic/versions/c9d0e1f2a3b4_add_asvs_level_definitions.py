"""add per-ingestion asvs level definitions

Revision ID: c9d0e1f2a3b4
Revises: b7c8d9e0f1a2
Create Date: 2026-06-11 00:00:03.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, Sequence[str], None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "standards_asvsleveldefinition",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("ingestion_job_id", sa.Integer(), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("classification_guidance", sa.String(), nullable=False),
        sa.Column("source_quote", sa.String(), nullable=True),
        sa.Column("context_marker", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("level IN (1, 2, 3)", name="ck_asvs_level_definition_range"),
        sa.ForeignKeyConstraint(["ingestion_job_id"], ["standards_standingestionjob.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ingestion_job_id", "level", name="unique_asvs_level_definition_per_job"),
    )
    op.create_index(
        op.f("ix_standards_asvsleveldefinition_ingestion_job_id"),
        "standards_asvsleveldefinition",
        ["ingestion_job_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_standards_asvsleveldefinition_ingestion_job_id"),
        table_name="standards_asvsleveldefinition",
    )
    op.drop_table("standards_asvsleveldefinition")
