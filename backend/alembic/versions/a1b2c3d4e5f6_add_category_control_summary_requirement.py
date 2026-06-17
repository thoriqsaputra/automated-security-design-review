"""add category control summary requirement

Revision ID: a1b2c3d4e5f6
Revises: 88b0f6f0f0e1
Create Date: 2026-06-16 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "88b0f6f0f0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "standards_categorycontrolsummaryrequirement",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("category_id", sa.BigInteger(), nullable=False),
        sa.Column("ingestion_job_id", sa.BigInteger(), nullable=True),
        sa.Column("parent_id", sa.BigInteger(), nullable=False),
        sa.Column("asvs_level", sa.Integer(), nullable=True),
        sa.Column("stable_key", sa.String(length=255), nullable=False),
        sa.Column("requirement_text", sa.String(), nullable=False),
        sa.Column("analysis_hint", sa.String(), nullable=False),
        sa.Column(
            "covered_child_keys",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["category_id"],
            ["standards_standardcategory.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["ingestion_job_id"],
            ["standards_standingestionjob.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["standards_categoryparameterparent.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "category_id",
            "stable_key",
            name="unique_cfsr_key_per_category",
        ),
        sa.CheckConstraint(
            "asvs_level IN (1, 2, 3)",
            name="ck_cfsr_asvs_level_range",
        ),
    )
    op.create_index(
        "ix_cfsr_category_id",
        "standards_categorycontrolsummaryrequirement",
        ["category_id"],
    )
    op.create_index(
        "ix_cfsr_ingestion_job_id",
        "standards_categorycontrolsummaryrequirement",
        ["ingestion_job_id"],
    )
    op.create_index(
        "ix_cfsr_parent_id",
        "standards_categorycontrolsummaryrequirement",
        ["parent_id"],
    )
    op.create_index(
        "ix_cfsr_asvs_level",
        "standards_categorycontrolsummaryrequirement",
        ["asvs_level"],
    )
    op.create_index(
        "ix_cfsr_stable_key",
        "standards_categorycontrolsummaryrequirement",
        ["stable_key"],
    )


def downgrade() -> None:
    op.drop_index("ix_cfsr_stable_key", table_name="standards_categorycontrolsummaryrequirement")
    op.drop_index("ix_cfsr_asvs_level", table_name="standards_categorycontrolsummaryrequirement")
    op.drop_index("ix_cfsr_parent_id", table_name="standards_categorycontrolsummaryrequirement")
    op.drop_index("ix_cfsr_ingestion_job_id", table_name="standards_categorycontrolsummaryrequirement")
    op.drop_index("ix_cfsr_category_id", table_name="standards_categorycontrolsummaryrequirement")
    op.drop_table("standards_categorycontrolsummaryrequirement")
