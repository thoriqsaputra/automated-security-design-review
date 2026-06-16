"""fix diagram requirement embedding timestamps

Revision ID: 3f4f8d2ab1c1
Revises: 88b0f6f0f0e1
Create Date: 2026-06-15 00:00:01.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "3f4f8d2ab1c1"
down_revision = "88b0f6f0f0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "standards_categorydiagramrequirementembedding",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        existing_nullable=False,
    )
    op.alter_column(
        "standards_categorydiagramrequirementembedding",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        server_default=sa.text("now()"),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "standards_categorydiagramrequirementembedding",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        server_default=None,
        existing_nullable=False,
    )
    op.alter_column(
        "standards_categorydiagramrequirementembedding",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        server_default=None,
        existing_nullable=False,
    )
