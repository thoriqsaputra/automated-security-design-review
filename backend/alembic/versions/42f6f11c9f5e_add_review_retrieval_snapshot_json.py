"""add review retrieval snapshot json

Revision ID: 42f6f11c9f5e
Revises: e2c9afd4e1df
Create Date: 2026-06-10 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "42f6f11c9f5e"
down_revision = "e2c9afd4e1df"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reviews_review",
        sa.Column("retrieval_snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("reviews_review", "retrieval_snapshot_json")
