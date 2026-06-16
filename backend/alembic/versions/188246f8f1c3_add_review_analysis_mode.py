"""add review analysis mode

Revision ID: 188246f8f1c3
Revises: 6205890a3380
Create Date: 2026-06-14 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "188246f8f1c3"
down_revision = "6205890a3380"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "reviews_review",
        sa.Column(
            "analysis_mode",
            sa.String(length=24),
            nullable=False,
            server_default="default",
        ),
    )
    op.create_check_constraint(
        "ck_review_analysis_mode_valid",
        "reviews_review",
        "analysis_mode IN ('default', 'text_only', 'diagram_only')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_review_analysis_mode_valid", "reviews_review", type_="check")
    op.drop_column("reviews_review", "analysis_mode")
