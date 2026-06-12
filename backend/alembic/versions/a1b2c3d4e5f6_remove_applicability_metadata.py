"""Remove applicability_metadata_json columns

Revision ID: a1b2c3d4e5f6
Revises: 6b1c4fd9f2a1
Create Date: 2026-06-11 00:00:01.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "6b1c4fd9f2a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("standards_categoryparameterchild", "applicability_metadata_json")
    op.drop_column("standards_categoryparameterparent", "applicability_metadata_json")


def downgrade() -> None:
    op.add_column(
        "standards_categoryparameterparent",
        sa.Column(
            "applicability_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "standards_categoryparameterchild",
        sa.Column(
            "applicability_metadata_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.alter_column(
        "standards_categoryparameterparent",
        "applicability_metadata_json",
        server_default=None,
    )
    op.alter_column(
        "standards_categoryparameterchild",
        "applicability_metadata_json",
        server_default=None,
    )
