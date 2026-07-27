"""remove requirement category default

Revision ID: e7f4a2c91b6d
Revises: b2e4d8f1a9c3
Create Date: 2026-07-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e7f4a2c91b6d"
down_revision: Union[str, Sequence[str], None] = "b2e4d8f1a9c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Require every caller to provide an explicit LLM-assigned category."""
    op.alter_column(
        "standards_categoryparameterchild",
        "requirement_category",
        existing_type=sa.String(length=20),
        existing_nullable=False,
        server_default=None,
    )


def downgrade() -> None:
    """Restore the legacy implicit design category."""
    op.alter_column(
        "standards_categoryparameterchild",
        "requirement_category",
        existing_type=sa.String(length=20),
        existing_nullable=False,
        server_default="design",
    )
