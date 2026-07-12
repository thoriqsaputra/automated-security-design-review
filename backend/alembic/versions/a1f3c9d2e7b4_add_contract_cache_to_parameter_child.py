"""add_contract_cache_to_parameter_child

Revision ID: a1f3c9d2e7b4
Revises: 1942011faed3
Create Date: 2026-07-12 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a1f3c9d2e7b4'
down_revision: Union[str, Sequence[str], None] = '1942011faed3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('standards_categoryparameterchild', sa.Column('synthesized_contract', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('standards_categoryparameterchild', sa.Column('contract_synthesized_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('standards_categoryparameterchild', sa.Column('contract_source_hash', sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('standards_categoryparameterchild', 'contract_source_hash')
    op.drop_column('standards_categoryparameterchild', 'contract_synthesized_at')
    op.drop_column('standards_categoryparameterchild', 'synthesized_contract')
