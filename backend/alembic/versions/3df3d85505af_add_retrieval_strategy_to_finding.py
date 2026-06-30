"""add retrieval_strategy to finding

Revision ID: 3df3d85505af
Revises: 83864b582e8b
Create Date: 2026-06-29

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '3df3d85505af'
down_revision: Union[str, Sequence[str], None] = '83864b582e8b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('reviews_finding', sa.Column('retrieval_strategy', sa.String(length=32), nullable=True))
    op.create_index('ix_reviews_finding_retrieval_strategy', 'reviews_finding', ['retrieval_strategy'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_reviews_finding_retrieval_strategy', table_name='reviews_finding')
    op.drop_column('reviews_finding', 'retrieval_strategy')
