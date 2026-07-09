"""add document snapshot fields to review

Revision ID: 1942011faed3
Revises: 706007a24faf
Create Date: 2026-07-08 10:55:00.620936

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1942011faed3'
down_revision: Union[str, Sequence[str], None] = '706007a24faf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('reviews_review', sa.Column('document_object_key', sa.String(length=1024), nullable=True))
    op.add_column('reviews_review', sa.Column('document_filename', sa.String(length=255), nullable=True))
    op.add_column('reviews_review', sa.Column('document_sha256', sa.String(length=64), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('reviews_review', 'document_sha256')
    op.drop_column('reviews_review', 'document_filename')
    op.drop_column('reviews_review', 'document_object_key')
