"""Merge parallel migration heads

Revision ID: 89dc43087e8f
Revises: 8c82b8701ebc, a1b2c3d4e5f6
Create Date: 2026-06-16 22:13:27.896995

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '89dc43087e8f'
down_revision: Union[str, Sequence[str], None] = ('8c82b8701ebc', 'a1b2c3d4e5f6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
