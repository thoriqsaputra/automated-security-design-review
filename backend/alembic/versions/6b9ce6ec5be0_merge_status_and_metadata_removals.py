"""merge status and metadata removals

Revision ID: 6b9ce6ec5be0
Revises: 4b9a1add11bd, a1b2c3d4e5f6
Create Date: 2026-06-11 17:28:45.513761

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6b9ce6ec5be0'
down_revision: Union[str, Sequence[str], None] = ('4b9a1add11bd', 'a1b2c3d4e5f6')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
