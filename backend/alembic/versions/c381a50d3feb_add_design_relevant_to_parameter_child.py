"""add_design_relevant_to_parameter_child

Revision ID: c381a50d3feb
Revises: 7dd0b1900779
Create Date: 2026-06-27 09:58:54.882318

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c381a50d3feb'
down_revision: Union[str, Sequence[str], None] = '7dd0b1900779'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'standards_categoryparameterchild',
        sa.Column('design_relevant', sa.Boolean(), server_default=sa.true(), nullable=False)
    )


def downgrade() -> None:
    op.drop_column('standards_categoryparameterchild', 'design_relevant')
