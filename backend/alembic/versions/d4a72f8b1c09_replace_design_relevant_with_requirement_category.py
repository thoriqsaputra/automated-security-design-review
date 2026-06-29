"""replace_design_relevant_with_requirement_category

Revision ID: d4a72f8b1c09
Revises: c381a50d3feb
Create Date: 2026-06-28 14:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4a72f8b1c09'
down_revision: Union[str, Sequence[str], None] = 'c381a50d3feb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('standards_categoryparameterchild', 'design_relevant')
    op.add_column(
        'standards_categoryparameterchild',
        sa.Column('requirement_category', sa.String(20), server_default='design', nullable=False)
    )


def downgrade() -> None:
    op.drop_column('standards_categoryparameterchild', 'requirement_category')
    op.add_column(
        'standards_categoryparameterchild',
        sa.Column('design_relevant', sa.Boolean(), server_default=sa.true(), nullable=False)
    )
