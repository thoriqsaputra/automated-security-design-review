"""add asvs levels

Revision ID: b7c8d9e0f1a2
Revises: 43dc58b3541e
Create Date: 2026-06-11 00:00:02.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, Sequence[str], None] = "43dc58b3541e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "standards_asvslevel",
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("classification_guidance", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("level IN (1, 2, 3)", name="ck_asvs_level_range"),
        sa.PrimaryKeyConstraint("level"),
        sa.UniqueConstraint("code"),
    )
    op.create_index(op.f("ix_standards_asvslevel_code"), "standards_asvslevel", ["code"], unique=False)

    op.bulk_insert(
        sa.table(
            "standards_asvslevel",
            sa.column("level", sa.Integer()),
            sa.column("code", sa.String()),
            sa.column("name", sa.String()),
            sa.column("description", sa.String()),
            sa.column("classification_guidance", sa.String()),
        ),
        [
            {
                "level": 1,
                "code": "L1",
                "name": "Opportunistic",
                "description": "Baseline application security verification for common web applications.",
                "classification_guidance": (
                    "Use L1 when the TSD describes a standard application without high-value assets, "
                    "regulated data, strong adversary assumptions, or extensive defense-in-depth controls."
                ),
            },
            {
                "level": 2,
                "code": "L2",
                "name": "Standard",
                "description": "Security verification for applications containing sensitive data or requiring meaningful assurance.",
                "classification_guidance": (
                    "Use L2 when the TSD describes sensitive data, authenticated business workflows, "
                    "role-based access, payment or personal data processing, or a need for stronger "
                    "control coverage than an opportunistic baseline."
                ),
            },
            {
                "level": 3,
                "code": "L3",
                "name": "Advanced",
                "description": "High-assurance verification for critical applications and high-value targets.",
                "classification_guidance": (
                    "Use L3 when the TSD describes critical systems, high-value assets, safety or "
                    "mission impact, strong threat actors, strict regulatory obligations, or explicit "
                    "high-assurance security architecture."
                ),
            },
        ],
    )

    op.add_column("standards_categoryparameterchild", sa.Column("asvs_level", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_standards_categoryparameterchild_asvs_level",
        "standards_categoryparameterchild",
        "standards_asvslevel",
        ["asvs_level"],
        ["level"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_standards_categoryparameterchild_asvs_level"),
        "standards_categoryparameterchild",
        ["asvs_level"],
        unique=False,
    )

    op.add_column("reviews_review", sa.Column("asvs_level_override", sa.Integer(), nullable=True))
    op.create_check_constraint(
        "ck_review_asvs_level_override_range",
        "reviews_review",
        "asvs_level_override IS NULL OR asvs_level_override IN (1, 2, 3)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_review_asvs_level_override_range", "reviews_review", type_="check")
    op.drop_column("reviews_review", "asvs_level_override")
    op.drop_index(op.f("ix_standards_categoryparameterchild_asvs_level"), table_name="standards_categoryparameterchild")
    op.drop_constraint(
        "fk_standards_categoryparameterchild_asvs_level",
        "standards_categoryparameterchild",
        type_="foreignkey",
    )
    op.drop_column("standards_categoryparameterchild", "asvs_level")
    op.drop_index(op.f("ix_standards_asvslevel_code"), table_name="standards_asvslevel")
    op.drop_table("standards_asvslevel")
