from typing import Optional, Dict, Any

from sqlalchemy import String, Integer, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from sdr.core.database import Base
from .base import StandardsBigIntBase


class CategoryParameterParent(Base, StandardsBigIntBase):
    __tablename__ = "standards_categoryparameterparent"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("standards_standardcategory.id", ondelete="CASCADE"), index=True)
    ingestion_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("standards_standingestionjob.id", ondelete="CASCADE"), nullable=True, index=True)
    
    stable_key: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str] = mapped_column(String(255))
    title_normalized: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    category = relationship("StandardCategory", backref="parent_parameters")
    ingestion_job = relationship("StandardIngestionJob", backref=backref("parameter_parents", cascade="all, delete-orphan"))
    children = relationship("CategoryParameterChild", back_populates="parent", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("category_id", "stable_key", name="unique_parent_key_per_category"),
    )

    def __str__(self):
        # Fallback if relationship isn't loaded
        cat_name = getattr(self.category, "name", "Category") if self.category else "Category"
        return f"{cat_name}: {self.title}"



class ASVSLevelDefinition(Base, StandardsBigIntBase):
    __tablename__ = "standards_asvsleveldefinition"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ingestion_job_id: Mapped[int] = mapped_column(ForeignKey("standards_standingestionjob.id", ondelete="CASCADE"), index=True)
    level: Mapped[int] = mapped_column(Integer)
    code: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String)
    classification_guidance: Mapped[str] = mapped_column(String)
    source_quote: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    context_marker: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    ingestion_job = relationship("StandardIngestionJob", backref=backref("asvs_level_definitions", cascade="all, delete-orphan"))

    __table_args__ = (
        CheckConstraint("level IN (1, 2, 3)", name="ck_asvs_level_definition_range"),
        UniqueConstraint("ingestion_job_id", "level", name="unique_asvs_level_definition_per_job"),
    )

    def __str__(self):
        return f"{self.code}: {self.name}"


class CategoryParameterChild(Base, StandardsBigIntBase):
    __tablename__ = "standards_categoryparameterchild"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("standards_categoryparameterparent.id", ondelete="CASCADE"), index=True)
    asvs_level: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    
    stable_key: Mapped[str] = mapped_column(String(255), index=True)
    requirement_text: Mapped[str] = mapped_column(String)
    details: Mapped[str] = mapped_column(String, default="")
    requirement_text_normalized: Mapped[str] = mapped_column(String)
    ordinal: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    parent = relationship("CategoryParameterParent", back_populates="children")


    __table_args__ = (
        UniqueConstraint("parent_id", "stable_key", name="unique_child_key_per_parent"),
    )

    def __str__(self):
        parent_title = getattr(self.parent, "title", "Parent") if self.parent else "Parent"
        return f"{parent_title}: {self.requirement_text[:80]}"
