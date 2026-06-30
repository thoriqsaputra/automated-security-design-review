from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from enum import Enum

class CategoryCodeEnum(str, Enum):
    web_application = "web_application"
    mobile = "mobile"

class CategoryCodeWithAutoEnum(str, Enum):
    web_application = "web_application"
    mobile = "mobile"
    auto = "auto"


# ---------------------------------------------------------------------------
# Summary JSON contract
# ---------------------------------------------------------------------------

class IngestionSummarySchema(BaseModel):
    """
    Typed contract for StandardIngestionJob.summary_json.
    """
    inserted: int = Field(default=0)
    sections: int = Field(default=0)
    migrated: int = Field(default=0)
    skipped: int = Field(default=0)
    errors: int = Field(default=0)
    embeddings_created: int = Field(default=0)
    embeddings_failed: int = Field(default=0)
    diagram_requirement_embeddings_created: int = Field(default=0)
    diagram_requirement_embeddings_failed: int = Field(default=0)
    mode: str = Field(default="manual")
    version_no: int = Field(default=1)
    resolved_categories: Dict[str, int] = Field(default_factory=dict)
    celery_task_id: Optional[str] = Field(default=None)
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    page_detection: Dict[str, Any] = Field(default_factory=dict)


class IngestionProgressSchema(BaseModel):
    phase: str = Field(default="queued")
    percentage: int = Field(default=0)
    total_document: int = Field(default=0)
    uploaded_document: int = Field(default=0)
    parsed_document: int = Field(default=0)
    processed_document: int = Field(default=0)
    failed_document: int = Field(default=0)
    status_label: str = Field(default="Queued")


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------

class StandardCategorySchema(BaseModel):
    id: int
    code: str
    name: str
    description: Optional[str] = None
    is_active: bool
    active_parameters_count: int = 0
    active_job_version: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class CategoryParameterChildSchema(BaseModel):
    id: int
    stable_key: str
    requirement_text: str

    requirement_text_normalized: str
    ordinal: int
    requirement_category: str = "design"
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)


class CategoryParameterParentSchema(BaseModel):
    id: int
    stable_key: str
    title: str
    title_normalized: str
    description: Optional[str] = None
    is_active: bool = True
    children: List[CategoryParameterChildSchema] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class CategoryDiagramRequirementSchema(BaseModel):
    id: int
    stable_key: str
    source_requirement_key: str
    requirement_text: str
    verification_hint: str
    parent_section: str
    diagram_type: str = ""
    ordinal: int = 0

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Ingestion Job
# ---------------------------------------------------------------------------

class StandardSourceDocumentSchema(BaseModel):
    id: int
    name: str
    status: str
    content_hash: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class StandardIngestionJobSchema(BaseModel):
    id: int
    category: Optional[StandardCategorySchema] = None
    requested_by_email: Optional[EmailStr] = None
    status: str
    version_no: int
    is_active: bool
    activated_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    summary: IngestionSummarySchema
    progress: IngestionProgressSchema
    summary_json: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    source_documents: List[StandardSourceDocumentSchema] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
