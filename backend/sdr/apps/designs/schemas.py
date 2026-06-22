from datetime import datetime
from typing import Optional, Any

from pydantic import BaseModel, ConfigDict, Field


class DesignSchema(BaseModel):
    id: int
    name: str
    document: str
    source_format: str
    original_filename: str
    created_at: datetime
    updated_at: datetime
    status: str
    processing_error: Optional[str] = None
    document_sha256: Optional[str] = None
    prepared_document_sha256: Optional[str] = None
    preparation_status: str = Field(default="queued")
    preparation_error: Optional[str] = None
    prepared_at: Optional[datetime] = None
    active_preparation_id: Optional[int] = None
    preparation_snapshot_json: Optional[dict] = None
    preparation_progress: Optional[dict] = None
    can_start_analysis: bool = False

    model_config = ConfigDict(from_attributes=True)


class DesignWithStatusSchema(DesignSchema):
    review_status: str = Field(default="no_review")
    review_id: Optional[str] = Field(default=None)
    review_has_unmet_findings: bool = Field(default=False)
    review_queue_position: Optional[int] = Field(default=None)
    review_queue_size: Optional[int] = Field(default=None)
    review_queue_state: str = Field(default="none")

    model_config = ConfigDict(from_attributes=True)


class DesignDetailSchema(DesignWithStatusSchema):
    has_review: bool = Field(default=False)
    # We use Any here to avoid circular dependency issues if ReviewSchema isn't available yet.
    # In a full migration, this should be ReviewSchema.
    review: Optional[Any] = Field(default=None)

    model_config = ConfigDict(from_attributes=True)
