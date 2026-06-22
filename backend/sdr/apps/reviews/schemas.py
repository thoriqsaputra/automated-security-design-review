from datetime import datetime
from typing import Optional, List, Dict, Any, TypeVar, Generic

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .models.choices import ReviewAnalysisMode

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    total_pages: int


# ---------------------------------------------------------------------------
# Citation Anchor
# ---------------------------------------------------------------------------

class CitationAnchorSchema(BaseModel):
    id: int
    anchor_type: str
    block_id: str
    page_number: int
    retrieval_origin: Optional[str] = None
    retrieval_origin_label: Optional[str] = None
    quoted_text: Optional[str] = None
    bbox_x0: Optional[float] = None
    bbox_y0: Optional[float] = None
    bbox_x1: Optional[float] = None
    bbox_y1: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

class FindingSchema(BaseModel):
    id: int
    review_id: int
    
    # Relationships
    category_id: Optional[int] = None
    parent_parameter_id: Optional[int] = None
    child_parameter_id: Optional[int] = None
    category_name: Optional[str] = None
    category_code: Optional[str] = None
    parent_parameter_title: Optional[str] = None
    child_parameter_stable_key: Optional[str] = None
    child_parameter_ordinal: Optional[int] = None
    
    # Classification
    finding_type: str
    met_status: Optional[str] = None
    severity: Optional[str] = None
    severity_score: Optional[float] = None
    severity_analysis: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    
    # Core content
    title: str
    description: str
    reason: Optional[str] = None
    recommendation: Optional[str] = None
    
    # Agent debate audit trail
    hunter_reasoning: Optional[str] = None
    critic_reasoning: Optional[str] = None
    mediator_reasoning: Optional[str] = None
    hunter_thought_process: Optional[str] = None
    critic_thought_process: Optional[str] = None
    mediator_thought_process: Optional[str] = None
    
    # Diagram-specific fields
    diagram_id: Optional[str] = None
    diagram_caption: Optional[str] = None
    vision_reasoning: Optional[str] = None
    vision_thought_process: Optional[str] = None

    @computed_field
    @property
    def diagram_image_url(self) -> Optional[str]:
        metadata = self.requirement_metadata or {}
        image_metadata = metadata.get("diagram_image") if isinstance(metadata, dict) else None
        if not isinstance(image_metadata, dict) or not image_metadata.get("object_name"):
            return None
        return f"/api/v1/reviews/findings/{self.id}/diagram-image"
    
    # Requirement traceability
    requirement_reference: Optional[str] = None
    requirement_text: Optional[str] = None
    requirement_metadata: Optional[Dict[str, Any]] = None
    
    # Computed
    is_actionable: bool = False
    has_citations: bool = False
    citation_count: int = 0
    evidence_sources: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Click-to-source
    citations: List[CitationAnchorSchema] = Field(default_factory=list)
    
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------

class ReviewCreateSchema(BaseModel):
    design_id: int
    category_id: int
    analysis_mode: ReviewAnalysisMode = Field(default=ReviewAnalysisMode.DEFAULT)


class ReviewTriggerSchema(BaseModel):
    analysis_mode: Optional[ReviewAnalysisMode] = None


class ReviewProgressSchema(BaseModel):
    stage: str
    label: str
    total_items: int
    completed_items: int
    failed_items: int
    remaining_items: int
    progress_percent: int
    current_parameter_reference: Optional[str] = None
    current_parameter_title: Optional[str] = None
    preparation: Optional[Dict[str, Any]] = None


class ReviewCategorySchema(BaseModel):
    id: int
    name: str
    code: str

    model_config = ConfigDict(from_attributes=True)


class ReviewSchema(BaseModel):
    id: int
    design_id: int
    design_name: Optional[str] = None
    
    category: Optional[ReviewCategorySchema] = None
    
    status: str
    celery_task_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    summary_json: Dict[str, Any] = Field(default_factory=dict)
    retrieval_snapshot_json: Optional[Dict[str, Any]] = None
    overview: Optional[str] = None
    analysis_mode: str = ReviewAnalysisMode.DEFAULT.value
    
    finding_counts: Dict[str, int] = Field(default_factory=dict)
    parent_rollups: List[Any] = Field(default_factory=list)
    progress: Optional[ReviewProgressSchema] = None
    
    queue_position: Optional[int] = None
    queue_size: Optional[int] = None
    queue_state: str = "none"

    @computed_field
    @property
    def document_url(self) -> Optional[str]:
        return f"/api/v1/reviews/{self.id}/document"
    
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
