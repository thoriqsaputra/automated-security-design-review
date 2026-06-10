# apps/ai/services/dto.py
"""
Data Transfer Objects (Pydantic models) for inter-service communication.
These define the contract between decoupled services.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from sdr.apps.ai.agents.base import (
    HunterResult,
    CriticResult,
    MediatorResult,
    VisionResult,
    Citation,
)
from sdr.apps.ai.retrieval.router import RetrievalResult
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph
from sdr.apps.standards.models import (
    CategoryParameterChild,
    StandardCategory,
    StandardIngestionJob,
)


class IngestionOutput(BaseModel):
    """Output from IngestionService._ingest_tsd()."""
    tsd_document: TSDDocument
    is_valid_tsd: bool = True
    screening_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class RetrievalIndexes(BaseModel):
    """Pre-built retrieval indexes from RetrievalService."""
    raptor_tree: Optional[RAPTORTree] = None
    tsd_graph: Optional[TSDGraph] = None

    class Config:
        arbitrary_types_allowed = True


class ParameterApplicabilityResult(BaseModel):
    """Result of pre-filtering parameters for applicability."""
    applicable_parameters: List[CategoryParameterChild]
    pre_filtered_parameters: List[CategoryParameterChild]
    pre_filter_details: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class DebateInput(BaseModel):
    """Input to DebateService for a single parameter."""
    parameter: CategoryParameterChild
    parameter_text: str
    parameter_section: str
    contract: Dict[str, Any] = Field(default_factory=dict)
    killed_assumptions: List[Dict[str, Any]] = Field(default_factory=list)
    hunter_plan: Dict[str, Any] = Field(default_factory=dict)
    retrieval_query_details: Dict[str, Any] = Field(default_factory=dict)
    context_chunks: List[str]
    context_chunk_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    diagram_captions: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class DebateOutput(BaseModel):
    """Output from DebateService._run_debate() — NO database written yet."""
    parameter: CategoryParameterChild
    hunter_result: HunterResult
    critic_result: CriticResult
    mediator_result: MediatorResult
    vision_results: List[tuple] = Field(default_factory=list)  # (DiagramInput, VisionResult)
    retrieval_result: Optional[RetrievalResult] = None
    debate_rounds: int = 1
    analysis_trace: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class PersistenceInput(BaseModel):
    """Input to PersistenceService for a single finding."""
    parameter: CategoryParameterChild
    category: StandardCategory
    ingestion_job: Optional[StandardIngestionJob]
    debate_output: DebateOutput
    is_pre_filtered: bool = False

    class Config:
        arbitrary_types_allowed = True


@dataclass
class AnalysisSummary:
    """Aggregated statistics for a completed TSD analysis run."""
    total_parameters: int = 0
    met_count: int = 0
    not_met_count: int = 0
    na_count: int = 0
    error_count: int = 0
    diagram_findings_count: int = 0
    citation_count: int = 0
    pre_filtered_count: int = 0
    screened_out: bool = False
    critical_findings: List[str] = None
    high_findings: List[str] = None

    def __post_init__(self):
        if self.critical_findings is None:
            self.critical_findings = []
        if self.high_findings is None:
            self.high_findings = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_parameters": self.total_parameters,
            "met_count": self.met_count,
            "not_met_count": self.not_met_count,
            "na_count": self.na_count,
            "error_count": self.error_count,
            "diagram_findings_count": self.diagram_findings_count,
            "citation_count": self.citation_count,
            "pre_filtered_count": self.pre_filtered_count,
            "screened_out": self.screened_out,
            "critical_findings": self.critical_findings[:10],
            "high_findings": self.high_findings[:10],
            "critical_findings_count": len(self.critical_findings),
            "high_findings_count": len(self.high_findings),
        }
