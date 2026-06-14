from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from sdr.apps.ai.agents.base import (
    HunterResult,
    CriticResult,
    MediatorResult,
    Citation,
)
from sdr.apps.ai.retrieval.core import RetrievalResult
from sdr.apps.ai.tsd_processing.ingestor import TSDDocument
from sdr.apps.ai.tsd_processing.raptor import RAPTORTree
from sdr.apps.ai.tsd_processing.graph_builder import TSDGraph
from sdr.apps.standards.models import (
    CategoryParameterChild,
    CategoryParameterParent,
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

    class Config:
        arbitrary_types_allowed = True


class DebateOutput(BaseModel):
    """Output from DebateService._run_debate() — NO database written yet."""
    parameter: CategoryParameterChild
    hunter_result: HunterResult
    critic_result: CriticResult
    mediator_result: MediatorResult
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

    class Config:
        arbitrary_types_allowed = True


DebateOutput.model_rebuild()


@dataclass
class AnalysisSummary:
    """Aggregated statistics for a completed TSD analysis run."""
    current_stage: str = "1_ingestion"
    total_parameters: int = 0
    debate_total_parameters: int = 0
    debate_completed_parameters: int = 0
    debate_remaining_parameters: int = 0
    persistence_total_parameters: int = 0
    persistence_completed_parameters: int = 0
    persistence_remaining_parameters: int = 0
    analysis_total_parameters: int = 0
    analysis_processed_parameters: int = 0
    analysis_remaining_parameters: int = 0
    met_count: int = 0
    not_met_count: int = 0
    na_count: int = 0
    error_count: int = 0
    diagram_findings_count: int = 0
    citation_count: int = 0
    screened_out: bool = False
    critical_findings: List[str] = None
    high_findings: List[str] = None
    asvs: Dict[str, Any] = None
    applicability: Dict[str, Any] = None

    def __post_init__(self):
        if self.critical_findings is None:
            self.critical_findings = []
        if self.high_findings is None:
            self.high_findings = []
        if self.asvs is None:
            self.asvs = {}
        if self.applicability is None:
            self.applicability = {
                "parents_total": 0,
                "parents_applicable": 0,
                "parents_not_applicable": 0,
                "children_marked_na_by_parent": 0,
                "parents": [],
            }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "total_parameters": self.total_parameters,
            "debate_total_parameters": self.debate_total_parameters,
            "debate_completed_parameters": self.debate_completed_parameters,
            "debate_remaining_parameters": self.debate_remaining_parameters,
            "persistence_total_parameters": self.persistence_total_parameters,
            "persistence_completed_parameters": self.persistence_completed_parameters,
            "persistence_remaining_parameters": self.persistence_remaining_parameters,
            "analysis_total_parameters": self.analysis_total_parameters,
            "analysis_processed_parameters": self.analysis_processed_parameters,
            "analysis_remaining_parameters": self.analysis_remaining_parameters,
            "met_count": self.met_count,
            "not_met_count": self.not_met_count,
            "na_count": self.na_count,
            "error_count": self.error_count,
            "diagram_findings_count": self.diagram_findings_count,
            "citation_count": self.citation_count,
            "screened_out": self.screened_out,
            "critical_findings": self.critical_findings[:10],
            "high_findings": self.high_findings[:10],
            "critical_findings_count": len(self.critical_findings),
            "high_findings_count": len(self.high_findings),
            "asvs": self.asvs,
            "applicability": self.applicability,
        }
