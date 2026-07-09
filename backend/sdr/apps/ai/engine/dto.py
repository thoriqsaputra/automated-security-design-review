from __future__ import annotations

import threading
from dataclasses import dataclass, field
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
from sdr.apps.standards.models import (
    CategoryParameterChild,
    StandardCategory,
    StandardIngestionJob,
)

DebatableParameter = CategoryParameterChild


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

    class Config:
        arbitrary_types_allowed = True


class DebateInput(BaseModel):
    """Input to DebateService for a single parameter."""
    parameter: DebatableParameter
    parameter_text: str
    parameter_section: str
    contract: Dict[str, Any] = Field(default_factory=dict)
    killed_assumptions: List[Dict[str, Any]] = Field(default_factory=list)
    hunter_plan: Dict[str, Any] = Field(default_factory=dict)
    retrieval_query_details: Dict[str, Any] = Field(default_factory=dict)
    context_chunks: List[str]
    original_context_chunks: List[str] = Field(default_factory=list)
    context_chunk_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    retrieval_refresh_callback: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True


class DebateOutput(BaseModel):
    """Output from DebateService._run_debate() — NO database written yet."""
    parameter: DebatableParameter
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
    parameter: DebatableParameter
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
    category_stats: Dict[str, Any] = None
    llm_usage: Dict[str, Any] = None
    # Guards concurrent mutation when text debate and diagram debate run as
    # parallel driver threads sharing this same summary instance. Reentrant
    # so nested helper calls (e.g. progress recorders calling persist) don't
    # deadlock on the same thread.
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def __post_init__(self):
        if self.critical_findings is None:
            self.critical_findings = []
        if self.high_findings is None:
            self.high_findings = []
        if self.category_stats is None:
            self.category_stats = {}
        if self.llm_usage is None:
            self.llm_usage = {}

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
            "category_stats": self.category_stats,
            "llm_usage": self.llm_usage,
        }
