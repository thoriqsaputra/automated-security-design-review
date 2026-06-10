"""
TSD Analysis Services — Modular, decoupled architecture.

Entry point: pipeline.run_tsd_analysis(review)
"""

from .dto import AnalysisSummary
from .ingestion_service import IngestionService
from .retrieval_service import RetrievalService
from .debate_service import DebateService
from .persistence_service import PersistenceService
from .pipeline import TSDAnalysisPipeline, run_tsd_analysis

__all__ = [
    "AnalysisSummary",
    "IngestionService",
    "RetrievalService",
    "DebateService",
    "PersistenceService",
    "TSDAnalysisPipeline",
    "run_tsd_analysis",
]