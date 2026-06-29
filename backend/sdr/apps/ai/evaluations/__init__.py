"""
Evaluation suite for the automated security design review system.

Structure:
  extraction/   — Standard ingestion quality evals (purity, coverage, category accuracy)
  debate/       — Multi-agent debate evals (ablation, citation trace, dynamics)
  retrieval/    — RAG retrieval evals (vector vs hybrid ablation, cross-boundary)
  vision/       — Vision agent evals (blindness mitigation)
  shared/       — Shared judges, metrics, and dataset utilities
  data/         — Ground truth files and gold standard ID sets
  results/      — Output directory (mounted as Docker volume from host)
"""
