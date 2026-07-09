import api from './client';

export type JsonRecord = Record<string, unknown>;
export type ReviewAnalysisMode = 'default' | 'text_only' | 'diagram_only';

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  size: number;
  total_pages: number;
}

export interface CitationAnchor {
  id: number;
  anchor_type: string;
  block_id: string;
  page_number: number;
  retrieval_origin: string | null;
  retrieval_origin_label: string | null;
  quoted_text: string | null;
  bbox_x0: number | null;
  bbox_y0: number | null;
  bbox_x1: number | null;
  bbox_y1: number | null;
  created_at: string;
}

export interface FindingEvidenceSource {
  key: string;
  label: string;
  count: number;
}

export interface Finding {
  id: number;
  review_id: number;
  category_id: number | null;
  parent_parameter_id: number | null;
  child_parameter_id: number | null;
  category_name: string | null;
  category_code: string | null;
  parent_parameter_title: string | null;
  child_parameter_stable_key: string | null;
  child_parameter_ordinal: number | null;
  finding_type: string;
  met_status: string | null;
  severity: string | null;
  severity_score: number | null;
  severity_analysis: JsonRecord | null;
  confidence_score: number | null;
  title: string;
  description: string;
  reason: string | null;
  recommendation: string | null;
  hunter_reasoning: string | null;
  critic_reasoning: string | null;
  mediator_reasoning: string | null;
  hunter_thought_process: string | null;
  critic_thought_process: string | null;
  mediator_thought_process: string | null;
  diagram_id: string | null;
  diagram_caption: string | null;
  diagram_image_url: string | null;
  vision_reasoning: string | null;
  vision_thought_process: string | null;
  requirement_reference: string | null;
  requirement_text: string | null;
  requirement_metadata: JsonRecord | null;
  is_actionable: boolean;
  has_citations: boolean;
  citation_count: number;
  evidence_sources: FindingEvidenceSource[];
  citations: CitationAnchor[];
  created_at: string;
  updated_at: string;
}

export interface ReviewProgress {
  stage: string;
  label: string;
  total_items: number;
  completed_items: number;
  failed_items: number;
  remaining_items: number;
  progress_percent: number;
}

export interface RaptorNodeSnapshot {
  id: string;
  parent_id: string | null;
  level: number;
  section_heading: string | null;
  text_preview: string;
  page_numbers: number[];
  source_block_count: number;
  child_count: number;
}

export interface RaptorSnapshot {
  status: string;
  total_nodes: number;
  max_level: number;
  root_node_id: string | null;
  nodes: RaptorNodeSnapshot[];
}

export interface RetrievalVisualization {
  status: string;
  generated_at: string | null;
  raptor: RaptorSnapshot | null;
}

export interface Review {
  id: number;
  design_id: number;
  design_name: string | null;
  category: Record<string, unknown> | null;
  status: string;
  celery_task_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  error_message: string | null;
  summary_json: JsonRecord;
  retrieval_snapshot_json?: RetrievalVisualization | null;
  overview: string | null;
  analysis_mode: ReviewAnalysisMode;
  document_url?: string | null;
  finding_counts: Record<string, number>;
  progress: ReviewProgress | null;
  created_at: string;
  updated_at: string;
}

export type DebateAgent = 'hunter' | 'critic' | 'mediator' | 'system';
export type DebateStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type DebateExecutionMode = 'single' | 'batch' | 'fallback';
export type DebateFindingType = 'requirement' | 'diagram';
export type CriticOutcome = 'UPHOLD' | 'OVERTURN' | 'PARTIAL';

export interface DebateTranscriptMessage {
  message_id: string;
  agent: DebateAgent;
  kind: string;
  status: DebateStatus | 'completed';
  content: string;
  started_at: string | null;
  completed_at: string | null;
  updated_at: string | null;
  critic_outcome?: CriticOutcome | null;
  requires_rebuttal?: boolean | null;
  round?: number | null;
}

export interface DebateStreamState {
  debate_id: string;
  finding_type: DebateFindingType;
  parameter_id: number | null;
  diagram_id: string | null;
  requirement_reference: string | null;
  requirement_text: string | null;
  section_title: string | null;
  category_code: string | null;
  status: DebateStatus;
  active_agent: DebateAgent | null;
  execution_mode: DebateExecutionMode;
  progress_percent: number;
  last_snippet: string;
  updated_at: string | null;
  finding_id: number | null;
  transcript: DebateTranscriptMessage[];
  critic_outcome?: CriticOutcome | null;
  requires_rebuttal?: boolean | null;
}

export interface DebateSnapshotPayload {
  review_id: number;
  review_status: string | null;
  error_message?: string | null;
  updated_at: string | null;
  last_event_id: string | null;
  debates: DebateStreamState[];
}

export interface DebateUpdatePayload {
  type: string;
  review_id: number;
  debate?: DebateStreamState;
  debates?: DebateStreamState[];
  review_status?: string | null;
  error_message?: string | null;
}

export const listReviews = (
  designId?: number,
  options?: {
    skip?: number;
    limit?: number;
  },
) =>
  api.get<Review[]>('/reviews/', {
    params: {
      ...(designId ? { design_id: designId } : {}),
      ...(options?.skip !== undefined ? { skip: options.skip } : {}),
      ...(options?.limit !== undefined ? { limit: options.limit } : {}),
    },
  });

export const getReview = (id: number) => api.get<Review>(`/reviews/${id}`);

export const createReview = (
  designId: number,
  categoryId: number,
  analysisMode: ReviewAnalysisMode = 'default',
) =>
  api.post<Review>('/reviews/', {
    design_id: designId,
    category_id: categoryId,
    analysis_mode: analysisMode,
  });

export const triggerReview = (id: number, analysisMode?: ReviewAnalysisMode) =>
  api.post<Review>(
    `/reviews/${id}/trigger`,
    analysisMode ? { analysis_mode: analysisMode } : undefined,
  );

export const cancelReview = (id: number) =>
  api.post<Review>(`/reviews/${id}/cancel`);

export const deleteReview = (id: number) => api.delete(`/reviews/${id}`);

export const getFindings = (
  reviewId: number, 
  page = 1, 
  size = 10,
  search?: string,
  metStatus?: string,
  severity?: string,
  findingType?: string
) =>
  api.get<PaginatedResponse<Finding>>(`/reviews/${reviewId}/findings`, { 
    params: { 
      page, 
      size,
      search: search || undefined,
      met_status: metStatus || undefined,
      severity: severity || undefined,
      finding_type: findingType || undefined,
    } 
  });

export const getRetrievalVisualization = (reviewId: number) =>
  api.get<RetrievalVisualization>(`/reviews/${reviewId}/retrieval-visualization`);

export const getReviewDebateStreamUrl = (reviewId: number) =>
  `/api/v1/reviews/${reviewId}/debates/stream`;
