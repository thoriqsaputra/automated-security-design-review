import api from './client';

export type JsonRecord = Record<string, unknown>;

export interface CitationAnchor {
  id: number;
  anchor_type: string;
  block_id: string;
  page_number: number;
  quoted_text: string | null;
  bbox_x0: number | null;
  bbox_y0: number | null;
  bbox_x1: number | null;
  bbox_y1: number | null;
  created_at: string;
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

export interface GraphNodeSnapshot {
  id: string;
  label: string;
  entity_type: string;
  source_pages: number[];
  source_block_count: number;
  degree: number;
  in_degree: number;
  out_degree: number;
}

export interface GraphEdgeSnapshot {
  source: string;
  target: string;
  relation_type: string | null;
  confidence: number;
  protocol: string | null;
  is_encrypted: boolean | null;
  requires_auth: boolean | null;
  source_pages: number[];
}

export interface GraphSnapshot {
  status: string;
  total_entities: number;
  total_relations: number;
  nodes: GraphNodeSnapshot[];
  edges: GraphEdgeSnapshot[];
}

export interface RetrievalVisualization {
  status: string;
  generated_at: string | null;
  raptor: RaptorSnapshot | null;
  graph: GraphSnapshot | null;
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
  asvs_level_override: number | null;
  finding_counts: Record<string, number>;
  progress: ReviewProgress | null;
  created_at: string;
  updated_at: string;
}

export const listReviews = (designId?: number) =>
  api.get<Review[]>('/reviews/', {
    params: designId ? { design_id: designId } : undefined,
  });

export const getReview = (id: number) => api.get<Review>(`/reviews/${id}`);

export const createReview = (designId: number, categoryId: number, asvsLevelOverride?: number | null) =>
  api.post<Review>('/reviews/', {
    design_id: designId,
    category_id: categoryId,
    asvs_level_override: asvsLevelOverride ?? null,
  });

export const triggerReview = (id: number) =>
  api.post<Review>(`/reviews/${id}/trigger`);

export const cancelReview = (id: number) =>
  api.post<Review>(`/reviews/${id}/cancel`);

export const getFindings = (reviewId: number) =>
  api.get<Finding[]>(`/reviews/${reviewId}/findings`);

export const getRetrievalVisualization = (reviewId: number) =>
  api.get<RetrievalVisualization>(`/reviews/${reviewId}/retrieval-visualization`);
