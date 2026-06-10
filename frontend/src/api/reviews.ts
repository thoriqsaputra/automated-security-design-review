import api from './client';

export interface Finding {
  id: number;
  review_id: number;
  category_name: string | null;
  category_code: string | null;
  parent_parameter_title: string | null;
  child_parameter_stable_key: string | null;
  finding_type: string;
  status: string;
  met_status: string | null;
  severity: string | null;
  severity_score: number | null;
  confidence_score: number | null;
  title: string;
  description: string;
  reason: string | null;
  recommendation: string | null;
  hunter_reasoning: string | null;
  critic_reasoning: string | null;
  mediator_reasoning: string | null;
  requirement_reference: string | null;
  requirement_text: string | null;
  is_actionable: boolean;
  citation_count: number;
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
  summary_json: Record<string, unknown>;
  overview: string | null;
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

export const createReview = (designId: number, categoryId: number) =>
  api.post<Review>('/reviews/', { design_id: designId, category_id: categoryId });

export const triggerReview = (id: number) =>
  api.post<Review>(`/reviews/${id}/trigger`);

export const cancelReview = (id: number) =>
  api.post<Review>(`/reviews/${id}/cancel`);

export const getFindings = (reviewId: number) =>
  api.get<Finding[]>(`/reviews/${reviewId}/findings`);
