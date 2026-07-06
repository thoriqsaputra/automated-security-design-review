import api from './client';
import type { RetrievalVisualization } from './reviews';

export interface Design {
  id: number;
  name: string;
  document: string;
  source_format: string;
  original_filename: string;
  created_at: string;
  updated_at: string;
  status: string;
  processing_error: string | null;
  document_sha256: string | null;
  prepared_document_sha256: string | null;
  preparation_status: string;
  preparation_error: string | null;
  prepared_at: string | null;
  active_preparation_id: number | null;
  preparation_snapshot_json: RetrievalVisualization | null;
  preparation_progress: Record<string, unknown> | null;
  can_start_analysis: boolean;
}

export interface DesignDetail extends Design {
  review_status: string;
  review_id: number | null;
  review_has_unmet_findings: boolean;
  has_review: boolean;
  review: unknown;
}

export const listDesigns = () => api.get<Design[]>('/designs/');

export const getDesign = (id: number) => api.get<DesignDetail>(`/designs/${id}`);

export const createDesign = (name: string, file: File) => {
  const form = new FormData();
  form.append('name', name);
  form.append('document', file);
  return api.post<Design>('/designs/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const deleteDesign = (id: number) => api.delete(`/designs/${id}`);

export const retryDesignPreparation = (id: number) =>
  api.post<DesignDetail>(`/designs/${id}/prepare`);

export const cancelDesignPreparation = (id: number) =>
  api.post<DesignDetail>(`/designs/${id}/cancel-preparation`);
