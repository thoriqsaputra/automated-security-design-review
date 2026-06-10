import api from './client';

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
