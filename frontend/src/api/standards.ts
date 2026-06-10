import api from './client';

export interface StandardCategory {
  id: number;
  code: string;
  name: string;
  description: string | null;
  is_active: boolean;
  active_parameters_count: number;
  active_job_version: number | null;
  created_at: string;
  updated_at: string;
}

export interface ParameterChild {
  id: number;
  stable_key: string;
  requirement_text: string;
  details: string;
  requirement_text_normalized: string;
  ordinal: number;
}

export interface ParameterParent {
  id: number;
  stable_key: string;
  title: string;
  title_normalized: string;
  description: string | null;
  children: ParameterChild[];
}

export interface SourceDocument {
  id: number;
  name: string;
  status: string;
  content_hash: string;
  created_at: string;
  updated_at: string;
}

export interface IngestionJob {
  id: number;
  category: StandardCategory | null;
  status: string;
  version_no: number;
  is_active: boolean;
  activated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  summary: Record<string, unknown>;
  progress: Record<string, unknown>;
  summary_json: Record<string, unknown>;
  error_message: string | null;
  source_documents: SourceDocument[];
  created_at: string;
  updated_at: string;
}

export interface CategoryParams {
  category: StandardCategory;
  parameters: ParameterParent[];
}

export const listCategories = () => api.get<StandardCategory[]>('/standards/categories/');

export const getCategoryParameters = (code: string) =>
  api.get<CategoryParams>(`/standards/categories/${code}/parameters`);

export const listIngestionJobs = (categoryCode?: string) =>
  api.get<IngestionJob[]>('/standards/ingestion/', {
    params: categoryCode ? { category_code: categoryCode } : undefined,
  });

export const getIngestionJob = (id: number) =>
  api.get<IngestionJob>(`/standards/ingestion/${id}`);

export const createIngestionJob = (categoryCode: string, file: File, startPage?: string, endPage?: string) => {
  const form = new FormData();
  form.append('category_code', categoryCode);
  form.append('document', file);
  if (startPage) form.append('start_page', startPage);
  if (endPage) form.append('end_page', endPage);
  return api.post<IngestionJob>('/standards/ingestion/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const activateIngestionJob = (id: number) =>
  api.post<IngestionJob>(`/standards/ingestion/${id}/activate`);

export const cancelIngestionJob = (id: number) =>
  api.post<IngestionJob>(`/standards/ingestion/${id}/cancel`);

export const deleteIngestionJob = (id: number) =>
  api.delete(`/standards/ingestion/${id}`);

export const deleteParameterParent = (parentId: number) =>
  api.delete(`/standards/categories/parameters/parent/${parentId}`);

export const deleteParameterChild = (childId: number) =>
  api.delete(`/standards/categories/parameters/child/${childId}`);
