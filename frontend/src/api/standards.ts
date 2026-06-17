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
  asvs_level: number | null;
  requirement_text: string;
  details: string;
  requirement_text_normalized: string;
  ordinal: number;
}

export interface ASVSLevel {
  level: number;
  code: string;
  name: string;
  description: string;
  classification_guidance: string;
}

export interface ASVSLevelDefinition extends ASVSLevel {
  id: number;
  ingestion_job_id: number;
  source_quote: string | null;
  context_marker: string | null;
  created_at: string;
  updated_at: string;
}

export interface ControlSummaryRequirement {
  id: number;
  stable_key: string;
  requirement_text: string;
  analysis_hint: string;
  asvs_level: number | null;
  covered_child_keys: string[];
  ordinal: number;
}

export interface ParameterParent {
  id: number;
  stable_key: string;
  title: string;
  title_normalized: string;
  description: string | null;
  children: ParameterChild[];
  control_summary_requirements: ControlSummaryRequirement[];
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
  progress: {
    status_label?: string;
    percentage?: number;
    [key: string]: unknown;
  };
  summary_json: Record<string, unknown>;
  error_message: string | null;
  source_documents: SourceDocument[];
  created_at: string;
  updated_at: string;
}

export interface DiagramRequirement {
  id: number;
  stable_key: string;
  asvs_level: number | null;
  requirement_text: string;
  verification_hint: string;
  parent_section: string;
}

export interface CategoryParams {
  category: StandardCategory;
  parameters: ParameterParent[];
  diagram_requirements?: DiagramRequirement[];
}

export const listCategories = () => api.get<StandardCategory[]>('/standards/categories/');

export const listAsvsLevels = () => api.get<ASVSLevel[]>('/standards/asvs-levels/');

export const getCategoryParameters = (code: string) =>
  api.get<CategoryParams>(`/standards/categories/${code}/parameters`);

export const listIngestionJobs = (categoryCode?: string) =>
  api.get<IngestionJob[]>('/standards/ingestion/', {
    params: categoryCode ? { category_code: categoryCode } : undefined,
  });

export const getIngestionJob = (id: number) =>
  api.get<IngestionJob>(`/standards/ingestion/${id}`);

export const createIngestionJob = (
  categoryCode: string,
  file: File,
  startPage?: string,
  endPage?: string,
  levelDefinitionStartPage?: string,
  levelDefinitionEndPage?: string,
) => {
  const form = new FormData();
  form.append('category_code', categoryCode);
  form.append('document', file);
  if (startPage) form.append('start_page', startPage);
  if (endPage) form.append('end_page', endPage);
  if (levelDefinitionStartPage) form.append('level_definition_start_page', levelDefinitionStartPage);
  if (levelDefinitionEndPage) form.append('level_definition_end_page', levelDefinitionEndPage);
  return api.post<IngestionJob>('/standards/ingestion/', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getIngestionJobAsvsLevelDefinitions = (id: number) =>
  api.get<ASVSLevelDefinition[]>(`/standards/ingestion/${id}/asvs-level-definitions`);

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
