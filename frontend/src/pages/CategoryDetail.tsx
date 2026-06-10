import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, ChevronDown, ChevronRight, CheckCircle2, Zap, Trash2, Ban, Upload } from 'lucide-react';
import Card from '../components/ui/Card';
import Modal from '../components/ui/Modal';
import StatusBadge from '../components/ui/StatusBadge';
import {
  getCategoryParameters,
  listIngestionJobs,
  activateIngestionJob,
  cancelIngestionJob,
  createIngestionJob,
  deleteIngestionJob,
  deleteParameterParent,
  deleteParameterChild,
  type ParameterParent,
  type StandardCategory,
  type IngestionJob,
} from '../api/standards';

export default function CategoryDetail() {
  const { code } = useParams<{ code: string }>();
  const navigate = useNavigate();
  const [category, setCategory] = useState<StandardCategory | null>(null);
  const [parameters, setParameters] = useState<ParameterParent[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedParent, setExpandedParent] = useState<number | null>(null);

  const [showUpload, setShowUpload] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [startPage, setStartPage] = useState('');
  const [endPage, setEndPage] = useState('');
  const [uploading, setUploading] = useState(false);

  const [jobsPage, setJobsPage] = useState(1);
  const [paramsPage, setParamsPage] = useState(1);
  const JOBS_PER_PAGE = 6;
  const PARAMS_PER_PAGE = 5;

  const loadInitial = () => {
    if (!code) return;
    setJobsPage(1);
    setParamsPage(1);
    setLoading(true);
    return Promise.all([
      getCategoryParameters(code).then(r => {
        setCategory(r.data.category);
        setParameters(r.data.parameters);
      }),
      listIngestionJobs(code).then(r => setJobs(r.data)),
    ]).finally(() => setLoading(false));
  };

  const pollJobs = () => {
    if (!code) return;
    listIngestionJobs(code).then(r => setJobs(r.data));
  };

  useEffect(() => { loadInitial(); }, [code]);

  useEffect(() => {
    const hasRunningJob = jobs.some(j => j.status === 'running' || j.status === 'pending');
    if (!hasRunningJob) return;

    const interval = setInterval(() => {
      pollJobs();
    }, 3000);

    return () => clearInterval(interval);
  }, [code, jobs]);

  const handleActivate = async (jobId: number) => {
    await activateIngestionJob(jobId);
    if (!code) return;
    Promise.all([
      getCategoryParameters(code).then(r => {
        setCategory(r.data.category);
        setParameters(r.data.parameters);
      }),
      listIngestionJobs(code).then(r => setJobs(r.data))
    ]);
  };

  const handleUpload = async () => {
    if (!code || !file) return;
    setUploading(true);
    try {
      await createIngestionJob(code, file, startPage, endPage);
      setShowUpload(false);
      setFile(null);
      setStartPage('');
      setEndPage('');
      loadInitial();
    } catch (error: any) {
      alert(`Failed to start ingestion: ${error.response?.data?.detail || error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteJob = async (jobId: number) => {
    if (window.confirm('Are you sure you want to delete this ingestion job?')) {
      try {
        await deleteIngestionJob(jobId);
        loadInitial();
      } catch (error: any) {
        alert(`Failed to delete job: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  const handleCancelJob = async (jobId: number) => {
    if (window.confirm('Are you sure you want to cancel this ingestion job?')) {
      try {
        await cancelIngestionJob(jobId);
        pollJobs();
      } catch (error: any) {
        alert(`Failed to cancel job: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  const handleDeleteParent = async (parentId: number) => {
    if (window.confirm('Are you sure you want to delete this parameter section?')) {
      try {
        await deleteParameterParent(parentId);
        loadInitial();
      } catch (error: any) {
        alert(`Failed to delete section: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  const handleDeleteChild = async (childId: number) => {
    if (window.confirm('Are you sure you want to delete this requirement?')) {
      try {
        await deleteParameterChild(childId);
        loadInitial();
      } catch (error: any) {
        alert(`Failed to delete requirement: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-flame border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const paginatedJobs = jobs.slice((jobsPage - 1) * JOBS_PER_PAGE, jobsPage * JOBS_PER_PAGE);
  const totalJobsPages = Math.ceil(jobs.length / JOBS_PER_PAGE);

  const paginatedParams = parameters.slice((paramsPage - 1) * PARAMS_PER_PAGE, paramsPage * PARAMS_PER_PAGE);
  const totalParamsPages = Math.ceil(parameters.length / PARAMS_PER_PAGE);

  return (
    <div className="space-y-6 animate-slide-up">
      <button onClick={() => navigate('/standards')} className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors">
        <ArrowLeft size={16} /> Back to Standards
      </button>

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">{category?.name || code}</h1>
          <p className="text-sm text-text-muted mt-1">{category?.description || `Category: ${code}`}</p>
        </div>
        <button
          onClick={() => setShowUpload(true)}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          <Upload size={16} /> Ingest Standard
        </button>
      </div>

      {/* Ingestion Jobs */}
      <div>
        <h2 className="text-lg font-semibold text-text-primary mb-3">Ingestion Jobs</h2>
        {jobs.length === 0 ? (
          <Card><p className="text-sm text-text-muted text-center py-4">No ingestion jobs for this category.</p></Card>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              {paginatedJobs.map(job => (
                <Card key={job.id}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div>
                        <p className="text-sm font-medium text-text-primary">
                          Version {job.version_no}
                          {job.is_active && (
                            <span className="ml-2 text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400">
                              Active
                            </span>
                          )}
                        </p>
                        <p className="text-xs text-text-muted">
                          {job.source_documents.map(d => d.name).join(', ') || 'No documents'}
                          {' · '}{new Date(job.created_at).toLocaleDateString()}
                          {job.status === 'completed' && (
                            <>
                              {' · '}
                              {job.is_active
                                ? `${parameters.reduce((sum, p) => sum + p.children.length, 0)} params`
                                : `${(job.summary_json as any)?.inserted || 0} params`}
                            </>
                          )}
                        </p>
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
                      <div className="flex items-center gap-3">
                          <StatusBadge status={job.status} />
                          {job.status === 'completed' && !job.is_active && (
                            <button
                              onClick={() => handleActivate(job.id)}
                              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 text-emerald-400 text-xs font-medium hover:bg-emerald-500/25 transition-colors"
                            >
                              <Zap size={12} /> Activate
                            </button>
                          )}
                        {!job.is_active && (
                          <button
                            onClick={() => handleDeleteJob(job.id)}
                            className="flex items-center justify-center p-1.5 rounded-lg text-text-muted hover:text-flame hover:bg-flame/10 transition-colors"
                            title="Delete Job"
                          >
                            <Trash2 size={16} />
                          </button>
                        )}
                        {(job.status === 'pending' || job.status === 'running') && (
                          <button
                            onClick={() => handleCancelJob(job.id)}
                            className="flex items-center justify-center p-1.5 rounded-lg text-text-muted hover:text-amber-500 hover:bg-amber-500/10 transition-colors"
                            title="Cancel Job"
                          >
                            <Ban size={16} />
                          </button>
                        )}
                      </div>
                      {(job.status === 'running' || job.status === 'pending') && job.progress && (
                        <div className="w-full flex flex-col items-end gap-1 mt-1">
                          <span className="text-xs text-text-muted">{job.progress.status_label || `${job.progress.percentage}%`}</span>
                          <div className="w-32 h-1.5 bg-surface-border rounded-full overflow-hidden">
                            <div
                              className="h-full bg-flame transition-all duration-500 ease-in-out"
                              style={{ width: `${job.progress.percentage}%` }}
                            />
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
            {totalJobsPages > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button
                  onClick={() => setJobsPage(p => Math.max(1, p - 1))}
                  disabled={jobsPage === 1}
                  className="px-3 py-1.5 text-xs font-medium bg-surface-base border border-surface-border rounded-lg disabled:opacity-50 text-text-primary hover:bg-surface-hover transition-colors"
                >
                  Previous
                </button>
                <span className="text-xs text-text-muted">Page {jobsPage} of {totalJobsPages}</span>
                <button
                  onClick={() => setJobsPage(p => Math.min(totalJobsPages, p + 1))}
                  disabled={jobsPage === totalJobsPages}
                  className="px-3 py-1.5 text-xs font-medium bg-surface-base border border-surface-border rounded-lg disabled:opacity-50 text-text-primary hover:bg-surface-hover transition-colors"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Parameters Tree */}
      <div>
        <h2 className="text-lg font-semibold text-text-primary mb-3">
          Extracted Parameters ({parameters.length} sections)
        </h2>
        {parameters.length === 0 ? (
          <Card><p className="text-sm text-text-muted text-center py-4">No parameters extracted yet. Complete an ingestion job first.</p></Card>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2">
              {paginatedParams.map(parent => (
                <Card key={parent.id}>
                  <div className="flex items-center justify-between">
                    <button
                      onClick={() => setExpandedParent(expandedParent === parent.id ? null : parent.id)}
                      className="flex-1 flex items-center justify-between text-left"
                    >
                      <div className="flex items-center gap-2">
                        {expandedParent === parent.id ? (
                          <ChevronDown size={16} className="text-flame" />
                        ) : (
                          <ChevronRight size={16} className="text-text-muted" />
                        )}
                        <div>
                          <p className="text-sm font-medium text-text-primary">{parent.title}</p>
                          <p className="text-xs text-text-muted">{parent.children.length} requirement(s)</p>
                        </div>
                      </div>
                      <span className="text-xs text-text-muted font-mono">{parent.stable_key}</span>
                    </button>
                    <button
                      onClick={() => handleDeleteParent(parent.id)}
                      className="ml-3 flex items-center justify-center p-1.5 rounded-lg text-text-muted hover:text-flame hover:bg-flame/10 transition-colors shrink-0"
                      title="Delete Section"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>

                  {expandedParent === parent.id && parent.children.length > 0 && (
                    <div className="mt-3 ml-6 space-y-2 border-l-2 border-surface-border pl-4">
                      {parent.children.map(child => (
                        <div key={child.id} className="flex items-start justify-between gap-4 group">
                          <div className="flex items-start gap-2">
                            <CheckCircle2 size={14} className="text-burgundy mt-0.5 shrink-0" />
                            <div>
                              <p className="text-sm text-text-primary">{child.requirement_text}</p>
                              {child.details && (
                                <p className="text-xs text-text-muted mt-0.5">{child.details}</p>
                              )}
                            </div>
                          </div>
                          <button
                            onClick={() => handleDeleteChild(child.id)}
                            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-flame/10 text-text-muted hover:text-flame transition-all shrink-0"
                            title="Delete Requirement"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              ))}
            </div>
            {totalParamsPages > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button
                  onClick={() => setParamsPage(p => Math.max(1, p - 1))}
                  disabled={paramsPage === 1}
                  className="px-3 py-1.5 text-xs font-medium bg-surface-base border border-surface-border rounded-lg disabled:opacity-50 text-text-primary hover:bg-surface-hover transition-colors"
                >
                  Previous
                </button>
                <span className="text-xs text-text-muted">Page {paramsPage} of {totalParamsPages}</span>
                <button
                  onClick={() => setParamsPage(p => Math.min(totalParamsPages, p + 1))}
                  disabled={paramsPage === totalParamsPages}
                  className="px-3 py-1.5 text-xs font-medium bg-surface-base border border-surface-border rounded-lg disabled:opacity-50 text-text-primary hover:bg-surface-hover transition-colors"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Upload Modal */}
      <Modal open={showUpload} onClose={() => setShowUpload(false)} title={`Ingest Standard: ${category?.name || code}`}>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Standard Document (PDF)</label>
            <input
              type="file"
              accept=".pdf"
              onChange={e => setFile(e.target.files?.[0] || null)}
              className="w-full text-sm text-text-muted file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-midnight-lighter file:text-text-secondary file:font-medium file:cursor-pointer hover:file:bg-surface-hover file:transition-colors"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">Start Page (Optional)</label>
              <input
                type="number"
                min="1"
                placeholder="e.g. 12"
                value={startPage}
                onChange={e => setStartPage(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">End Page (Optional)</label>
              <input
                type="number"
                min="1"
                placeholder="e.g. 55"
                value={endPage}
                onChange={e => setEndPage(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
              />
            </div>
          </div>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
          >
            {uploading ? 'Ingesting...' : 'Start Ingestion'}
          </button>
        </div>
      </Modal>
    </div>
  );
}
