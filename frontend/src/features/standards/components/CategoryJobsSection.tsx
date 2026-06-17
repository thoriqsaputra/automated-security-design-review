import { Ban, ShieldCheck, Trash2, Zap } from 'lucide-react';
import type { IngestionJob, ParameterParent } from '../../../api/standards';
import Card from '../../../components/ui/Card';
import PaginationControls from '../../../components/ui/PaginationControls';
import StatusBadge from '../../../components/ui/StatusBadge';
import { getAsvsSummaryForJob } from '../utils/standardsPresentation';

interface CategoryJobsSectionProps {
  jobs: IngestionJob[];
  paginatedJobs: IngestionJob[];
  totalJobsPages: number;
  jobsPage: number;
  onJobsPageChange: (page: number) => void;
  parameters: ParameterParent[];
  onActivate: (jobId: number) => void;
  onDelete: (jobId: number) => void;
  onCancel: (jobId: number) => void;
  onOpenDefinitions: (job: IngestionJob) => void;
}

export default function CategoryJobsSection({
  jobs,
  paginatedJobs,
  totalJobsPages,
  jobsPage,
  onJobsPageChange,
  parameters,
  onActivate,
  onDelete,
  onCancel,
  onOpenDefinitions,
}: CategoryJobsSectionProps) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-text-primary mb-3">Ingestion Jobs</h2>
      {jobs.length === 0 ? (
        <Card>
          <p className="text-sm text-text-muted text-center py-4">No ingestion jobs for this category.</p>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            {paginatedJobs.map((job) => {
              const asvsSummary = getAsvsSummaryForJob(job);
              return (
                <Card key={job.id}>
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
                      {job.source_documents.map((document) => document.name).join(', ') || 'No documents'}
                      {' · '}
                      {new Date(job.created_at).toLocaleDateString()}
                      {job.status === 'completed' && (
                        <>
                          {' · '}
                          {job.is_active
                            ? `${parameters.reduce((sum, parent) => sum + parent.children.length, 0)} params`
                            : `${Number((job.summary_json as { inserted?: number }).inserted || 0)} params`}
                        </>
                      )}
                    </p>
                    <div className="flex justify-between mt-3">
                      <div className="flex items-center gap-3">
                        <button
                          type="button"
                          onClick={() => onOpenDefinitions(job)}
                          className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors hover:bg-surface-hover ${asvsSummary.statusClass}`}
                        >
                          <ShieldCheck size={12} />
                          ASVS levels: {asvsSummary.extracted ? `extracted · ${asvsSummary.count}` : asvsSummary.status === 'fallback' ? 'fallback' : asvsSummary.status}
                        </button>
                        <StatusBadge status={job.status} />
                      </div>
                      <div className="flex items-center gap-3">
                        {job.status === 'completed' && !job.is_active && (
                          <button
                            onClick={() => onActivate(job.id)}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-500/15 text-emerald-400 text-xs font-medium hover:bg-emerald-500/25 transition-colors"
                          >
                            <Zap size={12} /> Activate
                          </button>
                        )}
                        {!job.is_active && (
                          <button
                            onClick={() => {
                              if (window.confirm('Are you sure you want to delete this ingestion job?')) {
                                onDelete(job.id);
                              }
                            }}
                            className="flex items-center justify-center p-1.5 rounded-lg text-text-muted hover:text-flame hover:bg-flame/10 transition-colors"
                            title="Delete Job"
                          >
                            <Trash2 size={16} />
                          </button>
                        )}
                        {(job.status === 'pending' || job.status === 'running') && (
                          <button
                            onClick={() => onCancel(job.id)}
                            className="flex items-center justify-center p-1.5 rounded-lg text-text-muted hover:text-amber-500 hover:bg-amber-500/10 transition-colors"
                            title="Cancel Job"
                          >
                            <Ban size={16} />
                          </button>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-2">
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
              );
            })}
          </div>

          <PaginationControls
            currentPage={jobsPage}
            totalPages={totalJobsPages}
            onPageChange={onJobsPageChange}
          />
        </div>
      )}
    </div>
  );
}
