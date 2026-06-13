import { useEffect, useState, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, ChevronDown, ChevronRight, CheckCircle2, Zap, Trash2, Ban, Upload, ShieldCheck, Search, Filter, X } from 'lucide-react';
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
  getIngestionJobAsvsLevelDefinitions,
  deleteParameterParent,
  deleteParameterChild,
  type ASVSLevelDefinition,
  type ParameterParent,
  type StandardCategory,
  type IngestionJob,
  type DiagramRequirement,
} from '../api/standards';

export default function CategoryDetail() {
  const { code } = useParams<{ code: string }>();
  const navigate = useNavigate();
  const [category, setCategory] = useState<StandardCategory | null>(null);
  const [parameters, setParameters] = useState<ParameterParent[]>([]);
  const [diagramRequirements, setDiagramRequirements] = useState<DiagramRequirement[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedParent, setExpandedParent] = useState<number | null>(null);

  const [showUpload, setShowUpload] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [startPage, setStartPage] = useState('');
  const [endPage, setEndPage] = useState('');
  const [levelDefinitionStartPage, setLevelDefinitionStartPage] = useState('');
  const [levelDefinitionEndPage, setLevelDefinitionEndPage] = useState('');
  const [uploading, setUploading] = useState(false);
  const [definitionsJob, setDefinitionsJob] = useState<IngestionJob | null>(null);
  const [definitions, setDefinitions] = useState<ASVSLevelDefinition[]>([]);
  const [definitionsLoading, setDefinitionsLoading] = useState(false);

  const [activeTab, setActiveTab] = useState<'requirements' | 'diagram'>('requirements');
  const [pageSize, setPageSize] = useState(10);
  const [jobsPage, setJobsPage] = useState(1);
  const [paramsPage, setParamsPage] = useState(1);
  const [diagramParamsPage, setDiagramParamsPage] = useState(1);
  const JOBS_PER_PAGE = 3;

  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [filterAsvsLevel, setFilterAsvsLevel] = useState<string>('all');

  useEffect(() => {
    setParamsPage(1);
    setDiagramParamsPage(1);
  }, [search, filterAsvsLevel, pageSize, activeTab]);

  const loadInitial = () => {
    if (!code) return;
    setJobsPage(1);
    setParamsPage(1);
    setLoading(true);
    return Promise.all([
      getCategoryParameters(code).then(r => {
        setCategory(r.data.category);
        setParameters(r.data.parameters);
        setDiagramRequirements(r.data.diagram_requirements || []);
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
        setDiagramRequirements(r.data.diagram_requirements || []);
      }),
      listIngestionJobs(code).then(r => setJobs(r.data))
    ]);
  };

  const handleUpload = async () => {
    if (!code || !file) return;
    setUploading(true);
    try {
      await createIngestionJob(code, file, startPage, endPage, levelDefinitionStartPage, levelDefinitionEndPage);
      setShowUpload(false);
      setFile(null);
      setStartPage('');
      setEndPage('');
      setLevelDefinitionStartPage('');
      setLevelDefinitionEndPage('');
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

  const openDefinitions = async (job: IngestionJob) => {
    setDefinitionsJob(job);
    setDefinitions([]);
    setDefinitionsLoading(true);
    try {
      const response = await getIngestionJobAsvsLevelDefinitions(job.id);
      setDefinitions(response.data);
    } catch (error: any) {
      alert(`Failed to load ASVS level definitions: ${error.response?.data?.detail || error.message}`);
      setDefinitionsJob(null);
    } finally {
      setDefinitionsLoading(false);
    }
  };

  const closeDefinitions = () => {
    setDefinitionsJob(null);
    setDefinitions([]);
    setDefinitionsLoading(false);
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

  const filteredParameters = useMemo(() => {
    return parameters.map(parent => {
      const parentMatchesSearch = search ? (
        parent.title.toLowerCase().includes(search.toLowerCase()) ||
        parent.stable_key.toLowerCase().includes(search.toLowerCase())
      ) : true;

      const filteredChildren = parent.children.filter(child => {
        const levelStr = child.asvs_level ? child.asvs_level.toString() : 'unknown';
        const matchLevel = filterAsvsLevel === 'all' || levelStr === filterAsvsLevel;
        const matchSearch = search ? (
          child.requirement_text.toLowerCase().includes(search.toLowerCase()) ||
          (child.details && child.details.toLowerCase().includes(search.toLowerCase()))
        ) : true;
        
        return matchLevel && (parentMatchesSearch || matchSearch);
      });

      return {
        ...parent,
        children: filteredChildren,
        _matchesParent: parentMatchesSearch
      };
    }).filter(parent => {
      return parent.children.length > 0 || (parent._matchesParent && filterAsvsLevel === 'all');
    });
  }, [parameters, search, filterAsvsLevel]);

  const filteredDiagramRequirements = useMemo(() => {
    return diagramRequirements.filter(req => {
      const levelStr = req.asvs_level ? req.asvs_level.toString() : 'unknown';
      const matchLevel = filterAsvsLevel === 'all' || levelStr === filterAsvsLevel;
      const matchSearch = search ? (
        req.requirement_text.toLowerCase().includes(search.toLowerCase()) ||
        req.verification_hint.toLowerCase().includes(search.toLowerCase()) ||
        req.parent_section.toLowerCase().includes(search.toLowerCase()) ||
        req.stable_key.toLowerCase().includes(search.toLowerCase())
      ) : true;
      return matchLevel && matchSearch;
    });
  }, [diagramRequirements, search, filterAsvsLevel]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-flame border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  const paginatedJobs = jobs.slice((jobsPage - 1) * JOBS_PER_PAGE, jobsPage * JOBS_PER_PAGE);
  const totalJobsPages = Math.ceil(jobs.length / JOBS_PER_PAGE) || 1;

  const paginatedParams = filteredParameters.slice((paramsPage - 1) * pageSize, paramsPage * pageSize);
  const totalParamsPages = Math.ceil(filteredParameters.length / pageSize) || 1;

  const paginatedDiagramParams = filteredDiagramRequirements.slice((diagramParamsPage - 1) * pageSize, diagramParamsPage * pageSize);
  const totalDiagramParamsPages = Math.ceil(filteredDiagramRequirements.length / pageSize) || 1;

  const asvsLevelClass = (level: number | null) => {
    if (level === 1) return 'bg-emerald-500/15 text-emerald-400';
    if (level === 2) return 'bg-sky-500/15 text-sky-400';
    if (level === 3) return 'bg-fuchsia-500/15 text-fuchsia-400';
    return 'bg-surface-hover text-text-muted';
  };
  const asvsLevelLabel = (level: number | null) => (level ? `L${level}` : 'Unknown');
  const parentLevelSummary = (parent: ParameterParent) => {
    const counts = parent.children.reduce<Record<string, number>>((acc, child) => {
      const key = asvsLevelLabel(child.asvs_level);
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    return ['L1', 'L2', 'L3', 'Unknown']
      .filter(key => counts[key])
      .map(key => `${key}: ${counts[key]}`)
      .join(' · ');
  };

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
                  {(() => {
                    const asvsSummary = (job.summary_json as any)?.asvs_level_definitions;
                    const asvsStatus = asvsSummary?.status || 'pending';
                    const asvsCount = Number(asvsSummary?.count || 0);
                    const isExtracted = asvsStatus === 'extracted' && asvsCount > 0;
                    const statusClass = isExtracted
                      ? 'bg-emerald-500/15 text-emerald-400'
                      : asvsStatus === 'fallback'
                        ? 'bg-amber-500/15 text-amber-400'
                        : 'bg-surface-hover text-text-muted';
                    return (
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
                        <button
                          type="button"
                          onClick={() => openDefinitions(job)}
                          className={`mt-2 inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-[11px] font-medium transition-colors hover:bg-surface-hover ${statusClass}`}
                        >
                          <ShieldCheck size={12} />
                          ASVS levels: {isExtracted ? `extracted · ${asvsCount}` : asvsStatus === 'fallback' ? 'fallback' : asvsStatus}
                        </button>
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
                    );
                  })()}
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
      {/* Tabs Navigation */}
      <div className="border-b border-surface-border mb-6 flex gap-6 px-2 overflow-x-auto no-scrollbar">
        {[
          { id: 'requirements', label: 'Requirement Text' },
          { id: 'diagram', label: 'Diagram Requirement' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`pb-3 text-sm font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === tab.id
                ? 'border-flame text-flame'
                : 'border-transparent text-text-muted hover:text-text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Filter Bar */}
      {(parameters.length > 0 || diagramRequirements.length > 0) && (
        <div className="flex flex-col sm:flex-row flex-wrap gap-3 bg-surface-base/50 p-3 rounded-xl border border-surface-border items-center mb-6">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={16} />
            <input 
              type="text" 
              placeholder={activeTab === 'requirements' ? "Search requirement text..." : "Search diagram requirements..."}
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && setSearch(searchInput)}
              onBlur={() => setSearch(searchInput)}
              className="w-full bg-midnight border border-surface-border text-sm rounded-lg pl-9 pr-3 py-2 text-text-primary focus:outline-none focus:border-flame transition-colors"
            />
            {search && (
              <button onClick={() => { setSearch(''); setSearchInput(''); }} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary">
                <X size={14} />
              </button>
            )}
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <div className="flex items-center gap-1.5 bg-midnight border border-surface-border rounded-lg px-1 py-1">
              <Filter size={14} className="text-text-muted ml-2" />
              <select
                value={filterAsvsLevel}
                onChange={(e) => setFilterAsvsLevel(e.target.value)}
                className="bg-transparent text-sm text-text-primary focus:outline-none py-1 pr-6 cursor-pointer appearance-none"
                style={{ backgroundImage: 'url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'%3e%3cpolyline points=\'6 9 12 15 18 9\'%3e%3c/polyline%3e%3c/svg%3e")', backgroundRepeat: 'no-repeat', backgroundPosition: 'right 0.25rem center', backgroundSize: '1em' }}
              >
                <option value="all">ASVS Level: All</option>
                <option value="1">ASVS Level: L1</option>
                <option value="2">ASVS Level: L2</option>
                <option value="3">ASVS Level: L3</option>
                <option value="unknown">ASVS Level: Unknown</option>
              </select>
            </div>

            {(search || filterAsvsLevel !== 'all') && (
              <button 
                onClick={() => {
                  setSearch('');
                  setSearchInput('');
                  setFilterAsvsLevel('all');
                }}
                className="text-xs font-semibold text-text-muted hover:text-text-primary transition-colors ml-2"
              >
                Clear All
              </button>
            )}
          </div>
        </div>
      )}

      {/* Tab Content */}
      {activeTab === 'requirements' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-text-primary">
              Extracted Parameters ({filteredParameters.length} sections)
            </h2>
          </div>
          {parameters.length === 0 ? (
            <Card><p className="text-sm text-text-muted text-center py-4">No parameters extracted yet. Complete an ingestion job first.</p></Card>
          ) : filteredParameters.length === 0 ? (
            <Card><p className="text-sm text-text-muted text-center py-4">No parameters match the current filters.</p></Card>
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
                            <p className="text-xs text-text-muted">
                              {parent.children.length} requirement(s)
                              {parentLevelSummary(parent) && ` · ${parentLevelSummary(parent)}`}
                            </p>
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
                                <div className="flex flex-wrap items-center gap-2">
                                  <p className="text-sm text-text-primary">{child.requirement_text}</p>
                                  <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold ${asvsLevelClass(child.asvs_level)}`}>
                                    {asvsLevelLabel(child.asvs_level)}
                                  </span>
                                </div>
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

              {/* Pagination Controls */}
              {totalParamsPages > 0 && (
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-surface-base/50 border border-surface-border rounded-xl p-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-text-muted">Items per page:</span>
                    <select 
                      value={pageSize}
                      onChange={(e) => setPageSize(Number(e.target.value))}
                      className="bg-midnight border border-surface-border text-sm rounded-lg px-2 py-1 text-text-primary focus:outline-none"
                    >
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                      <option value={20}>20</option>
                      <option value={50}>50</option>
                      <option value={100}>100</option>
                    </select>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setParamsPage(p => Math.max(1, p - 1))}
                      disabled={paramsPage === 1}
                      className="px-3 py-1.5 text-sm font-medium rounded-lg border border-surface-border bg-midnight hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-text-primary"
                    >
                      Previous
                    </button>

                    <div className="flex items-center gap-1 mx-2">
                      {Array.from({ length: Math.min(5, totalParamsPages) }).map((_, i) => {
                        let pageNum = i + 1;
                        if (totalParamsPages > 5) {
                          if (paramsPage > 3) {
                            pageNum = paramsPage - 2 + i;
                          }
                          if (paramsPage > totalParamsPages - 2) {
                            pageNum = totalParamsPages - 4 + i;
                          }
                        }
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setParamsPage(pageNum)}
                            className={`w-8 h-8 flex items-center justify-center text-sm font-medium rounded-lg border transition-colors ${
                              paramsPage === pageNum 
                                ? 'bg-flame/20 border-flame text-flame' 
                                : 'border-surface-border bg-midnight hover:bg-surface-hover text-text-primary'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>

                    <button
                      onClick={() => setParamsPage(p => Math.min(totalParamsPages, p + 1))}
                      disabled={paramsPage === totalParamsPages}
                      className="px-3 py-1.5 text-sm font-medium rounded-lg border border-surface-border bg-midnight hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-text-primary"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'diagram' && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-text-primary">
              Extracted Diagram Requirements ({filteredDiagramRequirements.length})
            </h2>
          </div>
          {diagramRequirements.length === 0 ? (
            <Card><p className="text-sm text-text-muted text-center py-4">No diagram requirements extracted yet. Complete an ingestion job first.</p></Card>
          ) : filteredDiagramRequirements.length === 0 ? (
            <Card><p className="text-sm text-text-muted text-center py-4">No diagram requirements match the current filters.</p></Card>
          ) : (
            <div className="space-y-3">
              {paginatedDiagramParams.map(req => (
                <Card key={req.id}>
                  <div className="flex items-start gap-4">
                    <div className="mt-1">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold ${asvsLevelClass(req.asvs_level)}`}>
                        {asvsLevelLabel(req.asvs_level)}
                      </span>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">{req.requirement_text}</p>
                      <p className="text-xs text-text-muted mt-1 italic">{req.verification_hint}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-[10px] font-mono text-text-muted bg-surface px-1.5 py-0.5 rounded">{req.stable_key}</span>
                        <span className="text-[10px] font-medium text-text-secondary">Parent: {req.parent_section}</span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}

              {/* Pagination Controls */}
              {totalDiagramParamsPages > 0 && (
                <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-surface-base/50 border border-surface-border rounded-xl p-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-text-muted">Items per page:</span>
                    <select 
                      value={pageSize}
                      onChange={(e) => setPageSize(Number(e.target.value))}
                      className="bg-midnight border border-surface-border text-sm rounded-lg px-2 py-1 text-text-primary focus:outline-none"
                    >
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                      <option value={20}>20</option>
                      <option value={50}>50</option>
                      <option value={100}>100</option>
                    </select>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={() => setDiagramParamsPage(p => Math.max(1, p - 1))}
                      disabled={diagramParamsPage === 1}
                      className="px-3 py-1.5 text-sm font-medium rounded-lg border border-surface-border bg-midnight hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-text-primary"
                    >
                      Previous
                    </button>

                    <div className="flex items-center gap-1 mx-2">
                      {Array.from({ length: Math.min(5, totalDiagramParamsPages) }).map((_, i) => {
                        let pageNum = i + 1;
                        if (totalDiagramParamsPages > 5) {
                          if (diagramParamsPage > 3) {
                            pageNum = diagramParamsPage - 2 + i;
                          }
                          if (diagramParamsPage > totalDiagramParamsPages - 2) {
                            pageNum = totalDiagramParamsPages - 4 + i;
                          }
                        }
                        return (
                          <button
                            key={pageNum}
                            onClick={() => setDiagramParamsPage(pageNum)}
                            className={`w-8 h-8 flex items-center justify-center text-sm font-medium rounded-lg border transition-colors ${
                              diagramParamsPage === pageNum 
                                ? 'bg-flame/20 border-flame text-flame' 
                                : 'border-surface-border bg-midnight hover:bg-surface-hover text-text-primary'
                            }`}
                          >
                            {pageNum}
                          </button>
                        );
                      })}
                    </div>

                    <button
                      onClick={() => setDiagramParamsPage(p => Math.min(totalDiagramParamsPages, p + 1))}
                      disabled={diagramParamsPage === totalDiagramParamsPages}
                      className="px-3 py-1.5 text-sm font-medium rounded-lg border border-surface-border bg-midnight hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-text-primary"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

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
              <label className="block text-xs font-medium text-text-secondary mb-1">Parameter Start Page (Optional)</label>
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
              <label className="block text-xs font-medium text-text-secondary mb-1">Parameter End Page (Optional)</label>
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
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">Level Definition Start Page (Optional)</label>
              <input
                type="number"
                min="1"
                placeholder="e.g. 8"
                value={levelDefinitionStartPage}
                onChange={e => setLevelDefinitionStartPage(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">Level Definition End Page (Optional)</label>
              <input
                type="number"
                min="1"
                placeholder="e.g. 10"
                value={levelDefinitionEndPage}
                onChange={e => setLevelDefinitionEndPage(e.target.value)}
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

      <Modal open={!!definitionsJob} onClose={closeDefinitions} title={`ASVS Level Definitions${definitionsJob ? ` · Version ${definitionsJob.version_no}` : ''}`}>
        {definitionsLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-7 h-7 border-2 border-flame border-t-transparent rounded-full animate-spin" />
          </div>
        ) : definitions.length === 0 ? (
          <div className="space-y-3">
            <p className="text-sm text-text-secondary">No version-specific ASVS level definitions were extracted for this ingestion job.</p>
            {definitionsJob && (definitionsJob.summary_json as any)?.asvs_level_definitions?.reason && (
              <p className="text-xs text-text-muted">{String((definitionsJob.summary_json as any).asvs_level_definitions.reason)}</p>
            )}
          </div>
        ) : (
          <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
            {definitions.map(definition => (
              <div key={definition.id} className="rounded-lg border border-surface-border bg-surface/60 p-3">
                <div className="flex items-center justify-between gap-3">
                  <h3 className="text-sm font-semibold text-text-primary">{definition.code} · {definition.name}</h3>
                  {definition.context_marker && <span className="text-[11px] text-text-muted">{definition.context_marker}</span>}
                </div>
                <p className="mt-2 text-xs text-text-secondary">{definition.classification_guidance}</p>
                {definition.source_quote && (
                  <p className="mt-2 border-l-2 border-surface-border pl-3 text-xs text-text-muted">{definition.source_quote}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </Modal>
    </div>
  );
}
