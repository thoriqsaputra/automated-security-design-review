import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  activateIngestionJob,
  cancelIngestionJob,
  createIngestionJob,
  deleteIngestionJob,
  deleteParameterChild,
  deleteParameterParent,
  toggleParameterChild,
  toggleParameterParent,
  getCategoryParameters,
  listIngestionJobs,
  type DiagramRequirement,
  type IngestionJob,
  type ParameterParent,
  type StandardCategory,
} from '../../../api/standards';
import {
  countDiagramRequirements,
  flattenParameterChildren,
} from '../utils/standardsPresentation';

type ParameterParentWithMatch = ParameterParent & { _matchesParent?: boolean };

export function useCategoryDetail(code?: string) {
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
  const [uploading, setUploading] = useState(false);
  const [activeTab, setActiveTab] = useState<'requirements' | 'diagram'>('requirements');
  const [pageSize, setPageSize] = useState(10);
  const [jobsPage, setJobsPage] = useState(1);
  const [paramsPage, setParamsPage] = useState(1);
  const [diagramParamsPage, setDiagramParamsPage] = useState(1);
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('design');
  const [feedback, setFeedback] = useState<string | null>(null);

  const jobsPerPage = 3;

  // Tracks which job's parameters are currently loaded into `parameters`, so
  // the polling loop can tell when the backend has auto-activated a DIFFERENT
  // job (on ingestion completion, with no explicit user action) and knows it
  // needs to refetch — otherwise the parameter tree silently goes stale.
  const activeJobIdRef = useRef<number | null>(null);

  const findActiveJobId = (jobList: IngestionJob[]): number | null =>
    jobList.find((job) => job.is_active)?.id ?? null;

  const [prevCode, setPrevCode] = useState(code);
  if (code !== prevCode) {
    setPrevCode(code);
    setLoading(true);
    setJobsPage(1);
    setParamsPage(1);
  }

  const fetchCategoryData = async (catCode: string) => {
    const [parameterResponse, jobsResponse] = await Promise.all([
      getCategoryParameters(catCode),
      listIngestionJobs(catCode),
    ]);
    return {
      category: parameterResponse.data.category,
      parameters: parameterResponse.data.parameters,
      diagram_requirements: parameterResponse.data.diagram_requirements || [],
      jobs: jobsResponse.data,
    };
  };

  const loadInitial = async () => {
    if (!code) {
      return;
    }
    setLoading(true);
    setJobsPage(1);
    setParamsPage(1);
    try {
      const data = await fetchCategoryData(code);
      setCategory(data.category);
      setParameters(data.parameters);
      setDiagramRequirements(data.diagram_requirements);
      setJobs(data.jobs);
      activeJobIdRef.current = findActiveJobId(data.jobs);
      setFeedback(null);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'Failed to load category details.');
    } finally {
      setLoading(false);
    }
  };

  const pollJobs = async () => {
    if (!code) {
      return;
    }
    const response = await listIngestionJobs(code);
    setJobs(response.data);

    // The backend auto-activates a job on successful completion with no
    // explicit "Activate" click — if that just happened, the parameter tree
    // (and diagram requirements) for the newly-active job haven't been
    // fetched yet. Refresh them the moment the poll notices the active job
    // changed, instead of waiting for a user action or a page reload.
    const newActiveJobId = findActiveJobId(response.data);
    if (newActiveJobId !== activeJobIdRef.current) {
      activeJobIdRef.current = newActiveJobId;
      try {
        const data = await fetchCategoryData(code);
        setCategory(data.category);
        setParameters(data.parameters);
        setDiagramRequirements(data.diagram_requirements);
      } catch (error) {
        setFeedback(error instanceof Error ? error.message : 'Failed to refresh updated parameters.');
      }
    }
  };

  useEffect(() => {
    if (!code) return;

    let isMounted = true;
    fetchCategoryData(code)
      .then((data) => {
        if (!isMounted) return;
        setCategory(data.category);
        setParameters(data.parameters);
        setDiagramRequirements(data.diagram_requirements);
        setJobs(data.jobs);
        activeJobIdRef.current = findActiveJobId(data.jobs);
        setFeedback(null);
        setLoading(false);
      })
      .catch((error) => {
        if (!isMounted) return;
        setFeedback(error instanceof Error ? error.message : 'Failed to load category details.');
        setLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [code]);

  const [prevResetDeps, setPrevResetDeps] = useState([activeTab, pageSize, search, categoryFilter]);
  if (
    prevResetDeps[0] !== activeTab ||
    prevResetDeps[1] !== pageSize ||
    prevResetDeps[2] !== search ||
    prevResetDeps[3] !== categoryFilter
  ) {
    setPrevResetDeps([activeTab, pageSize, search, categoryFilter]);
    setParamsPage(1);
    setDiagramParamsPage(1);
  }

  const hasRunningJob = useMemo(
    () => jobs.some((job) => job.status === 'running' || job.status === 'pending'),
    [jobs],
  );

  useEffect(() => {
    if (!hasRunningJob) {
      return;
    }

    const interval = setInterval(() => {
      void pollJobs();
    }, 3000);

    return () => clearInterval(interval);
    // hasRunningJob is a derived boolean primitive, not the `jobs` array
    // itself, so this interval is only torn down/recreated when the
    // running/pending set actually flips — not on every 3s poll tick.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasRunningJob, code]);

  const commitSearch = () => {
    setParamsPage(1);
    setDiagramParamsPage(1);
    setSearch(searchInput);
  };

  const clearFilters = () => {
    setSearch('');
    setSearchInput('');
  };

  const handleActivate = async (jobId: number) => {
    await activateIngestionJob(jobId);
    await loadInitial();
  };

  const handleUpload = async () => {
    if (!code || !file) {
      return;
    }
    setUploading(true);
    try {
      await createIngestionJob(code, file, startPage, endPage);
      setShowUpload(false);
      setFile(null);
      setStartPage('');
      setEndPage('');
      await loadInitial();
      navigate(`/standards/${code}`);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'Failed to start ingestion.');
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteJob = async (jobId: number) => {
    await deleteIngestionJob(jobId);
    await loadInitial();
  };

  const handleCancelJob = async (jobId: number) => {
    await cancelIngestionJob(jobId);
    await pollJobs();
  };

  const handleDeleteParent = async (parentId: number) => {
    await deleteParameterParent(parentId);
    await loadInitial();
  };

  const handleDeleteChild = async (childId: number) => {
    await deleteParameterChild(childId);
    await loadInitial();
  };

  const handleToggleParent = async (parentId: number) => {
    try {
      const response = await toggleParameterParent(parentId);
      setParameters((prev) =>
        prev.map((p) => {
          if (p.id === parentId) {
            return {
              ...p,
              is_active: response.data.is_active,
              children: p.children.map((c) => ({ ...c, is_active: response.data.is_active })),
            };
          }
          return p;
        }),
      );
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'Failed to toggle parent.');
    }
  };

  const handleToggleChild = async (childId: number) => {
    try {
      const response = await toggleParameterChild(childId);
      setParameters((prev) =>
        prev.map((p) => ({
          ...p,
          children: p.children.map((c) => (c.id === childId ? { ...c, is_active: response.data.is_active } : c)),
        })),
      );
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'Failed to toggle child.');
    }
  };

  const totalParameterCount = useMemo(
    () => parameters.reduce((sum, p) => sum + p.children.length, 0),
    [parameters],
  );

  const filteredParameters = useMemo(() => {
    return parameters
      .map((parent) => {
        const parentMatchesSearch = search
          ? parent.title.toLowerCase().includes(search.toLowerCase())
            || parent.stable_key.toLowerCase().includes(search.toLowerCase())
          : true;

        const filteredChildren = parent.children.filter((child) => {
          const matchCategory = categoryFilter === 'all' || child.requirement_category === categoryFilter;
          const matchSearch = search
            ? child.requirement_text.toLowerCase().includes(search.toLowerCase())
            : true;

          return matchCategory && (parentMatchesSearch || matchSearch);
        });

        return {
          ...parent,
          children: filteredChildren,
          _matchesParent: parentMatchesSearch,
        };
      })
      .filter((parent) => parent.children.length > 0) as ParameterParentWithMatch[];
  }, [parameters, search, categoryFilter]);

  const filteredDiagramRequirements = useMemo(() => {
    return diagramRequirements.filter((requirement) => {
      return search
        ? requirement.requirement_text.toLowerCase().includes(search.toLowerCase())
          || requirement.verification_hint.toLowerCase().includes(search.toLowerCase())
          || requirement.parent_section.toLowerCase().includes(search.toLowerCase())
          || requirement.stable_key.toLowerCase().includes(search.toLowerCase())
        : true;
    });
  }, [diagramRequirements, search]);

  const parameterCounts = useMemo(
    () => flattenParameterChildren(filteredParameters).length,
    [filteredParameters],
  );
  const diagramCounts = useMemo(
    () => countDiagramRequirements(filteredDiagramRequirements),
    [filteredDiagramRequirements],
  );

  const paginatedJobs = jobs.slice((jobsPage - 1) * jobsPerPage, jobsPage * jobsPerPage);
  const totalJobsPages = Math.ceil(jobs.length / jobsPerPage) || 1;
  const paginatedParams = filteredParameters.slice((paramsPage - 1) * pageSize, paramsPage * pageSize);
  const totalParamsPages = Math.ceil(filteredParameters.length / pageSize) || 1;
  const paginatedDiagramParams = filteredDiagramRequirements.slice((diagramParamsPage - 1) * pageSize, diagramParamsPage * pageSize);
  const totalDiagramParamsPages = Math.ceil(filteredDiagramRequirements.length / pageSize) || 1;

  return {
    category,
    parameters,
    diagramRequirements,
    jobs,
    loading,
    expandedParent,
    setExpandedParent,
    showUpload,
    setShowUpload,
    file,
    setFile,
    startPage,
    setStartPage,
    endPage,
    setEndPage,
    uploading,
    activeTab,
    setActiveTab,
    pageSize,
    setPageSize,
    jobsPage,
    setJobsPage,
    paramsPage,
    setParamsPage,
    diagramParamsPage,
    setDiagramParamsPage,
    search,
    searchInput,
    setSearchInput,
    categoryFilter,
    setCategoryFilter,
    commitSearch,
    clearFilters,
    feedback,
    handleActivate,
    handleUpload,
    handleDeleteJob,
    handleCancelJob,
    handleDeleteParent,
    handleDeleteChild,
    handleToggleParent,
    handleToggleChild,
    filteredParameters,
    filteredDiagramRequirements,
    parameterCounts,
    totalParameterCount,
    diagramCounts,
    paginatedJobs,
    totalJobsPages,
    paginatedParams,
    totalParamsPages,
    paginatedDiagramParams,
    totalDiagramParamsPages,
  };
}
