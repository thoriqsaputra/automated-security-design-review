import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  activateIngestionJob,
  cancelIngestionJob,
  createIngestionJob,
  deleteIngestionJob,
  deleteParameterChild,
  deleteParameterParent,
  getCategoryParameters,
  getIngestionJobAsvsLevelDefinitions,
  listIngestionJobs,
  type ASVSLevelDefinition,
  type DiagramRequirement,
  type IngestionJob,
  type ParameterParent,
  type StandardCategory,
} from '../../../api/standards';
import {
  countByAsvsLevel,
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
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [filterAsvsLevel, setFilterAsvsLevel] = useState('all');
  const [feedback, setFeedback] = useState<string | null>(null);

  const jobsPerPage = 3;

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
  };

  useEffect(() => {
    if (!code) return;
    
    let isMounted = true;
    fetchCategoryData(code)
      .then(data => {
        if (!isMounted) return;
        setCategory(data.category);
        setParameters(data.parameters);
        setDiagramRequirements(data.diagram_requirements);
        setJobs(data.jobs);
        setFeedback(null);
        setLoading(false);
      })
      .catch(error => {
        if (!isMounted) return;
        setFeedback(error instanceof Error ? error.message : 'Failed to load category details.');
        setLoading(false);
      });

    return () => { isMounted = false; };
  }, [code]);

  const [prevResetDeps, setPrevResetDeps] = useState([activeTab, filterAsvsLevel, pageSize, search]);
  if (
    prevResetDeps[0] !== activeTab ||
    prevResetDeps[1] !== filterAsvsLevel ||
    prevResetDeps[2] !== pageSize ||
    prevResetDeps[3] !== search
  ) {
    setPrevResetDeps([activeTab, filterAsvsLevel, pageSize, search]);
    setParamsPage(1);
    setDiagramParamsPage(1);
  }

  useEffect(() => {
    const hasRunningJob = jobs.some((job) => job.status === 'running' || job.status === 'pending');
    if (!hasRunningJob) {
      return;
    }

    const interval = setInterval(() => {
      void pollJobs();
    }, 3000);

    return () => clearInterval(interval);
  }, [jobs, code]);

  const commitSearch = () => {
    setParamsPage(1);
    setDiagramParamsPage(1);
    setSearch(searchInput);
  };

  const clearFilters = () => {
    setSearch('');
    setSearchInput('');
    setFilterAsvsLevel('all');
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
      
      await createIngestionJob(code, file, startPage, endPage, levelDefinitionStartPage, levelDefinitionEndPage);
      setShowUpload(false);
      setFile(null);
      setStartPage('');
      setEndPage('');
      setLevelDefinitionStartPage('');
      setLevelDefinitionEndPage('');
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

  const openDefinitions = async (job: IngestionJob) => {
    setDefinitionsJob(job);
    setDefinitions([]);
    setDefinitionsLoading(true);
    try {
      const response = await getIngestionJobAsvsLevelDefinitions(job.id);
      setDefinitions(response.data);
    } catch (error) {
      setFeedback(error instanceof Error ? error.message : 'Failed to load ASVS level definitions.');
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
    await deleteParameterParent(parentId);
    await loadInitial();
  };

  const handleDeleteChild = async (childId: number) => {
    await deleteParameterChild(childId);
    await loadInitial();
  };

  const filteredParameters = useMemo(() => {
    return parameters
      .map((parent) => {
        const parentMatchesSearch = search
          ? parent.title.toLowerCase().includes(search.toLowerCase())
            || parent.stable_key.toLowerCase().includes(search.toLowerCase())
          : true;

        const filteredChildren = parent.children.filter((child) => {
          const levelStr = child.asvs_level ? child.asvs_level.toString() : 'unknown';
          const matchLevel = filterAsvsLevel === 'all' || levelStr === filterAsvsLevel;
          const matchSearch = search
            ? child.requirement_text.toLowerCase().includes(search.toLowerCase())
              || (child.details && child.details.toLowerCase().includes(search.toLowerCase()))
            : true;

          return matchLevel && (parentMatchesSearch || matchSearch);
        });

        return {
          ...parent,
          children: filteredChildren,
          _matchesParent: parentMatchesSearch,
        };
      })
      .filter((parent) => parent.children.length > 0 || (parent._matchesParent && filterAsvsLevel === 'all')) as ParameterParentWithMatch[];
  }, [filterAsvsLevel, parameters, search]);

  const filteredDiagramRequirements = useMemo(() => {
    return diagramRequirements.filter((requirement) => {
      const levelStr = requirement.asvs_level ? requirement.asvs_level.toString() : 'unknown';
      const matchLevel = filterAsvsLevel === 'all' || levelStr === filterAsvsLevel;
      const matchSearch = search
        ? requirement.requirement_text.toLowerCase().includes(search.toLowerCase())
          || requirement.verification_hint.toLowerCase().includes(search.toLowerCase())
          || requirement.parent_section.toLowerCase().includes(search.toLowerCase())
          || requirement.stable_key.toLowerCase().includes(search.toLowerCase())
        : true;
      return matchLevel && matchSearch;
    });
  }, [diagramRequirements, filterAsvsLevel, search]);

  const parameterCounts = useMemo(
    () => countByAsvsLevel(flattenParameterChildren(filteredParameters)),
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
    levelDefinitionStartPage,
    setLevelDefinitionStartPage,
    levelDefinitionEndPage,
    setLevelDefinitionEndPage,
    uploading,
    definitionsJob,
    definitions,
    definitionsLoading,
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
    commitSearch,
    clearFilters,
    filterAsvsLevel,
    setFilterAsvsLevel,
    feedback,
    handleActivate,
    handleUpload,
    handleDeleteJob,
    handleCancelJob,
    openDefinitions,
    closeDefinitions,
    handleDeleteParent,
    handleDeleteChild,
    filteredParameters,
    filteredDiagramRequirements,
    parameterCounts,
    diagramCounts,
    paginatedJobs,
    totalJobsPages,
    paginatedParams,
    totalParamsPages,
    paginatedDiagramParams,
    totalDiagramParamsPages,
  };
}
