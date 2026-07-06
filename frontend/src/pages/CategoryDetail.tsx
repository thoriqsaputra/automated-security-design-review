import { ArrowLeft, Upload } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';
import Card from '../components/ui/Card';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import CategoryJobsSection from '../features/standards/components/CategoryJobsSection';
import CategoryParametersPanel from '../features/standards/components/CategoryParametersPanel';
import CategoryUploadModal from '../features/standards/components/CategoryUploadModal';
import { useCategoryDetail } from '../features/standards/hooks/useCategoryDetail';

export default function CategoryDetail() {
  const { code } = useParams<{ code: string }>();
  const navigate = useNavigate();
  const {
    category,
    parameters,
    diagramRequirements,
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
    jobs,
  } = useCategoryDetail(code);

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <button
        onClick={() => navigate('/standards')}
        className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors"
      >
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

      {feedback && (
        <Card className="border-amber-500/30">
          <p className="text-sm text-amber-300">{feedback}</p>
        </Card>
      )}

      <CategoryJobsSection
        jobs={jobs}
        paginatedJobs={paginatedJobs}
        totalJobsPages={totalJobsPages}
        jobsPage={jobsPage}
        onJobsPageChange={setJobsPage}
        onActivate={(jobId) => void handleActivate(jobId)}
        onDelete={(jobId) => void handleDeleteJob(jobId)}
        onCancel={(jobId) => void handleCancelJob(jobId)}
      />

      <CategoryParametersPanel
        activeTab={activeTab}
        onTabChange={setActiveTab}
        hasAnyContent={parameters.length > 0 || diagramRequirements.length > 0}
        search={search}
        searchInput={searchInput}
        onSearchInputChange={setSearchInput}
        onSearchCommit={commitSearch}
        onClearFilters={clearFilters}
        categoryFilter={categoryFilter}
        onCategoryFilterChange={setCategoryFilter}
        totalParameterCount={totalParameterCount}
        parameters={parameters}
        filteredParameters={filteredParameters}
        parameterCounts={parameterCounts}
        paginatedParams={paginatedParams}
        paramsPage={paramsPage}
        totalParamsPages={totalParamsPages}
        pageSize={pageSize}
        onParamsPageChange={setParamsPage}
        onPageSizeChange={setPageSize}
        expandedParent={expandedParent}
        onToggleParent={(parentId) => setExpandedParent(expandedParent === parentId ? null : parentId)}
        onToggleParentActive={(parentId) => void handleToggleParent(parentId)}
        onDeleteParent={(parentId) => void handleDeleteParent(parentId)}
        onToggleChildActive={(childId) => void handleToggleChild(childId)}
        onDeleteChild={(childId) => void handleDeleteChild(childId)}
        diagramRequirements={diagramRequirements}
        filteredDiagramRequirements={filteredDiagramRequirements}
        diagramCounts={diagramCounts}
        paginatedDiagramRequirements={paginatedDiagramParams}
        diagramParamsPage={diagramParamsPage}
        totalDiagramParamsPages={totalDiagramParamsPages}
        onDiagramParamsPageChange={setDiagramParamsPage}
      />

      <CategoryUploadModal
        open={showUpload}
        title={`Ingest Standard: ${category?.name || code}`}
        uploading={uploading}
        file={file}
        startPage={startPage}
        endPage={endPage}
        onClose={() => setShowUpload(false)}
        onFileChange={setFile}
        onStartPageChange={setStartPage}
        onEndPageChange={setEndPage}
        onSubmit={() => void handleUpload()}
      />
    </div>
  );
}
