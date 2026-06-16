import { ArrowLeft, Play, XCircle } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';
import ReviewPipeline from '../components/flow/ReviewPipeline';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import ReviewDebatePanel from '../features/reviews/components/ReviewDebatePanel';
import ReviewFindingsPanel from '../features/reviews/components/ReviewFindingsPanel';
import ReviewOverviewPanel from '../features/reviews/components/ReviewOverviewPanel';
import ReviewRetrievalPanel from '../features/reviews/components/ReviewRetrievalPanel';
import { useReviewDetail } from '../features/reviews/hooks/useReviewDetail';
import { ANALYSIS_MODE_OPTIONS, REVIEW_TABS } from '../features/reviews/utils/reviewPresentation';

export default function ReviewDetail() {
  const { designId, id } = useParams<{ designId: string; id: string }>();
  const navigate = useNavigate();
  const {
    review,
    findings,
    retrievalVisualization,
    loading,
    loadingFindings,
    triggering,
    cancelling,
    selectedAnalysisMode,
    setSelectedAnalysisMode,
    expandedFinding,
    setExpandedFinding,
    activeTab,
    setActiveTab,
    search,
    searchInput,
    setSearchInput,
    commitSearch,
    clearFilters,
    filterMetStatus,
    setFilterMetStatus,
    filterSeverity,
    setFilterSeverity,
    filterFindingType,
    setFilterFindingType,
    currentPage,
    setCurrentPage,
    pageSize,
    setPageSize,
    totalFindings,
    totalPages,
    currentStage,
    debatedTotal,
    debatedProcessed,
    debatedRemaining,
    persistenceTotal,
    persistenceProcessed,
    persistenceRemaining,
    skippedByParentApplicability,
    handleTrigger,
    handleCancel,
  } = useReviewDetail(id);

  if (loading || !review) {
    return <LoadingSpinner />;
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <button
        onClick={() => navigate(`/designs/${designId}`)}
        className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors"
      >
        <ArrowLeft size={16} /> Back to Design
      </button>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Review #{review.id}</h1>
          <p className="text-sm text-text-muted mt-1">
            {review.design_name || `Design #${review.design_id}`}
            {review.category && ` · ${String((review.category as Record<string, unknown>).name || '')}`}
          </p>
        </div>
        <div className="flex items-stretch gap-4 sm:items-end">
          {['pending', 'cancelled', 'failed'].includes(review.status) && (
            <div className="min-w-[220px]">
              <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-text-muted">
                Analysis Mode
              </label>
              <select
                value={selectedAnalysisMode}
                onChange={(event) => setSelectedAnalysisMode(event.target.value as typeof selectedAnalysisMode)}
                className="w-full rounded-xl border border-surface-border bg-surface-base px-3 py-2.5 text-sm text-text-primary transition-colors focus:outline-none focus:border-crimson"
              >
                {ANALYSIS_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          )}
          <div className="flex gap-2">
            {review.status === 'running' && (
              <button
                onClick={() => void handleCancel()}
                disabled={cancelling}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-surface-base border border-crimson/50 text-crimson text-sm font-semibold hover:bg-crimson/10 transition-all disabled:opacity-40"
              >
                <XCircle size={16} /> {cancelling ? 'Cancelling...' : 'Cancel Review'}
              </button>
            )}
            {['pending', 'cancelled', 'failed'].includes(review.status) && (
              <button
                onClick={() => void handleTrigger()}
                disabled={triggering}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold hover:shadow-lg hover:shadow-crimson/30 transition-all disabled:opacity-40"
              >
                <Play size={16} /> {triggering ? 'Starting...' : review.status === 'pending' ? 'Trigger Review' : 'Re-trigger Review'}
              </button>
            )}
          </div>
        </div>
      </div>

      <ReviewPipeline reviewStatus={review.status} currentStage={currentStage} />

      <div className="border-b border-surface-border mb-6 flex gap-6 px-2 overflow-x-auto no-scrollbar">
        {REVIEW_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
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

      {activeTab === 'overview' && <ReviewOverviewPanel review={review} findingCount={totalFindings} />}

      {activeTab === 'retrieval' && (
        <ReviewRetrievalPanel review={review} retrievalVisualization={retrievalVisualization} />
      )}

      {activeTab === 'debate' && (
        <ReviewDebatePanel
          debatedTotal={debatedTotal}
          debatedProcessed={debatedProcessed}
          debatedRemaining={debatedRemaining}
          persistenceTotal={persistenceTotal}
          persistenceProcessed={persistenceProcessed}
          persistenceRemaining={persistenceRemaining}
          skippedByParentApplicability={skippedByParentApplicability}
        />
      )}

      {activeTab === 'findings' && (
        <ReviewFindingsPanel
          review={review}
          findings={findings}
          totalFindings={totalFindings}
          totalPages={totalPages}
          loadingFindings={loadingFindings}
          expandedFinding={expandedFinding}
          onToggleFinding={(findingId) => setExpandedFinding(expandedFinding === findingId ? null : findingId)}
          search={search}
          searchInput={searchInput}
          onSearchInputChange={setSearchInput}
          onSearchCommit={commitSearch}
          filterMetStatus={filterMetStatus}
          filterSeverity={filterSeverity}
          filterFindingType={filterFindingType}
          onFilterMetStatusChange={(value) => {
            setCurrentPage(1);
            setFilterMetStatus(value);
          }}
          onFilterSeverityChange={(value) => {
            setCurrentPage(1);
            setFilterSeverity(value);
          }}
          onFilterFindingTypeChange={(value) => {
            setCurrentPage(1);
            setFilterFindingType(value);
          }}
          onClearFilters={clearFilters}
          currentPage={currentPage}
          onCurrentPageChange={setCurrentPage}
          pageSize={pageSize}
          onPageSizeChange={(nextPageSize) => {
            setCurrentPage(1);
            setPageSize(nextPageSize);
          }}
        />
      )}
    </div>
  );
}
