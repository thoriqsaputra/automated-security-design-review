import { Play, XCircle } from 'lucide-react';
import ReviewPipeline from '../../../components/flow/ReviewPipeline';
import LoadingSpinner from '../../../components/ui/LoadingSpinner';
import { useReviewDetail, type UseReviewDetailResult } from '../hooks/useReviewDetail';
import { ANALYSIS_MODE_OPTIONS, REVIEW_TABS } from '../utils/reviewPresentation';
import ReviewDebatePanel from './ReviewDebatePanel';
import ReviewFindingsPanel from './ReviewFindingsPanel';
import ReviewOverviewPanel from './ReviewOverviewPanel';

interface BaseProps {
  showControls?: boolean;
}

type Props =
  | (BaseProps & { reviewId: string; reviewState?: never })
  | (BaseProps & { reviewId?: never; reviewState: UseReviewDetailResult });

export default function ReviewWorkspace({ reviewId, reviewState, showControls = true }: Props) {
  const state = reviewState ?? useReviewDetail(reviewId);
  const {
    review,
    findings,
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
    handleTrigger,
    handleCancel,
  } = state;

  if (loading || !review) {
    return <LoadingSpinner />;
  }

  return (
    <div className="space-y-6">
      {showControls && (
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
            {['pending', 'cancelled', 'failed'].includes(review.status) && (
              <div className="min-w-[220px]">
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-text-muted">
                  Analysis Mode
                </label>
                <select
                  value={selectedAnalysisMode}
                  onChange={(event) => setSelectedAnalysisMode(event.target.value as typeof selectedAnalysisMode)}
                  className="w-full rounded-xl border border-surface-border bg-surface-base px-3 py-2.5 text-sm text-text-primary transition-colors focus:border-crimson focus:outline-none"
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
                  className="flex items-center gap-2 rounded-xl border border-crimson/50 bg-surface-base px-4 py-2.5 text-sm font-semibold text-crimson transition-all hover:bg-crimson/10 disabled:opacity-40"
                >
                  <XCircle size={16} /> {cancelling ? 'Cancelling...' : 'Cancel Review'}
                </button>
              )}
              {['pending', 'cancelled', 'failed'].includes(review.status) && (
                <button
                  onClick={() => void handleTrigger()}
                  disabled={triggering}
                  className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-crimson to-flame px-4 py-2.5 text-sm font-semibold text-white transition-all hover:shadow-lg hover:shadow-crimson/30 disabled:opacity-40"
                >
                  <Play size={16} />{' '}
                  {triggering
                    ? 'Starting...'
                    : review.status === 'pending'
                      ? 'Trigger Review'
                      : 'Re-trigger Review'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {!['completed_clean', 'completed_with_findings', 'approved', 'rejected'].includes(review.status) && (
        <ReviewPipeline reviewStatus={review.status} currentStage={currentStage} />
      )}

      <div className="no-scrollbar mb-6 flex gap-6 overflow-x-auto border-b border-surface-border px-2">
        {REVIEW_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`border-b-2 pb-3 text-sm font-semibold whitespace-nowrap transition-colors ${
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

      {activeTab === 'debate' && (
        <ReviewDebatePanel
          reviewId={review.id}
          reviewStatus={review.status}
          analysisMode={selectedAnalysisMode}
          debatedTotal={debatedTotal}
          debatedProcessed={debatedProcessed}
          debatedRemaining={debatedRemaining}
          persistenceTotal={persistenceTotal}
          persistenceProcessed={persistenceProcessed}
          persistenceRemaining={persistenceRemaining}
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
