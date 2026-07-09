import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Check, ChevronDown, FileText, Play, Search, Trash2 } from 'lucide-react';
import type { JsonRecord, RetrievalVisualization } from '../api/reviews';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Card from '../components/ui/Card';
import LlmUsageCard from '../components/ui/LlmUsageCard';
import StatusBadge from '../components/ui/StatusBadge';
import RaptorTreeView from '../components/flow/RaptorTreeView';
import {
  getDesign,
  retryDesignPreparation,
  cancelDesignPreparation,
  reingestDesign,
  type DesignDetail,
} from '../api/designs';
import { listCategories, type StandardCategory } from '../api/standards';
import {
  cancelReview,
  createReview,
  deleteReview,
  listReviews,
  triggerReview,
  type Review,
  type ReviewAnalysisMode,
} from '../api/reviews';
import CreateReviewModal from '../features/designs/components/CreateReviewModal';
import DesignReingestModal from '../features/designs/components/DesignReingestModal';
import ReviewWorkspace from '../features/reviews/components/ReviewWorkspace';
import { useReviewDetail } from '../features/reviews/hooks/useReviewDetail';
import { ANALYSIS_MODE_OPTIONS, formatAnalysisModeLabel } from '../features/reviews/utils/reviewPresentation';

const REVIEWS_FETCH_LIMIT = 200;

function getCategoryLabel(review: Review): string {
  const category = review.category as JsonRecord | null;
  const name = typeof category?.name === 'string' ? category.name : '';
  const code = typeof category?.code === 'string' ? category.code : '';
  if (name && code) {
    return `${name} (${code})`;
  }
  return name || code || 'Uncategorized';
}

function formatReviewDate(date: string): string {
  return new Date(date).toLocaleString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function formatDuration(totalSeconds: number): string {
  if (!totalSeconds || totalSeconds <= 0) {
    return '0s';
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  if (minutes <= 0) {
    return `${seconds}s`;
  }
  return `${minutes}m ${seconds}s`;
}

export default function DesignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [design, setDesign] = useState<DesignDetail | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [categories, setCategories] = useState<StandardCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [selectedCat, setSelectedCat] = useState<number | ''>('');
  const [analysisMode, setAnalysisMode] = useState<ReviewAnalysisMode>('default');
  const [creating, setCreating] = useState(false);
  const [retryingPreparation, setRetryingPreparation] = useState(false);
  const [cancellingPreparation, setCancellingPreparation] = useState(false);
  const [showReingestModal, setShowReingestModal] = useState(false);
  const [reingestFile, setReingestFile] = useState<File | null>(null);
  const [reingesting, setReingesting] = useState(false);
  const [reingestError, setReingestError] = useState<string | null>(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [reviewSearch, setReviewSearch] = useState('');
  const [deletingReviewId, setDeletingReviewId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<'prepared_retrieval' | 'review_details'>('review_details');

  const load = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    return Promise.all([
      getDesign(Number(id)).then(r => setDesign(r.data)),
      listReviews(Number(id), { limit: REVIEWS_FETCH_LIMIT }).then(r => setReviews(r.data)),
      listCategories().then(r => setCategories(r.data)),
    ]).finally(() => setLoading(false));
  }, [id]);

  const requestSeqRef = useRef(0);
  const refreshPreparationStatus = useCallback(async () => {
    if (!id) return;
    const seq = ++requestSeqRef.current;
    const response = await getDesign(Number(id));
    if (seq !== requestSeqRef.current) return;
    setDesign(response.data);
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (!design || !['queued', 'running'].includes(design.preparation_status)) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      void refreshPreparationStatus();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [design, refreshPreparationStatus]);

  const handleCreateReview = async () => {
    if (!selectedCat || !id || !design?.can_start_analysis) return;
    setCreating(true);
    try {
      const res = await createReview(Number(id), Number(selectedCat), analysisMode);
      await triggerReview(res.data.id, analysisMode);
      setShowReviewModal(false);
      setSelectedCat('');
      setAnalysisMode('default');
      await load();
      setSearchParams({ reviewId: String(res.data.id) });
      setPickerOpen(false);
      setReviewSearch('');
    } finally {
      setCreating(false);
    }
  };

  const handleRetryPreparation = async () => {
    if (!id) return;
    setRetryingPreparation(true);
    try {
      await retryDesignPreparation(Number(id));
      maxSeenPercentRef.current = 0;
      await load();
    } finally {
      setRetryingPreparation(false);
    }
  };

  const handleCancelPreparation = async () => {
    if (!id) return;
    setCancellingPreparation(true);
    try {
      await cancelDesignPreparation(Number(id));
      await load();
    } finally {
      setCancellingPreparation(false);
    }
  };

  const handleReingest = async () => {
    if (!id || !reingestFile) return;
    setReingesting(true);
    setReingestError(null);
    try {
      await reingestDesign(Number(id), reingestFile);
      maxSeenPercentRef.current = 0;
      setShowReingestModal(false);
      setReingestFile(null);
      await load();
    } catch (err) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Failed to reingest design document.';
      setReingestError(detail);
    } finally {
      setReingesting(false);
    }
  };


  const handleDeleteReview = async (e: React.MouseEvent, reviewId: number) => {
    e.stopPropagation();
    if (!confirm(`Delete Review #${reviewId}? This cannot be undone.`)) return;
    setDeletingReviewId(reviewId);
    try {
      await deleteReview(reviewId);
      if (selectedReview?.id === reviewId) {
        setSearchParams({}, { replace: true });
      }
      await load();
    } finally {
      setDeletingReviewId(null);
    }
  };

  const requestedReviewId = searchParams.get('reviewId');

  const selectedReview = useMemo(() => {
    if (reviews.length === 0) {
      return null;
    }
    if (!requestedReviewId) {
      return reviews[0];
    }
    return reviews.find((review) => String(review.id) === requestedReviewId) || reviews[0];
  }, [requestedReviewId, reviews]);

  useEffect(() => {
    if (reviews.length === 0) {
      if (requestedReviewId) {
        setSearchParams({}, { replace: true });
      }
      return;
    }
    if (!selectedReview) {
      return;
    }
    if (String(selectedReview.id) !== requestedReviewId) {
      setSearchParams({ reviewId: String(selectedReview.id) }, { replace: true });
    }
  }, [requestedReviewId, reviews.length, selectedReview, setSearchParams]);

  const filteredReviews = useMemo(() => {
    const query = reviewSearch.trim().toLowerCase();
    if (!query) {
      return reviews;
    }
    return reviews.filter((review) => {
      const haystack = [
        `review ${review.id}`,
        review.status,
        getCategoryLabel(review),
        formatAnalysisModeLabel(review.analysis_mode),
        new Date(review.created_at).toLocaleDateString(),
        new Date(review.created_at).toLocaleString(),
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [reviewSearch, reviews]);

  const selectReview = (reviewId: number) => {
    setSearchParams({ reviewId: String(reviewId) });
    setPickerOpen(false);
    setReviewSearch('');
  };

  const selectedReviewState = useReviewDetail(selectedReview ? String(selectedReview.id) : undefined);

  // `reviews` is only refetched on mount and after explicit user actions, so its
  // entries go stale once a review finishes while the user is viewing it —
  // `selectedReviewState.review` is the one object that's actively polled (every
  // 8s while running, see useReviewDetail), so mirror its status back into the
  // list/picker whenever it changes.
  useEffect(() => {
    const freshReview = selectedReviewState.review;
    if (!freshReview) {
      return;
    }
    setReviews((prev) => {
      const index = prev.findIndex((review) => review.id === freshReview.id);
      if (index === -1) {
        return prev;
      }
      const current = prev[index];
      if (current.status === freshReview.status && current.updated_at === freshReview.updated_at) {
        return prev;
      }
      const next = [...prev];
      next[index] = { ...current, ...freshReview };
      return next;
    });
  }, [selectedReviewState.review]);

  const runningReview = useMemo(
    () => reviews.find((review) => review.status === 'running') || null,
    [reviews]
  );
  const hasRunningReview = Boolean(runningReview);

  const [cancellingRunningReview, setCancellingRunningReview] = useState(false);
  const handleCancelRunningReview = async () => {
    if (!runningReview) return;
    setCancellingRunningReview(true);
    try {
      await cancelReview(runningReview.id);
      await load();
    } finally {
      setCancellingRunningReview(false);
    }
  };

  const hasControls = Boolean(
    selectedReview &&
      ['pending', 'cancelled', 'failed', 'running'].includes(selectedReviewState.review?.status || '')
  );
  // When the selected review is the running one, the left "Review Control"
  // cluster already renders its own Cancel Review button — don't duplicate it
  // in the toolbar too.
  const showToolbarCancelButton =
    hasRunningReview && !(hasControls && selectedReviewState.review?.status === 'running');
  // When the selected review is pending/cancelled/failed/running, the left
  // cluster already offers the right action (Trigger/Re-trigger/Cancel) for
  // it — showing "Start TSD Analysis" in the toolbar too is redundant and,
  // for a cancelled/failed review, misleading (it starts a brand new review
  // instead of retrying the existing one).
  const showToolbarStartButton = !showToolbarCancelButton && !hasControls;

  const preparationLabel = useMemo(() => {
    if (!design) return '';
    if (design.preparation_status === 'ready') {
      return design.prepared_at
        ? `Ready for analysis • prepared ${formatReviewDate(design.prepared_at)}`
        : 'Ready for analysis';
    }
    if (design.preparation_status === 'failed') {
      return design.preparation_error || 'Preparation failed.';
    }
    if (design.preparation_status === 'stale') {
      return 'Preparation is stale and must be rebuilt.';
    }
    return 'Preparing RAPTOR index for debate.';
  }, [design]);

  const preparationProgress = useMemo(() => {
    if (!design) {
      return null;
    }
    if (!isRecord(design.preparation_progress)) {
      return null;
    }
    return design.preparation_progress as Record<string, unknown>;
  }, [design]);

  const maxSeenPercentRef = useRef(0);
  useEffect(() => {
    maxSeenPercentRef.current = 0;
  }, [id]);

  const overallPreparationPercent = useMemo(() => {
    if (!preparationProgress) return maxSeenPercentRef.current;
    const raw = preparationProgress.percentage;
    const value = typeof raw === 'number' ? raw : Number(raw || 0);
    const clamped = Math.max(value, maxSeenPercentRef.current);
    maxSeenPercentRef.current = clamped;
    return clamped;
  }, [preparationProgress]);

  const preparationCurrentStep = useMemo(() => {
    if (!preparationProgress) return '';
    return typeof preparationProgress.current_step === 'string'
      ? preparationProgress.current_step
      : '';
  }, [preparationProgress]);

  const preparationStatusLabel = useMemo(() => {
    if (!preparationProgress) return '';
    return typeof preparationProgress.status_label === 'string'
      ? preparationProgress.status_label
      : '';
  }, [preparationProgress]);

  const preparationSnapshot = useMemo(() => {
    if (!design) return null;
    const snapshot = design.preparation_snapshot_json;
    if (!snapshot || !isRecord(snapshot)) {
      return null;
    }
    return snapshot as RetrievalVisualization;
  }, [design]);

  const preparationStats = useMemo(() => {
    if (!design || !isRecord(design.preparation_stats_json)) {
      return null;
    }
    return design.preparation_stats_json as Record<string, unknown>;
  }, [design]);

  const preparationLlmUsage = useMemo(() => {
    if (!preparationStats || !isRecord(preparationStats.llm_usage)) {
      return null;
    }
    return preparationStats.llm_usage as Record<string, unknown>;
  }, [preparationStats]);

  const preparationDurationLabel = useMemo(() => {
    const totalSeconds = Number(design?.preparation_duration_seconds) || 0;
    if (totalSeconds <= 0) {
      return null;
    }
    return formatDuration(totalSeconds);
  }, [design?.preparation_duration_seconds]);

  useEffect(() => {
    if (selectedReview) {
      setActiveTab('review_details');
      return;
    }
    if (design?.preparation_status === 'ready' && preparationSnapshot) {
      setActiveTab('prepared_retrieval');
    }
  }, [design?.preparation_status, preparationSnapshot, selectedReview]);

  if (loading || !design) {
    return <LoadingSpinner />;
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <button onClick={() => navigate('/designs')} className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors">
        <ArrowLeft size={16} /> Back to Designs
      </button>

      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-gradient-to-br from-crimson to-flame">
            <FileText size={24} className="text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-text-primary">{design.name}</h1>
            <p className="text-sm text-text-muted">{new Date(design.created_at).toLocaleDateString()} • {reviews.length} reviews</p>
            <p className="mt-2 text-sm text-text-muted">{preparationLabel}</p>
          </div>
        </div>
      </div>

      <Card className="overflow-visible flex flex-col gap-4">
        {['queued', 'running'].includes(design.preparation_status) && preparationProgress && (
          <div className="space-y-3">
            <div>
              <div className="mb-1 flex items-start justify-between text-xs">
                <div className="flex flex-col">
                  <span className="font-medium text-text-primary">
                    {preparationStatusLabel || 'Preparing TSD for retrieval and debate'}
                  </span>
                  {preparationCurrentStep && (
                    <span className="text-text-muted">{preparationCurrentStep}</span>
                  )}
                </div>
                <span className="text-text-muted">{overallPreparationPercent}%</span>
              </div>
              <div className="h-2 rounded-full bg-surface-border overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-crimson to-flame transition-all duration-500 ease-out"
                  style={{ width: `${Math.max(4, overallPreparationPercent)}%` }}
                />
              </div>
            </div>

          </div>
        )}

        <div className="flex flex-col gap-4 lg:flex-row lg:items-center">
          <div className={`flex w-full flex-col gap-4 lg:flex-1 ${hasControls ? 'lg:flex-row lg:items-center' : ''}`}>
          {hasControls && (
            <div className="lg:shrink-0">
              {selectedReview && ['pending', 'cancelled', 'failed'].includes(selectedReviewState.review?.status || '') ? (
                <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
                <div className="min-w-[220px]">
                  <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-text-muted">
                    Analysis Mode
                  </label>
                  <select
                    value={selectedReviewState.selectedAnalysisMode}
                    onChange={(event) =>
                      selectedReviewState.setSelectedAnalysisMode(event.target.value as ReviewAnalysisMode)
                    }
                    className="w-full rounded-xl border border-surface-border bg-surface-base px-3 py-2.5 text-sm text-text-primary transition-colors focus:border-crimson focus:outline-none"
                  >
                    {ANALYSIS_MODE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={() => void selectedReviewState.handleTrigger()}
                  disabled={selectedReviewState.triggering || !design.can_start_analysis}
                  className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-crimson to-flame px-4 py-2.5 text-sm font-semibold text-white transition-all hover:shadow-lg hover:shadow-crimson/30 disabled:opacity-40"
                >
                  <Play size={16} />
                  {selectedReviewState.triggering
                    ? 'Starting...'
                    : selectedReviewState.review?.status === 'pending'
                      ? 'Trigger Review'
                      : 'Re-trigger Review'}
                </button>
              </div>
            ) : selectedReviewState.review?.status === 'running' ? (
              <div className="flex flex-col gap-3">
                <label className="mb-1 block text-xs font-semibold uppercase tracking-wider text-text-muted">
                  Review Control
                </label>
                <button
                  onClick={() => void selectedReviewState.handleCancel()}
                  disabled={selectedReviewState.cancelling}
                  className="flex items-center gap-2 rounded-xl border border-crimson/50 bg-surface-base px-4 py-2.5 text-sm font-semibold text-crimson transition-all hover:bg-crimson/10 disabled:opacity-40"
                >
                  {selectedReviewState.cancelling ? 'Cancelling...' : 'Cancel Review'}
                </button>
              </div>
            ) : null}
            </div>
          )}
          <div className="relative w-full flex-1 min-w-[240px]">
            <button
              onClick={() => setPickerOpen((open) => !open)}
              className="flex w-full items-center justify-between rounded-xl border border-surface-border bg-surface-base px-4 py-3 text-left transition-colors hover:border-burgundy"
            >
              <div className="min-w-0">
                {selectedReview ? (
                  <>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-text-primary">Review #{selectedReview.id}</span>
                      <StatusBadge status={selectedReview.status} />
                    </div>
                    <p className="mt-1 truncate text-xs text-text-muted">
                      {getCategoryLabel(selectedReview)} · {formatAnalysisModeLabel(selectedReview.analysis_mode)} ·{' '}
                      {formatReviewDate(selectedReview.created_at)}
                    </p>
                  </>
                ) : (
                  <>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold text-text-primary">No reviews yet</span>
                    </div>
                  </>
                )}
              </div>
              <ChevronDown
                size={18}
                className={`ml-3 shrink-0 text-text-muted transition-transform ${pickerOpen ? 'rotate-180' : ''}`}
              />
            </button>
            {pickerOpen && (
              <div className="absolute right-0 z-20 mt-2 w-full rounded-2xl border border-surface-border bg-midnight shadow-2xl shadow-black/30">
                <div className="space-y-3 border-b border-surface-border p-3">
                  {showToolbarCancelButton ? (
                    <button
                      onClick={() => {
                        setPickerOpen(false);
                        void handleCancelRunningReview();
                      }}
                      disabled={cancellingRunningReview}
                      className="flex w-full items-center justify-center gap-2 rounded-xl border border-crimson/50 bg-surface-base px-4 py-2.5 text-sm font-semibold text-crimson transition-all hover:bg-crimson/10 disabled:opacity-40"
                    >
                      {cancellingRunningReview ? 'Cancelling...' : 'Cancel Review'}
                    </button>
                  ) : showToolbarStartButton ? (
                    <button
                      onClick={() => {
                        if (!design.can_start_analysis || creating) {
                          return;
                        }
                        setPickerOpen(false);
                        setShowReviewModal(true);
                      }}
                      disabled={!design.can_start_analysis || creating}
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-crimson to-flame px-4 py-2.5 text-sm font-semibold text-white transition-all hover:shadow-lg hover:shadow-crimson/30 disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      <Play size={16} /> Start TSD Analysis
                    </button>
                  ) : null}
                  {reviews.length > 0 && (
                    <div className="flex items-center gap-2 rounded-xl border border-surface-border bg-surface-base px-3 py-2.5">
                      <Search size={16} className="text-text-muted" />
                      <input
                        value={reviewSearch}
                        onChange={(event) => setReviewSearch(event.target.value)}
                        placeholder="Search review versions..."
                        className="w-full bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
                      />
                    </div>
                  )}
                </div>
                <div className="max-h-80 overflow-y-auto p-2">
                  {reviews.length === 0 ? (
                    <div className="px-3 py-6 text-center text-sm text-text-muted">
                      No reviews yet for this design.
                    </div>
                  ) : filteredReviews.length === 0 ? (
                    <div className="px-3 py-6 text-center text-sm text-text-muted">
                      No review versions match "{reviewSearch}".
                    </div>
                  ) : (
                    filteredReviews.map((review) => {
                      const isActive = selectedReview?.id === review.id;
                      return (
                        <button
                          key={review.id}
                          onClick={() => selectReview(review.id)}
                          className={`mb-1 flex w-full items-start justify-between rounded-xl px-3 py-3 text-left transition-colors ${isActive ? 'bg-burgundy/20' : 'hover:bg-surface-hover'
                            }`}
                        >
                          <div className="min-w-0 pr-4">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-semibold text-text-primary">Review #{review.id}</span>
                              <StatusBadge status={review.status} />
                            </div>
                            <p className="mt-1 text-xs text-text-muted">
                              {getCategoryLabel(review)} · {formatAnalysisModeLabel(review.analysis_mode)}
                            </p>
                            <p className="mt-1 text-xs text-text-muted">{formatReviewDate(review.created_at)}</p>
                          </div>
                          <div className="mt-0.5 flex shrink-0 items-center gap-2">
                            {isActive && <Check size={16} className="text-flame" />}
                            <span
                              role="button"
                              tabIndex={0}
                              onClick={(e) => void handleDeleteReview(e, review.id)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' || e.key === ' ') {
                                  e.preventDefault();
                                  void handleDeleteReview(e as unknown as React.MouseEvent, review.id);
                                }
                              }}
                              aria-disabled={deletingReviewId === review.id}
                              className={`rounded-lg p-1.5 text-text-muted transition-colors hover:bg-crimson/15 hover:text-crimson ${deletingReviewId === review.id ? 'pointer-events-none opacity-40' : ''}`}
                            >
                              <Trash2 size={14} />
                            </span>
                          </div>
                        </button>
                      );
                    })
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="flex shrink-0 items-center justify-end gap-3">
          {['queued', 'running'].includes(design.preparation_status) && (
            <button
              onClick={() => void handleCancelPreparation()}
              disabled={cancellingPreparation}
              className="rounded-xl border border-crimson/50 bg-surface-base px-4 py-2.5 text-sm font-semibold text-crimson transition-colors hover:bg-crimson/10 disabled:opacity-40"
            >
              {cancellingPreparation ? 'Cancelling...' : 'Cancel Preparation'}
            </button>
          )}
          {(design.preparation_status === 'failed' || design.preparation_status === 'stale' || design.preparation_status === 'cancelled') && (
            <button
              onClick={() => void handleRetryPreparation()}
              disabled={retryingPreparation}
              className="rounded-xl border border-surface-border px-4 py-2 text-sm font-semibold text-text-primary transition-colors hover:border-burgundy disabled:opacity-40"
            >
              {retryingPreparation ? 'Retrying...' : 'Retry Preparation'}
            </button>
          )}
          <button
            onClick={() => {
              setReingestError(null);
              setReingestFile(null);
              setShowReingestModal(true);
            }}
            disabled={hasRunningReview || ['queued', 'running'].includes(design.preparation_status)}
            title={hasRunningReview ? 'Cannot reingest while a review is pending or running.' : undefined}
            className="rounded-xl border border-surface-border px-4 py-2 text-sm font-semibold text-text-primary transition-colors hover:border-burgundy disabled:opacity-40"
          >
            Reingest
          </button>
          {showToolbarCancelButton ? (
            <button
              onClick={() => void handleCancelRunningReview()}
              disabled={cancellingRunningReview}
              className="flex items-center gap-2 rounded-xl border border-crimson/50 bg-surface-base px-4 py-2.5 text-sm font-semibold text-crimson transition-all hover:bg-crimson/10 disabled:opacity-40"
            >
              {cancellingRunningReview ? 'Cancelling...' : 'Cancel Review'}
            </button>
          ) : showToolbarStartButton ? (
            <button
              onClick={() => setShowReviewModal(true)}
              disabled={!design.can_start_analysis || creating}
              className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-crimson to-flame px-4 py-2.5 text-sm font-semibold text-white transition-all hover:shadow-lg hover:shadow-crimson/30 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play size={16} />
              Start TSD Analysis
            </button>
          ) : null}
        </div>
      </div>
    </Card>

      <div className="no-scrollbar mb-6 flex gap-6 overflow-x-auto border-b border-surface-border px-2">
        {design.preparation_status === 'ready' && preparationSnapshot && (
          <button
            onClick={() => setActiveTab('prepared_retrieval')}
            className={`border-b-2 pb-3 text-sm font-semibold whitespace-nowrap transition-colors ${
              activeTab === 'prepared_retrieval'
                ? 'border-flame text-flame'
                : 'border-transparent text-text-muted hover:text-text-primary'
            }`}
          >
            Prepared Retrieval
          </button>
        )}
        <button
          onClick={() => setActiveTab('review_details')}
          className={`border-b-2 pb-3 text-sm font-semibold whitespace-nowrap transition-colors ${
            activeTab === 'review_details'
              ? 'border-flame text-flame'
              : 'border-transparent text-text-muted hover:text-text-primary'
          }`}
        >
          Review Details
        </button>
      </div>

      {activeTab === 'prepared_retrieval' && design.preparation_status === 'ready' && preparationSnapshot && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-text-primary">Prepared Retrieval Structures</h2>
              <p className="mt-1 text-sm text-text-muted">
                Inspect the RAPTOR summary tree built from this uploaded TSD.
              </p>
            </div>
            {preparationSnapshot.generated_at && (
              <p className="text-xs text-text-muted">
                Generated {new Date(preparationSnapshot.generated_at).toLocaleString()}
              </p>
            )}
          </div>

          {preparationSnapshot.raptor?.status === 'ready' ? (
            <RaptorTreeView snapshot={preparationSnapshot.raptor} />
          ) : (
            <Card>
              <p className="py-4 text-center text-sm text-text-muted">
                RAPTOR tree was not available for this prepared TSD.
              </p>
            </Card>
          )}

          {(preparationDurationLabel || preparationLlmUsage) && (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[240px_minmax(0,1fr)]">
              {preparationDurationLabel && (
                <Card>
                  <p className="text-sm font-semibold text-text-primary">Preparation Duration</p>
                  <p className="mt-1 text-xs text-text-muted">End-to-end TSD ingestion and indexing time</p>
                  <div className="mt-4 text-2xl font-bold text-text-primary">{preparationDurationLabel}</div>
                </Card>
              )}
              {preparationLlmUsage && (
                <LlmUsageCard title="TSD Ingestion LLM Usage" usage={preparationLlmUsage} />
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'review_details' &&
        (selectedReview ? (
          <ReviewWorkspace
            key={selectedReview.id}
            reviewState={selectedReviewState}
            showControls={false}
          />
        ) : (
          <Card>
            <p className="py-8 text-center text-sm text-text-muted">
              No reviews yet for this design. Start a TSD analysis once preparation is ready.
            </p>
          </Card>
        ))}

      <CreateReviewModal
        open={showReviewModal}
        categories={categories}
        selectedCategory={selectedCat}
        analysisMode={analysisMode}
        creating={creating}
        onClose={() => setShowReviewModal(false)}
        onCategoryChange={setSelectedCat}
        onAnalysisModeChange={setAnalysisMode}
        onSubmit={() => void handleCreateReview()}
      />

      <DesignReingestModal
        open={showReingestModal}
        file={reingestFile}
        reingesting={reingesting}
        error={reingestError}
        onClose={() => setShowReingestModal(false)}
        onFileChange={setReingestFile}
        onSubmit={() => void handleReingest()}
      />
    </div>
  );
}
