import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Check, ChevronDown, FileText, Play, Search } from 'lucide-react';
import type { JsonRecord } from '../api/reviews';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Card from '../components/ui/Card';
import StatusBadge from '../components/ui/StatusBadge';
import { getDesign, type DesignDetail } from '../api/designs';
import { listCategories, type StandardCategory } from '../api/standards';
import {
  createReview,
  listReviews,
  type Review,
  type ReviewAnalysisMode,
} from '../api/reviews';
import CreateReviewModal from '../features/designs/components/CreateReviewModal';
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
  const [pickerOpen, setPickerOpen] = useState(false);
  const [reviewSearch, setReviewSearch] = useState('');

  const load = useCallback(async () => {
    if (!id) return;
    setLoading(true);
    return Promise.all([
      getDesign(Number(id)).then(r => setDesign(r.data)),
      listReviews(Number(id), { limit: REVIEWS_FETCH_LIMIT }).then(r => setReviews(r.data)),
      listCategories().then(r => setCategories(r.data)),
    ]).finally(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleCreateReview = async () => {
    if (!selectedCat || !id) return;
    setCreating(true);
    try {
      const res = await createReview(Number(id), Number(selectedCat), analysisMode);
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

  const hasControls = Boolean(
    selectedReview &&
      ['pending', 'cancelled', 'failed', 'running'].includes(selectedReviewState.review?.status || '')
  );

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
          </div>
        </div>
      </div>

      <Card className="overflow-visible">
        <div className={`flex flex-col gap-4 ${hasControls ? 'lg:flex-row lg:items-start lg:justify-between' : ''}`}>
          {hasControls && (
            <div>
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
                  disabled={selectedReviewState.triggering}
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
          <div className={`relative w-full ${hasControls ? 'max-w-xl' : ''}`}>
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
                    <p className="mt-1 truncate text-xs text-text-muted">
                      Create a review to inspect retrieval, debate, and findings.
                    </p>
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
                  <button
                    onClick={() => {
                      setPickerOpen(false);
                      setShowReviewModal(true);
                    }}
                    className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-crimson to-flame px-4 py-2.5 text-sm font-semibold text-white transition-all hover:shadow-lg hover:shadow-crimson/30"
                  >
                    <Play size={16} /> New Review
                  </button>
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
      </Card>

      {selectedReview ? (
        <>
          <ReviewWorkspace
            key={selectedReview.id}
            reviewState={selectedReviewState}
            showControls={false}
          />
        </>
      ) : (
        <Card>
          <p className="py-8 text-center text-sm text-text-muted">
            No reviews yet for this design. Create one to start the analysis workflow.
          </p>
        </Card>
      )}

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
    </div>
  );
}
