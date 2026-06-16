import { useCallback, useEffect, useState } from 'react';
import {
  cancelReview,
  getFindings,
  getRetrievalVisualization,
  getReview,
  triggerReview,
  type Finding,
  type RetrievalVisualization,
  type Review,
  type ReviewAnalysisMode,
} from '../../../api/reviews';
import {
  isRecord,
  normalizeAnalysisMode,
  type ReviewTab,
} from '../utils/reviewPresentation';

interface UseReviewDetailResult {
  review: Review | null;
  findings: Finding[];
  retrievalVisualization: RetrievalVisualization | null;
  loading: boolean;
  loadingFindings: boolean;
  triggering: boolean;
  cancelling: boolean;
  selectedAnalysisMode: ReviewAnalysisMode;
  setSelectedAnalysisMode: (mode: ReviewAnalysisMode) => void;
  expandedFinding: number | null;
  setExpandedFinding: (findingId: number | null) => void;
  activeTab: ReviewTab;
  setActiveTab: (tab: ReviewTab) => void;
  search: string;
  searchInput: string;
  setSearchInput: (value: string) => void;
  commitSearch: () => void;
  clearFilters: () => void;
  filterMetStatus: string;
  setFilterMetStatus: (value: string) => void;
  filterSeverity: string;
  setFilterSeverity: (value: string) => void;
  filterFindingType: string;
  setFilterFindingType: (value: string) => void;
  currentPage: number;
  setCurrentPage: (page: number) => void;
  pageSize: number;
  setPageSize: (pageSize: number) => void;
  totalFindings: number;
  totalPages: number;
  currentStage?: string;
  debatedTotal: number | null;
  debatedProcessed: number | null;
  debatedRemaining: number | null;
  persistenceTotal: number | null;
  persistenceProcessed: number | null;
  persistenceRemaining: number | null;
  skippedByParentApplicability: number | null;
  reload: () => Promise<void>;
  handleTrigger: () => Promise<void>;
  handleCancel: () => Promise<void>;
}

export function useReviewDetail(reviewId?: string): UseReviewDetailResult {
  const [review, setReview] = useState<Review | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [retrievalVisualization, setRetrievalVisualization] = useState<RetrievalVisualization | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingFindings, setLoadingFindings] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [selectedAnalysisMode, setSelectedAnalysisMode] = useState<ReviewAnalysisMode>('default');
  const [expandedFinding, setExpandedFinding] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<ReviewTab>('overview');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [filterMetStatus, setFilterMetStatus] = useState('all');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterFindingType, setFilterFindingType] = useState('all');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalFindings, setTotalFindings] = useState(0);
  const [totalPages, setTotalPages] = useState(1);

  const loadFindings = useCallback(async () => {
    if (!reviewId) {
      return;
    }

    setLoadingFindings(true);
    try {
      const response = await getFindings(
        Number(reviewId),
        currentPage,
        pageSize,
        search || undefined,
        filterMetStatus !== 'all' ? filterMetStatus : undefined,
        filterSeverity !== 'all' ? filterSeverity : undefined,
        filterFindingType !== 'all' ? filterFindingType : undefined,
      );
      setFindings(response.data.items);
      setTotalFindings(response.data.total);
      setTotalPages(response.data.total_pages);
    } catch {
      setFindings([]);
    } finally {
      setLoadingFindings(false);
    }
  }, [currentPage, filterFindingType, filterMetStatus, filterSeverity, pageSize, reviewId, search]);

  useEffect(() => {
    queueMicrotask(() => {
      void loadFindings();
    });
  }, [loadFindings]);

  const reload = useCallback(async () => {
    if (!reviewId) {
      return;
    }

    setLoading(true);
    try {
      const [reviewResponse, retrievalResponse] = await Promise.all([
        getReview(Number(reviewId)),
        getRetrievalVisualization(Number(reviewId)).catch(() => null),
      ]);
      const nextReview = {
        ...reviewResponse.data,
        analysis_mode: normalizeAnalysisMode(reviewResponse.data.analysis_mode),
      };
      setReview(nextReview);
      setSelectedAnalysisMode(nextReview.analysis_mode);
      setRetrievalVisualization(retrievalResponse?.data || null);
    } finally {
      setLoading(false);
    }
  }, [reviewId]);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    if (review?.status !== 'running') {
      return;
    }

    const timer = setInterval(() => {
      void reload();
      void loadFindings();
    }, 8000);

    return () => clearInterval(timer);
  }, [loadFindings, reload, review?.status]);

  const commitSearch = useCallback(() => {
    setCurrentPage(1);
    setSearch(searchInput);
  }, [searchInput]);

  const clearFilters = useCallback(() => {
    setCurrentPage(1);
    setSearch('');
    setSearchInput('');
    setFilterMetStatus('all');
    setFilterSeverity('all');
    setFilterFindingType('all');
  }, []);

  const handleTrigger = useCallback(async () => {
    if (!reviewId || triggering) {
      return;
    }

    setTriggering(true);
    try {
      await triggerReview(Number(reviewId), selectedAnalysisMode);
    } finally {
      await reload();
      setTriggering(false);
    }
  }, [reload, reviewId, selectedAnalysisMode, triggering]);

  const handleCancel = useCallback(async () => {
    if (!reviewId || cancelling) {
      return;
    }

    setCancelling(true);
    try {
      await cancelReview(Number(reviewId));
    } finally {
      await reload();
      setCancelling(false);
    }
  }, [cancelling, reload, reviewId]);

  const summary = isRecord(review?.summary_json) ? review.summary_json : {};
  const progress = isRecord(review?.progress) ? (review.progress as Record<string, unknown>) : null;
  const livePreparation = progress && isRecord(progress.preparation) ? progress.preparation : null;
  const liveDebate = livePreparation && isRecord(livePreparation.debate) ? livePreparation.debate : null;
  const livePersistence = livePreparation && isRecord(livePreparation.persistence) ? livePreparation.persistence : null;
  const applicabilitySummary = isRecord(summary.applicability) ? summary.applicability : {};

  const debatedTotal =
    typeof liveDebate?.total === 'number'
      ? liveDebate.total
      : typeof summary.debate_total_parameters === 'number'
        ? summary.debate_total_parameters
        : typeof summary.analysis_total_parameters === 'number'
          ? summary.analysis_total_parameters
          : null;

  const debatedProcessed =
    typeof liveDebate?.completed === 'number'
      ? liveDebate.completed
      : typeof summary.debate_completed_parameters === 'number'
        ? summary.debate_completed_parameters
        : typeof summary.analysis_processed_parameters === 'number'
          ? summary.analysis_processed_parameters
          : null;

  const debatedRemaining =
    typeof liveDebate?.remaining === 'number'
      ? liveDebate.remaining
      : typeof summary.debate_remaining_parameters === 'number'
        ? summary.debate_remaining_parameters
        : typeof summary.analysis_remaining_parameters === 'number'
          ? summary.analysis_remaining_parameters
          : null;

  const persistenceTotal =
    typeof livePersistence?.total === 'number'
      ? livePersistence.total
      : typeof summary.persistence_total_parameters === 'number'
        ? summary.persistence_total_parameters
        : null;

  const persistenceProcessed =
    typeof livePersistence?.completed === 'number'
      ? livePersistence.completed
      : typeof summary.persistence_completed_parameters === 'number'
        ? summary.persistence_completed_parameters
        : null;

  const persistenceRemaining =
    typeof livePersistence?.remaining === 'number'
      ? livePersistence.remaining
      : typeof summary.persistence_remaining_parameters === 'number'
        ? summary.persistence_remaining_parameters
        : null;

  const skippedByParentApplicability =
    typeof livePreparation?.skipped_by_parent_applicability === 'number'
      ? livePreparation.skipped_by_parent_applicability
      : typeof applicabilitySummary.children_marked_na_by_parent === 'number'
        ? applicabilitySummary.children_marked_na_by_parent
        : null;

  const currentStage =
    isRecord(review?.summary_json) && typeof review.summary_json.current_stage === 'string'
      ? review.summary_json.current_stage
      : undefined;

  return {
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
    reload,
    handleTrigger,
    handleCancel,
  };
}
