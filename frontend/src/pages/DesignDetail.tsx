import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, Play, ArrowLeft } from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { getDesign, type DesignDetail } from '../api/designs';
import { listCategories, type StandardCategory } from '../api/standards';
import {
  createReview,
  listReviews,
  type Review,
  type ReviewAnalysisMode,
} from '../api/reviews';
import CreateReviewModal from '../features/designs/components/CreateReviewModal';
import DesignReviewsSection from '../features/designs/components/DesignReviewsSection';

export default function DesignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [design, setDesign] = useState<DesignDetail | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [categories, setCategories] = useState<StandardCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [selectedCat, setSelectedCat] = useState<number | ''>('');
  const [analysisMode, setAnalysisMode] = useState<ReviewAnalysisMode>('default');
  const [creating, setCreating] = useState(false);

  const [reviewsPage, setReviewsPage] = useState(1);
  const REVIEWS_PER_PAGE = 6;

  const load = useCallback(() => {
    if (!id) return;
    return Promise.all([
      getDesign(Number(id)).then(r => setDesign(r.data)),
      listReviews(Number(id)).then(r => setReviews(r.data)),
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
      const res = await createReview(Number(id), Number(selectedCat), null, analysisMode);
      setShowReviewModal(false);
      setSelectedCat('');
      setAnalysisMode('default');
      navigate(`/designs/${id}/reviews/${res.data.id}`);
    } finally {
      setCreating(false);
    }
  };

  if (loading || !design) {
    return <LoadingSpinner />;
  }

  const paginatedReviews = reviews.slice((reviewsPage - 1) * REVIEWS_PER_PAGE, reviewsPage * REVIEWS_PER_PAGE);
  const totalReviewsPages = Math.ceil(reviews.length / REVIEWS_PER_PAGE);

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
        <button
          onClick={() => setShowReviewModal(true)}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          <Play size={16} /> New Review
        </button>
      </div>

      {/* Reviews */}
      <DesignReviewsSection
        designId={id || ''}
        reviews={paginatedReviews}
        reviewsPage={reviewsPage}
        totalReviewsPages={totalReviewsPages}
        onNavigate={navigate}
        onPageChange={setReviewsPage}
      />

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
