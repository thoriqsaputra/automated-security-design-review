import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, Play, ArrowLeft } from 'lucide-react';
import Card from '../components/ui/Card';
import StatusBadge from '../components/ui/StatusBadge';
import Modal from '../components/ui/Modal';
import { getDesign, type DesignDetail } from '../api/designs';
import { listCategories, type StandardCategory } from '../api/standards';
import { createReview, listReviews, type Review } from '../api/reviews';

export default function DesignDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [design, setDesign] = useState<DesignDetail | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [categories, setCategories] = useState<StandardCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [selectedCat, setSelectedCat] = useState<number | ''>('');
  const [creating, setCreating] = useState(false);

  const [reviewsPage, setReviewsPage] = useState(1);
  const REVIEWS_PER_PAGE = 6;

  const load = () => {
    if (!id) return;
    setReviewsPage(1);
    return Promise.all([
      getDesign(Number(id)).then(r => setDesign(r.data)),
      listReviews(Number(id)).then(r => setReviews(r.data)),
      
      listCategories().then(r => setCategories(r.data)),
    ]).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [id]);

  const handleCreateReview = async () => {
    if (!selectedCat || !id) return;
    setCreating(true);
    try {
      const res = await createReview(Number(id), Number(selectedCat));
      setShowReviewModal(false);
      navigate(`/designs/${id}/reviews/${res.data.id}`);
    } finally {
      setCreating(false);
    }
  };

  if (loading || !design) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-flame border-t-transparent rounded-full animate-spin" />
      </div>
    );
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
            <p className="text-sm text-text-muted">{design.original_filename}</p>
          </div>
        </div>
        <button
          onClick={() => setShowReviewModal(true)}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          <Play size={16} /> New Review
        </button>
      </div>

      {/* Info Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {[
          { label: 'Status', value: <StatusBadge status={design.status} /> },
          { label: 'Created', value: new Date(design.created_at).toLocaleDateString() },
          { label: 'Reviews', value: reviews.length },
        ].map(item => (
          <Card key={item.label}>
            <p className="text-xs text-text-muted mb-1">{item.label}</p>
            <div className="text-sm font-medium text-text-primary">{item.value}</div>
          </Card>
        ))}
      </div>

      {/* Reviews */}
      <div>
        <h2 className="text-lg font-semibold text-text-primary mb-3">Reviews</h2>
        {reviews.length === 0 ? (
          <Card><p className="text-sm text-text-muted text-center py-4">No reviews yet for this design.</p></Card>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2">
              {paginatedReviews.map(r => (
                <Card key={r.id} hover onClick={() => navigate(`/designs/${id}/reviews/${r.id}`)}>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-text-primary">Review #{r.id}</p>
                      <p className="text-xs text-text-muted">{new Date(r.created_at).toLocaleDateString()}</p>
                    </div>
                    <StatusBadge status={r.status} />
                  </div>
                </Card>
              ))}
            </div>
            {totalReviewsPages > 1 && (
              <div className="flex items-center justify-center gap-2">
                <button
                  onClick={() => setReviewsPage(p => Math.max(1, p - 1))}
                  disabled={reviewsPage === 1}
                  className="px-3 py-1.5 text-xs font-medium bg-surface-base border border-surface-border rounded-lg disabled:opacity-50 text-text-primary hover:bg-surface-hover transition-colors"
                >
                  Previous
                </button>
                <span className="text-xs text-text-muted">Page {reviewsPage} of {totalReviewsPages}</span>
                <button
                  onClick={() => setReviewsPage(p => Math.min(totalReviewsPages, p + 1))}
                  disabled={reviewsPage === totalReviewsPages}
                  className="px-3 py-1.5 text-xs font-medium bg-surface-base border border-surface-border rounded-lg disabled:opacity-50 text-text-primary hover:bg-surface-hover transition-colors"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Review Modal */}
      <Modal open={showReviewModal} onClose={() => setShowReviewModal(false)} title="Create Security Review">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Standard Category</label>
            <select
              value={selectedCat}
              onChange={e => setSelectedCat(Number(e.target.value))}
              className="w-full px-3 py-2.5 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
            >
              <option value="">Select a category...</option>
              {categories.map(c => (
                <option key={c.id} value={c.id}>{c.name} ({c.code})</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleCreateReview}
            disabled={!selectedCat || creating}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
          >
            {creating ? 'Creating...' : 'Create Review'}
          </button>
        </div>
      </Modal>
    </div>
  );
}
