import type { Review } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import PaginationControls from '../../../components/ui/PaginationControls';
import StatusBadge from '../../../components/ui/StatusBadge';

interface DesignReviewsSectionProps {
  designId: string;
  reviews: Review[];
  reviewsPage: number;
  totalReviewsPages: number;
  onNavigate: (url: string) => void;
  onPageChange: (page: number) => void;
}

export default function DesignReviewsSection({
  designId,
  reviews,
  reviewsPage,
  totalReviewsPages,
  onNavigate,
  onPageChange,
}: DesignReviewsSectionProps) {
  return (
    <div>
      <h2 className="text-lg font-semibold text-text-primary mb-3">Reviews</h2>
      {reviews.length === 0 ? (
        <Card>
          <p className="text-sm text-text-muted text-center py-4">No reviews yet for this design.</p>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {reviews.map((review) => (
              <Card key={review.id} hover onClick={() => onNavigate(`/designs/${designId}/reviews/${review.id}`)}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-text-primary">Review #{review.id}</p>
                    <p className="text-xs text-text-muted">{new Date(review.created_at).toLocaleDateString()}</p>
                  </div>
                  <StatusBadge status={review.status} />
                </div>
              </Card>
            ))}
          </div>
          <PaginationControls currentPage={reviewsPage} totalPages={totalReviewsPages} onPageChange={onPageChange} />
        </div>
      )}
    </div>
  );
}
