import { XCircle } from 'lucide-react';
import type { Review } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import LlmUsageCard, { formatDuration } from '../../../components/ui/LlmUsageCard';
import { formatAnalysisModeLabel, isRecord, normalizeAnalysisMode } from '../utils/reviewPresentation';

interface ReviewOverviewPanelProps {
  review: Review;
  findingCount: number;
}

function reviewDurationLabel(review: Review): string {
  if (!review.started_at) {
    return '—';
  }
  const start = new Date(review.started_at).getTime();
  const end = review.completed_at ? new Date(review.completed_at).getTime() : Date.now();
  const totalSeconds = (end - start) / 1000;
  return totalSeconds > 0 ? formatDuration(totalSeconds) : '—';
}

export default function ReviewOverviewPanel({ review, findingCount }: ReviewOverviewPanelProps) {
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        {[
          { label: 'Analysis Mode', value: formatAnalysisModeLabel(normalizeAnalysisMode(review.analysis_mode)) },
          { label: 'Findings', value: findingCount },
          { label: 'Duration', value: reviewDurationLabel(review) },
          { label: 'Started', value: review.started_at ? new Date(review.started_at).toLocaleString() : '—' },
          { label: 'Completed', value: review.completed_at ? new Date(review.completed_at).toLocaleString() : '—' },
        ].map((item) => (
          <Card key={item.label}>
            <p className="text-xs text-text-muted mb-1">{item.label}</p>
            <div className="text-sm font-medium text-text-primary">{item.value}</div>
          </Card>
        ))}
      </div>

      <LlmUsageCard
        title="LLM Usage (Review)"
        usage={isRecord(review.summary_json.llm_usage) ? review.summary_json.llm_usage : null}
      />

      {review.overview && (
        <Card>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Overview</h3>
          <p className="text-sm text-text-secondary leading-relaxed">{review.overview}</p>
        </Card>
      )}

      {review.error_message && (
        <Card className="border-crimson/30">
          <div className="flex items-start gap-2">
            <XCircle size={16} className="text-crimson mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-medium text-crimson">Error</p>
              <p className="text-sm text-text-secondary mt-1">{review.error_message}</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
