import { useEffect, useMemo, useState } from 'react';
import type { ReviewAnalysisMode } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import { useReviewDebateStream } from '../hooks/useReviewDebateStream';
import { isPresent } from '../utils/reviewPresentation';
import ActiveDebateCard from './ActiveDebateCard';
import DebateLiveStream from './DebateLiveStream';

interface ReviewDebatePanelProps {
  reviewId: number;
  reviewStatus: string;
  analysisMode: ReviewAnalysisMode;
  debatedTotal: number | null;
  debatedProcessed: number | null;
  debatedRemaining: number | null;
  persistenceTotal: number | null;
  persistenceProcessed: number | null;
  persistenceRemaining: number | null;
  skippedByParentApplicability: number | null;
}

export default function ReviewDebatePanel(props: ReviewDebatePanelProps) {
  const {
    reviewId,
    reviewStatus,
    analysisMode,
    debatedTotal,
    debatedProcessed,
    debatedRemaining,
    persistenceTotal,
    persistenceProcessed,
    persistenceRemaining,
    skippedByParentApplicability,
  } = props;
  const { debates, connected, streamError, errorMessage } = useReviewDebateStream(reviewId, reviewStatus, analysisMode);
  const [selectedDebateId, setSelectedDebateId] = useState<string | null>(null);

  useEffect(() => {
    if (!debates.length) {
      setSelectedDebateId(null);
      return;
    }
    const stillExists = selectedDebateId && debates.some((debate) => debate.debate_id === selectedDebateId);
    if (stillExists) {
      return;
    }
    setSelectedDebateId(debates[0].debate_id);
  }, [debates, selectedDebateId]);

  const selectedDebate = useMemo(
    () => debates.find((debate) => debate.debate_id === selectedDebateId) || null,
    [debates, selectedDebateId],
  );

  const showSummary =
    isPresent(debatedTotal)
    || isPresent(debatedProcessed)
    || isPresent(debatedRemaining)
    || isPresent(persistenceTotal)
    || isPresent(persistenceProcessed)
    || isPresent(persistenceRemaining)
    || isPresent(skippedByParentApplicability);

  if (analysisMode === 'diagram_only') {
    return (
      <div className="space-y-6 animate-fade-in">
        <Card>
          <p className="text-sm text-text-muted">
            Live debate monitoring is currently available only for text-based requirements.
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {showSummary && (
        <Card>
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-text-primary">Analysis Progress</h3>
              <p className="mt-1 text-xs text-text-muted">
                Debate counts reflect post-ASVS and post-parent-applicability children. Persistence tracks final write-out after debate completes.
              </p>
            </div>
            <div className="rounded-full border border-surface-border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
              {connected ? 'Live' : 'Polling fallback'}
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
            {[
              { label: 'Debated Total', value: debatedTotal ?? '—' },
              { label: 'Debated Completed', value: debatedProcessed ?? '—' },
              { label: 'Debated Remaining', value: debatedRemaining ?? '—' },
              { label: 'Skipped By Parent Applicability', value: skippedByParentApplicability ?? '—' },
            ].map((item) => (
              <div key={item.label} className="rounded-lg border border-surface-border bg-midnight/30 p-3">
                <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
                <p className="mt-1 text-lg font-semibold text-text-primary">{item.value}</p>
              </div>
            ))}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-3">
            {[
              { label: 'Persistence Total', value: persistenceTotal ?? '—' },
              { label: 'Persistence Completed', value: persistenceProcessed ?? '—' },
              { label: 'Persistence Remaining', value: persistenceRemaining ?? '—' },
            ].map((item) => (
              <div key={item.label} className="rounded-lg border border-surface-border bg-midnight/30 p-3">
                <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
                <p className="mt-1 text-lg font-semibold text-text-primary">{item.value}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {(streamError || errorMessage) && (
        <Card className="border-amber-500/30 bg-amber-500/10">
          <p className="text-sm text-amber-300">{streamError || errorMessage}</p>
        </Card>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-text-primary">Active And Completed Debates</h3>
            <div className="text-xs text-text-muted">{debates.length} tracked debate{debates.length === 1 ? '' : 's'}</div>
          </div>
          {debates.length ? (
            <div className="grid gap-4 md:grid-cols-2">
              {debates.map((debate) => (
                <ActiveDebateCard
                  key={debate.debate_id}
                  debate={debate}
                  selected={debate.debate_id === selectedDebateId}
                  onSelect={() => setSelectedDebateId(debate.debate_id)}
                />
              ))}
            </div>
          ) : (
            <Card>
              <p className="text-sm text-text-muted">
                No debate events have been emitted yet. Once text-requirement analysis starts, active debate cards will appear here.
              </p>
            </Card>
          )}
        </div>

        <DebateLiveStream debate={selectedDebate} />
      </div>
    </div>
  );
}
