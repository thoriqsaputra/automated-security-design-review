import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, ListFilter } from 'lucide-react';
import type { ReviewAnalysisMode } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import Modal from '../../../components/ui/Modal';
import { useReviewDebateStream } from '../hooks/useReviewDebateStream';
import { isPresent } from '../utils/reviewPresentation';
import ActiveDebateCard from './ActiveDebateCard';
import DebateLiveStream from './DebateLiveStream';
import DebateStatusBadge from './DebateStatusBadge';

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
  } = props;
  const { debates, connected, streamError, errorMessage } = useReviewDebateStream(reviewId, reviewStatus, analysisMode);
  const [selectedDebateId, setSelectedDebateId] = useState<string | null>(null);
  const [isPickerOpen, setIsPickerOpen] = useState(false);

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

  const selectedIndex = selectedDebate ? debates.findIndex((debate) => debate.debate_id === selectedDebateId) : -1;

  const stepDebate = (offset: number) => {
    if (!debates.length || selectedIndex === -1) {
      return;
    }
    const nextIndex = (selectedIndex + offset + debates.length) % debates.length;
    setSelectedDebateId(debates[nextIndex].debate_id);
  };

  const showSummary =
    isPresent(debatedTotal)
    || isPresent(debatedProcessed)
    || isPresent(debatedRemaining)
    || isPresent(persistenceTotal)
    || isPresent(persistenceProcessed)
    || isPresent(persistenceRemaining);

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
                Debate counts reflect post-filtering requirement selection. Persistence tracks final write-out after debate completes.
              </p>
            </div>
            <div className="rounded-full border border-surface-border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-text-secondary">
              {connected ? 'Live' : 'Polling fallback'}
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-3">
            {[
              { label: 'Debated Total', value: debatedTotal ?? '—' },
              { label: 'Debated Completed', value: debatedProcessed ?? '—' },
              { label: 'Debated Remaining', value: debatedRemaining ?? '—' },
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

      {debates.length ? (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <button
                type="button"
                onClick={() => stepDebate(-1)}
                disabled={debates.length < 2}
                className="rounded-full border border-surface-border p-1.5 text-text-muted transition-colors hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Previous debate"
              >
                <ChevronLeft size={16} />
              </button>
              <div className="min-w-0">
                <div className="truncate text-xs font-semibold uppercase tracking-[0.18em] text-text-muted">
                  {selectedDebate ? (selectedDebate.requirement_reference || selectedDebate.debate_id) : 'No debate selected'}
                </div>
                <div className="mt-1 flex items-center gap-2">
                  {selectedDebate && <DebateStatusBadge status={selectedDebate.status} />}
                  {selectedDebate?.active_agent && (
                    <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-text-secondary">
                      {selectedDebate.active_agent}
                    </span>
                  )}
                </div>
              </div>
              <button
                type="button"
                onClick={() => stepDebate(1)}
                disabled={debates.length < 2}
                className="rounded-full border border-surface-border p-1.5 text-text-muted transition-colors hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
                aria-label="Next debate"
              >
                <ChevronRight size={16} />
              </button>
            </div>
            <button
              type="button"
              onClick={() => setIsPickerOpen(true)}
              className="flex items-center gap-2 rounded-full border border-surface-border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.16em] text-text-secondary transition-colors hover:text-text-primary"
            >
              <ListFilter size={14} />
              Switch debate
              {selectedIndex !== -1 && (
                <span className="text-text-muted">{selectedIndex + 1} of {debates.length}</span>
              )}
            </button>
          </div>
        </Card>
      ) : (
        <Card>
          <p className="text-sm text-text-muted">
            No debate events have been emitted yet. Once text-requirement analysis starts, active debates will appear here.
          </p>
        </Card>
      )}

      <DebateLiveStream debate={selectedDebate} />

      <Modal open={isPickerOpen} onClose={() => setIsPickerOpen(false)} title="Switch debate">
        <div className="max-h-[70vh] space-y-3 overflow-y-auto">
          {debates.map((debate) => (
            <ActiveDebateCard
              key={debate.debate_id}
              debate={debate}
              selected={debate.debate_id === selectedDebateId}
              onSelect={() => {
                setSelectedDebateId(debate.debate_id);
                setIsPickerOpen(false);
              }}
            />
          ))}
        </div>
      </Modal>
    </div>
  );
}
