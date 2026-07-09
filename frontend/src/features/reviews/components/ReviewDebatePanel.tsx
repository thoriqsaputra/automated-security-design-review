import { useEffect, useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, Filter, ListFilter } from 'lucide-react';
import type { DebateStreamState, ReviewAnalysisMode } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import Modal from '../../../components/ui/Modal';
import { isPresent } from '../utils/reviewPresentation';
import ActiveDebateCard from './ActiveDebateCard';
import DebateLiveStream from './DebateLiveStream';
import DebateStatusBadge from './DebateStatusBadge';

type FindingTypeFilter = 'all' | 'requirement' | 'diagram';
type CriticStatusFilter = 'all' | 'needs_rebuttal' | 'upheld';

const selectArrowStyle = {
  backgroundImage:
    "url(\"data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e\")",
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.25rem center',
  backgroundSize: '1em',
} as const;

interface ReviewDebatePanelProps {
  analysisMode: ReviewAnalysisMode;
  debates: DebateStreamState[];
  connected: boolean;
  streamError: string | null;
  errorMessage: string | null;
  debatedTotal: number | null;
  debatedProcessed: number | null;
  debatedRemaining: number | null;
  persistenceTotal: number | null;
  persistenceProcessed: number | null;
  persistenceRemaining: number | null;
}

export default function ReviewDebatePanel(props: ReviewDebatePanelProps) {
  const {
    debates,
    connected,
    streamError,
    errorMessage,
    debatedTotal,
    debatedProcessed,
    debatedRemaining,
    persistenceTotal,
    persistenceProcessed,
    persistenceRemaining,
  } = props;
  const [selectedDebateId, setSelectedDebateId] = useState<string | null>(null);
  const [isPickerOpen, setIsPickerOpen] = useState(false);
  const [findingTypeFilter, setFindingTypeFilter] = useState<FindingTypeFilter>('all');
  const [criticStatusFilter, setCriticStatusFilter] = useState<CriticStatusFilter>('all');

  const filteredDebates = useMemo(
    () =>
      debates.filter((debate) => {
        if (findingTypeFilter !== 'all' && debate.finding_type !== findingTypeFilter) {
          return false;
        }
        // A debate "needs rebuttal" if any round's critic pushed back, not just
        // the final one — earlier rounds can be OVERTURN/PARTIAL even after the
        // debate ultimately resolves to UPHOLD in a later round.
        const everNeededRebuttal =
          debate.requires_rebuttal || debate.transcript.some((message) => message.requires_rebuttal);
        if (criticStatusFilter === 'needs_rebuttal' && !everNeededRebuttal) {
          return false;
        }
        if (criticStatusFilter === 'upheld' && (everNeededRebuttal || !debate.critic_outcome)) {
          return false;
        }
        return true;
      }),
    [debates, findingTypeFilter, criticStatusFilter],
  );

  useEffect(() => {
    if (!filteredDebates.length) {
      setSelectedDebateId(null);
      return;
    }
    const stillExists = selectedDebateId && filteredDebates.some((debate) => debate.debate_id === selectedDebateId);
    if (stillExists) {
      return;
    }
    setSelectedDebateId(filteredDebates[0].debate_id);
  }, [filteredDebates, selectedDebateId]);

  const selectedDebate = useMemo(
    () => filteredDebates.find((debate) => debate.debate_id === selectedDebateId) || null,
    [filteredDebates, selectedDebateId],
  );

  const selectedIndex = selectedDebate ? filteredDebates.findIndex((debate) => debate.debate_id === selectedDebateId) : -1;

  const stepDebate = (offset: number) => {
    if (!filteredDebates.length || selectedIndex === -1) {
      return;
    }
    const nextIndex = (selectedIndex + offset + filteredDebates.length) % filteredDebates.length;
    setSelectedDebateId(filteredDebates[nextIndex].debate_id);
  };

  const showSummary =
    isPresent(debatedTotal)
    || isPresent(debatedProcessed)
    || isPresent(debatedRemaining)
    || isPresent(persistenceTotal)
    || isPresent(persistenceProcessed)
    || isPresent(persistenceRemaining);

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

      {debates.length > 0 && (
        <Card>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1.5 rounded-lg border border-surface-border bg-midnight px-1 py-1">
              <Filter size={14} className="ml-2 text-text-muted" />
              <select
                value={findingTypeFilter}
                onChange={(event) => setFindingTypeFilter(event.target.value as FindingTypeFilter)}
                className="cursor-pointer appearance-none bg-transparent py-1 pr-6 text-sm text-text-primary focus:outline-none"
                style={selectArrowStyle}
              >
                <option value="all">Type: All</option>
                <option value="requirement">Type: Requirement</option>
                <option value="diagram">Type: Diagram</option>
              </select>
            </div>
            <div className="flex items-center gap-1.5 rounded-lg border border-surface-border bg-midnight px-1 py-1">
              <Filter size={14} className="ml-2 text-text-muted" />
              <select
                value={criticStatusFilter}
                onChange={(event) => setCriticStatusFilter(event.target.value as CriticStatusFilter)}
                className="cursor-pointer appearance-none bg-transparent py-1 pr-6 text-sm text-text-primary focus:outline-none"
                style={selectArrowStyle}
              >
                <option value="all">Critic: All</option>
                <option value="needs_rebuttal">Critic: Needs rebuttal</option>
                <option value="upheld">Critic: Upheld</option>
              </select>
            </div>
            {(findingTypeFilter !== 'all' || criticStatusFilter !== 'all') && (
              <button
                type="button"
                onClick={() => {
                  setFindingTypeFilter('all');
                  setCriticStatusFilter('all');
                }}
                className="text-xs font-semibold text-text-muted transition-colors hover:text-text-primary"
              >
                Clear filters
              </button>
            )}
          </div>
        </Card>
      )}

      {filteredDebates.length ? (
        <Card>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <button
                type="button"
                onClick={() => stepDebate(-1)}
                disabled={filteredDebates.length < 2}
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
                disabled={filteredDebates.length < 2}
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
                <span className="text-text-muted">{selectedIndex + 1} of {filteredDebates.length}</span>
              )}
            </button>
          </div>
        </Card>
      ) : (
        <Card>
          <p className="text-sm text-text-muted">
            {debates.length
              ? 'No debates match the current filters.'
              : 'No debate events have been emitted yet. Once requirement or diagram analysis starts, active debates will appear here.'}
          </p>
        </Card>
      )}

      <DebateLiveStream debate={selectedDebate} />

      <Modal open={isPickerOpen} onClose={() => setIsPickerOpen(false)} title="Switch debate">
        <div className="max-h-[70vh] space-y-3 overflow-y-auto">
          {filteredDebates.map((debate) => (
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
