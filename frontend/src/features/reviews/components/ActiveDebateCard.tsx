import Card from '../../../components/ui/Card';
import type { DebateStreamState } from '../../../api/reviews';
import DebateStatusBadge from './DebateStatusBadge';

interface ActiveDebateCardProps {
  debate: DebateStreamState;
  selected: boolean;
  onSelect: () => void;
}

export default function ActiveDebateCard({ debate, selected, onSelect }: ActiveDebateCardProps) {
  return (
    <Card
      hover
      onClick={onSelect}
      className={selected ? 'border-flame shadow-lg shadow-flame/10' : ''}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-text-muted">
            {debate.requirement_reference || debate.debate_id}
          </div>
          <h3 className="mt-2 text-sm font-semibold leading-6 text-text-primary">
            {debate.requirement_text || 'Untitled requirement'}
          </h3>
        </div>
        <DebateStatusBadge status={debate.status} />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.18em] text-text-secondary">
        <span className="rounded-full border border-surface-border px-2 py-0.5 text-[10px]">
          {debate.finding_type === 'diagram' ? 'Diagram' : 'Requirement'}
        </span>
        <span>{debate.execution_mode}</span>
        {debate.active_agent && <span>· {debate.active_agent}</span>}
        {debate.requires_rebuttal && (
          <span className="rounded-full border border-flame/30 bg-flame/10 px-2 py-0.5 text-[10px] text-flame">
            Needs rebuttal
          </span>
        )}
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-midnight/60">
        <div
          className="h-full rounded-full bg-gradient-to-r from-flame to-amber-400 transition-[width] duration-300"
          style={{ width: `${Math.max(4, debate.progress_percent || 0)}%` }}
        />
      </div>

      <p className="mt-4 max-h-[4.5rem] overflow-hidden text-sm leading-6 text-text-muted">
        {debate.last_snippet || 'Waiting for the next debate update...'}
      </p>
    </Card>
  );
}
