import Card from '../../../components/ui/Card';
import type { DebateStreamState } from '../../../api/reviews';
import AgentMessageBubble from './AgentMessageBubble';
import DebateStatusBadge from './DebateStatusBadge';

interface DebateLiveStreamProps {
  debate: DebateStreamState | null;
}

export default function DebateLiveStream({ debate }: DebateLiveStreamProps) {
  if (!debate) {
    return (
      <Card className="min-h-[320px]">
        <p className="text-sm text-text-muted">Select a debate card to inspect the live transcript.</p>
      </Card>
    );
  }

  return (
    <Card className="min-h-[320px]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">
            {debate.requirement_reference || debate.debate_id}
          </div>
          <h3 className="mt-2 text-lg font-semibold text-text-primary">
            {debate.requirement_text || 'Untitled requirement'}
          </h3>
          {debate.section_title && (
            <p className="mt-1 text-sm text-text-muted">{debate.section_title}</p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <DebateStatusBadge status={debate.status} />
          {debate.active_agent && (
            <span className="rounded-full border border-surface-border px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-text-secondary">
              {debate.active_agent}
            </span>
          )}
        </div>
      </div>

      <div className="mt-4 h-2 overflow-hidden rounded-full bg-midnight/60">
        <div
          className="h-full rounded-full bg-gradient-to-r from-flame to-amber-400 transition-[width] duration-300"
          style={{ width: `${Math.max(4, debate.progress_percent || 0)}%` }}
        />
      </div>

      <div className="mt-5 space-y-3">
        {debate.transcript.length ? (
          debate.transcript.map((message) => (
            <AgentMessageBubble key={message.message_id} message={message} />
          ))
        ) : (
          <div className="rounded-xl border border-dashed border-surface-border p-4 text-sm text-text-muted">
            No transcript has been emitted for this debate yet.
          </div>
        )}
      </div>
    </Card>
  );
}
