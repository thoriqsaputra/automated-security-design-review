import type { DebateTranscriptMessage } from '../../../api/reviews';

const agentStyles: Record<string, string> = {
  hunter: 'border-sky-500/30 bg-sky-500/10',
  critic: 'border-amber-500/30 bg-amber-500/10',
  mediator: 'border-emerald-500/30 bg-emerald-500/10',
  system: 'border-surface-border bg-midnight/40',
};

const outcomeStyles: Record<string, string> = {
  UPHOLD: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300',
  OVERTURN: 'border-flame/30 bg-flame/10 text-flame',
  PARTIAL: 'border-amber-500/30 bg-amber-500/10 text-amber-300',
};

interface AgentMessageBubbleProps {
  message: DebateTranscriptMessage;
}

export default function AgentMessageBubble({ message }: AgentMessageBubbleProps) {
  const style = agentStyles[message.agent] || agentStyles.system;
  return (
    <div className={`rounded-xl border p-4 ${style}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">
            {message.agent}
          </div>
          {message.content && (
            <span className="rounded-full border border-surface-border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
              Chain of thought
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {message.critic_outcome && (
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${outcomeStyles[message.critic_outcome] || outcomeStyles.UPHOLD}`}
            >
              {message.requires_rebuttal ? 'Needs rebuttal' : message.critic_outcome}
            </span>
          )}
          <div className="text-[11px] text-text-muted">{message.status.replace(/_/g, ' ')}</div>
        </div>
      </div>
      <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-text-primary">
        {message.content || 'Waiting for output...'}
      </p>
    </div>
  );
}
