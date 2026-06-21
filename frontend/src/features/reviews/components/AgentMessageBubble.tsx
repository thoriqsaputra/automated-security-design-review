import type { DebateTranscriptMessage } from '../../../api/reviews';

const agentStyles: Record<string, string> = {
  hunter: 'border-sky-500/30 bg-sky-500/10',
  critic: 'border-amber-500/30 bg-amber-500/10',
  mediator: 'border-emerald-500/30 bg-emerald-500/10',
  system: 'border-surface-border bg-midnight/40',
};

interface AgentMessageBubbleProps {
  message: DebateTranscriptMessage;
}

export default function AgentMessageBubble({ message }: AgentMessageBubbleProps) {
  const style = agentStyles[message.agent] || agentStyles.system;
  return (
    <div className={`rounded-xl border p-4 ${style}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">
          {message.agent}
        </div>
        <div className="text-[11px] text-text-muted">{message.status.replace(/_/g, ' ')}</div>
      </div>
      <p className="mt-3 whitespace-pre-wrap break-words text-sm leading-6 text-text-primary">
        {message.content || 'Waiting for output...'}
      </p>
    </div>
  );
}
