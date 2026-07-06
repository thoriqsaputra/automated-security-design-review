import Card from './Card';

export type JsonRecord = Record<string, unknown>;

interface LlmUsageCardProps {
  title?: string;
  usage: JsonRecord | null | undefined;
}

function formatDuration(totalSeconds: number): string {
  if (!totalSeconds || totalSeconds <= 0) {
    return '0s';
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  if (minutes <= 0) {
    return `${seconds}s`;
  }
  return `${minutes}m ${seconds}s`;
}

function formatCount(value: unknown): string {
  const num = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(num) ? num.toLocaleString() : '—';
}

/**
 * Renders the {call_count, prompt_tokens, completion_tokens, total_tokens,
 * duration_seconds, error_count} shape persisted by the backend's
 * usage_tracker into summary_json/stats_json.llm_usage. Shared between the
 * review overview panel and the standard ingestion job list since both
 * pipelines persist the exact same shape.
 */
export default function LlmUsageCard({ title = 'LLM Usage', usage }: LlmUsageCardProps) {
  if (!usage || typeof usage !== 'object') {
    return null;
  }
  const callCount = Number(usage.call_count) || 0;
  if (callCount === 0) {
    return null;
  }
  const durationSeconds = Number(usage.duration_seconds) || 0;
  const errorCount = Number(usage.error_count) || 0;

  const stats = [
    { label: 'Calls', value: formatCount(usage.call_count) },
    { label: 'Prompt Tokens', value: formatCount(usage.prompt_tokens) },
    { label: 'Completion Tokens', value: formatCount(usage.completion_tokens) },
    { label: 'Total Tokens', value: formatCount(usage.total_tokens) },
    { label: 'Cumulative LLM Call Time', value: formatDuration(durationSeconds) },
  ];

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-text-primary">{title}</h3>
        {errorCount > 0 && (
          <span className="text-xs text-crimson">{errorCount} failed call{errorCount === 1 ? '' : 's'}</span>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        {stats.map((item) => (
          <div key={item.label} className="bg-midnight/30 p-3 rounded-lg border border-surface-border">
            <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
            <div className="mt-1 text-sm font-medium text-text-primary">{item.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}
